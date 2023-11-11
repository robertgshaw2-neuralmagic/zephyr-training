import torch
from composer.core import Event, Algorithm
from composer import Callback, State, Logger

import torch.nn.functional as TF
from composer.trainer.dist_strategy import prepare_fsdp_module
from composer.utils import get_device

####################################################################################################################################################################################
# utilities for printing sparsity levels

def print_layer_sparsity(model: torch.nn.Module) -> None:
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear):
            mask = torch.where(module.weight == 0, torch.tensor(0, dtype=torch.uint8), torch.tensor(1, dtype=torch.uint8))
            print(f"[{name} sparsity = {torch.sum(mask == 0)/mask.numel()}]")
        else:
            print_layer_sparsity(module)

class PrintSparsityCallback(Callback):
    def epoch_end(self, state: State, logger: Logger):
        with torch.no_grad():
            print_layer_sparsity(state.model)

####################################################################################################################################################################################


####################################################################################################################################################################################
# equivalent to ConstantPruningModifier
# https://github.com/IST-DASLab/SparseFinetuning/blob/main/scripts/train/train_sparse.py#L33

class MaskPrunedWeights(Algorithm):
    def match(self, event, state):
        # masking weights after optimizer step should be sufficient
        # if we detect weird behaviour, we can also do it before the forward pass
        # by adding `or event == Event.BATCH_START`
        return event == Event.BATCH_END

    @torch.no_grad()
    def apply(self, event, state, logger):
        def mask_weights(module):
            if hasattr(module, 'mask'):
                module.weight *= module.mask

        state.model.apply(mask_weights)

def attach_masks(model, to_layer):
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear) and torch.sum(module.weight == 0)/module.weight.numel() >= 0.1:
            mask = torch.where(module.weight == 0, torch.tensor(0, dtype=torch.uint8), torch.tensor(1, dtype=torch.uint8))
            module.register_buffer("mask", mask, persistent=False)
            print(f"[Debugging] attaching mask to {name} with sparsity = {torch.sum(mask == 0)/mask.numel()}")
        else:
            attach_masks(module, to_layer)

####################################################################################################################################################################################

####################################################################################################################################################################################
# squarehead loss
# https://github.com/IST-DASLab/SparseFinetuning/blob/main/scripts/train/train_sparse.py#L49C1-L127C52

def kldiv_loss(student_logits, teacher_logits, temperature):
    "Kullback-Leibler divergence loss"
    num_tokens = student_logits.numel() / student_logits.size(-1)
    return (
        TF.kl_div(
            input=TF.log_softmax(student_logits / temperature, dim=-1),
            target=TF.log_softmax(teacher_logits / temperature, dim=-1),
            log_target=True,
            reduction="sum",
        )
        * (temperature**2)
        / num_tokens
    )


class KnowledgeDistillation(Algorithm):
    def __init__(self, teacher, temperature, hardness_ce, hardness_kldiv, hardness_squarehead):
        self.teacher = teacher
        self.temperature = temperature
        # loss = hardness_ce x CrossEntropyLoss + hardness_kldiv x KLDivLoss + hardness_squarehead x SquareHeadLoss
        self.hardness_ce = hardness_ce
        self.hardness_kldiv = hardness_kldiv
        self.hardness_squarehead = hardness_squarehead
        self.first_time = True

    def match(self, event, state):
        """
        Event.AFTER_LOSS = augment loss with knowledge distillation
        Event.FIT_START = initialize FSDP for teacher model
        Event.BEFORE_FORWARD = enable hidden states for SquareHead KD
        """
        return event == Event.AFTER_LOSS or event == Event.FIT_START or event == Event.BEFORE_FORWARD

    def apply(self, event, state, logger):
        if event == Event.FIT_START and self.first_time:
            self.first_time = False  # just to be sure FIT_START is reached only once
            if torch.distributed.get_world_size() > 1:
                prepare_fsdp_module(
                    model=self.teacher,
                    optimizers=None,
                    fsdp_config=state.fsdp_config,
                    precision=state.precision,
                    device=get_device(None),
                    auto_microbatching=False,
                )
            else:
                self.teacher = self.teacher.to("cuda")
        elif event == Event.BEFORE_FORWARD:
            state.batch["output_hidden_states"] = True
        elif event == Event.AFTER_LOSS:
            with torch.no_grad():
                teacher_outputs = self.teacher(state.batch)

            loss_gen_tokens = state.batch['labels'] != -100
            student_logits = state.outputs.logits[loss_gen_tokens]
            teacher_logits = teacher_outputs.logits[loss_gen_tokens]

            kl_loss = self.hardness_kldiv * kldiv_loss(student_logits, teacher_logits, self.temperature)
            squarehead_loss = torch.tensor(0.0)

            if self.hardness_squarehead > 0:
                layerwise_losses = []
                for i in range(1, len(state.outputs.hidden_states)):
                    useful_tokens = state.batch['attention_mask'] == 1
                    student_states = state.outputs.hidden_states[i][useful_tokens]
                    teacher_states = teacher_outputs.hidden_states[i][useful_tokens]
                    layerwise_losses.append((student_states - teacher_states).pow(2).mean() / (teacher_states.pow(2).mean() + torch.finfo(torch.bfloat16).eps))

                squarehead_loss = self.hardness_squarehead * sum(layerwise_losses)

            to_log = {
                "losses/ce": self.hardness_ce * state.loss.item(),
                "losses/kldiv": kl_loss.item(),
                "losses/squarehead": squarehead_loss.item(),
            }
            logger.log_metrics(to_log)

            state.loss *= self.hardness_ce
            state.loss += kl_loss + squarehead_loss
####################################################################################################################################################################################