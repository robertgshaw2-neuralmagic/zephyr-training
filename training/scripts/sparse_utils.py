import torch
from composer.core import Event, Algorithm
from composer import Callback, State, Logger

def print_layer_sparsity(model: torch.nn.Module) -> None:
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear):
            mask = torch.where(module.weight == 0, torch.tensor(0, dtype=torch.uint8), torch.tensor(1, dtype=torch.uint8))
            print(f"[Layer {name} sparsity = {torch.sum(mask == 0)/mask.numel()}]")

class MeasureSparsityCallback(Callback):
    def batch_start(self, state: State, logger: Logger):
        with torch.no_grad():
            print_layer_sparsity(state.model)

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