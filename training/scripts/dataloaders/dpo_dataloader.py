import re, random

from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase
from trl.trainer.utils import DPODataCollatorWithPadding
import datasets as hf_datasets

from composer.core.data_spec import DataSpec
from composer.utils import dist
from llmfoundry.data.finetuning.tasks import dataset_constructor
from llmfoundry.data.text_data import get_tokens_per_batch_func
from omegaconf import DictConfig

def _strip_prefix(s, pattern):
    # Use re.escape to escape any special characters in the pattern
    return re.sub(f"^{re.escape(pattern)}", "", s)

def apply_chat_template(example, tokenizer, assistant_prefix="<|assistant|>\n"):
    if all(k in example.keys() for k in ("chosen", "rejected")):
        
        # TODO: handle case where the dataset has system messages
        assert example["chosen"][0]["role"] == "user"
        assert example["chosen"][0]["role"] == "user"
        
        # Filter out the prompt
        prompt_messages = [[msg for msg in example["chosen"] if msg["role"] == "user"][0]]
        prompt_messages.insert(0, {"role": "system", "content": ""})
        
        chosen_messages = example["chosen"][1:]
        rejected_messages = example["rejected"][1:]
        example["text_chosen"] = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
        example["text_rejected"] = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
        example["text_prompt"] = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )

    example["text_chosen"] = _strip_prefix(example["text_chosen"], assistant_prefix)
    example["text_rejected"] = _strip_prefix(example["text_rejected"], assistant_prefix)

    return example

def build_from_hf(
    cfg: DictConfig,
    max_seq_len: int, 
    tokenizer: PreTrainedTokenizerBase,
):
    dataset_name = cfg.hf_name
    split = cfg.split.replace('-', '_')
    dataset = hf_datasets.load_dataset(dataset_name, split=split)

    if cfg.num_samples != "all":
        dataset = dataset.shuffle().select(range(cfg.num_samples))
    
    tokenizer.truncation_side = "left"

    dataset = dataset.map(
        apply_chat_template, 
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=list(dataset.features),
        desc="Formatting comparisons with prompt template",
    )
    
    for index in random.sample(range(len(dataset)), 1):
        print(f"\n\n--------------------------------------------------------------------------------------------------------")
        print(f"PROMPT:")
        print(f"{dataset[index]['text_prompt']}")
        print(f"--------------------------------------------------------------------------------------------------------")
        print(f"CHOSEN:")
        print(f"{dataset[index]['text_chosen']}\n")
        print(f"--------------------------------------------------------------------------------------------------------")
        print(f"REJECTED:")
        print(f"{dataset[index]['text_rejected']}")
        print(f"--------------------------------------------------------------------------------------------------------\n\n")

    # build packed dataset (note: tokenized here)
    return dataset.rename_columns({
        "text_prompt": "prompt", 
        "text_chosen": "chosen", 
        "text_rejected": "rejected"
    })

def build_dpo_dataloader(
    cfg: DictConfig,
    tokenizer: PreTrainedTokenizerBase,
    device_batch_size: int) -> DataSpec:

    # Use EOS as the pad token if none exists << we 
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # build packed dataset
    dataset = build_from_hf(
        cfg.dataset,
        max_seq_len=cfg.dataset.max_seq_len,
        tokenizer=tokenizer,
    )

    data_collator = DPODataCollatorWithPadding(
        tokenizer,
        max_length=cfg.dataset.max_seq_len,
        max_prompt_length=cfg.dataset.max_prompt_length,
        label_pad_token_id=-100,
        padding_value=0,
        truncation_mode="keep_end",
    )

    # put into a dataloader
    dl = DataLoader(
        dataset,
        collate_fn=data_collator,
        batch_size=device_batch_size,
        drop_last=cfg.drop_last,
        sampler=dist.get_sampler(dataset, drop_last=cfg.drop_last, shuffle=cfg.dataset.shuffle),
        num_workers=cfg.num_workers,
        pin_memory=cfg.get('pin_memory', True),
        prefetch_factor=cfg.get('prefetch_factor', 2),
        persistent_workers=cfg.get('persistent_workers', True),
        timeout=cfg.get('timeout', 0),
    )

    # token_counting_func = get_tokens_per_batch_func(
    #     pad_token_id=tokenizer.pad_token_id)
    # 
    # return DataSpec(dataloader=dl, get_num_tokens_in_batch=token_counting_func)
    return DataSpec(dataloader=dl)
