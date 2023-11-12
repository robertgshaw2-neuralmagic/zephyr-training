import os, random

from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase, default_data_collator, DataCollatorWithPadding
import datasets as hf_datasets

from composer.core.data_spec import DataSpec
from composer.utils import dist
from llmfoundry.data.finetuning.tasks import dataset_constructor
from llmfoundry.data.text_data import get_tokens_per_batch_func
from omegaconf import DictConfig

class DataCollatorForCausalLM(DataCollatorWithPadding):
    def __call__(self, features):
        batch = super().__call__(features)
        
        # create labels
        batch["labels"] = batch["input_ids"].clone()
        
        # mask out any padding from the loss
        batch["labels"][batch["attention_mask"] == 0] = -100

        return batch

def apply_chat_template(example, tokenizer, max_seq_len):
    messages = example["messages"]
    # We add an empty system message if there is none
    if messages[0]["role"] != "system":
        messages.insert(0, {"role": "system", "content": ""})

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )

    outputs = tokenizer(
        text,
        truncation=True,
        padding=False,
        max_length=max_seq_len,
        return_overflowing_tokens=False,
        return_length=False,
    )

    return {
        "input_ids": outputs["input_ids"],
        "attention_mask": outputs["attention_mask"],
    }

def build_from_hf(
    cfg: DictConfig,
    tokenizer: PreTrainedTokenizerBase,
):
    dataset_name = cfg.hf_name
    split = cfg.split.replace('-', '_')
    dataset = hf_datasets.load_dataset(dataset_name, split=split)

    if cfg.num_samples != "all":
        dataset = dataset.shuffle().select(range(cfg.num_samples))

    column_names = dataset.column_names

    if tokenizer.padding_side != "left":
        print("------------------------------------------------")
        print("setting padding side and truncation side to left")
        tokenizer.padding_side = "left"

    if tokenizer.truncation_side != "left":
        print("------------------------------------------------")
        print("setting truncation side to left")
        tokenizer.truncation_side = "left"

    dataset = dataset.map(
        apply_chat_template, 
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_seq_len": cfg.max_seq_len,
        })
    
    return dataset.remove_columns(column_names)

def build_chat_text_dataloader(
    cfg: DictConfig,
    tokenizer: PreTrainedTokenizerBase,
    device_batch_size: int) -> DataSpec:

    # Use EOS as the pad token if none exists << we 
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # build packed dataset
    dataset = build_from_hf(
        cfg.dataset,
        tokenizer=tokenizer,
    )

    # put into a dataloader
    dl = DataLoader(
        dataset,
        collate_fn=DataCollatorForCausalLM(tokenizer),
        batch_size=device_batch_size,
        drop_last=cfg.drop_last,
        sampler=dist.get_sampler(dataset, drop_last=cfg.drop_last, shuffle=cfg.dataset.shuffle),
        num_workers=cfg.num_workers,
        pin_memory=cfg.get('pin_memory', True),
        prefetch_factor=cfg.get('prefetch_factor', 2),
        persistent_workers=cfg.get('persistent_workers', True),
        timeout=cfg.get('timeout', 0),
    )

    token_counting_func = get_tokens_per_batch_func(
        pad_token_id=tokenizer.pad_token_id)

    return DataSpec(dataloader=dl, get_num_tokens_in_batch=token_counting_func)
