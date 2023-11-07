import datasets as hf_datasets
from composer.core.data_spec import DataSpec
from composer.utils import dist
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase
from llmfoundry.data.finetuning.tasks import dataset_constructor
from llmfoundry.data.text_data import get_tokens_per_batch_func
from trl import DataCollatorForCompletionOnlyLM
from functools import partial
import os

system_prompt = {
    "content": "You are a friendly chatbot",
    "role": "system"
}

def add_system_prompt(batch):
    updated_messages = []
    for element in batch["messages"]:
        updated_messages.append([system_prompt] + element)

    return {"messages_with_sys_prompt": updated_messages}

def apply_chat_template(tokenizer, messages_col, element):
    return tokenizer.apply_chat_template(element[messages_col], tokenize=False)
    
    # strs = []
    # for example in batch[messages_col]:
    #     strs.append(tokenizer.apply_chat_template(example, tokenize=False))

    # return strs

def build_from_hf(
    cfg: DictConfig, 
    max_seq_len: int, 
    tokenizer: PreTrainedTokenizerBase,
):
    dataset_name = cfg.hf_name
    split = cfg.split.replace('-', '_')
    dataset = hf_datasets.load_dataset(dataset_name, split=split)

    dataset = dataset.map(
        add_system_prompt,
        batched=True,
        num_proc=1,
        batch_size=32,
    )

    chat_formatting_func = partial(apply_chat_template, tokenizer, "messages_with_sys_prompt")

    def tokenize(element):
        outputs = tokenizer(
            chat_formatting_func(element),
            truncation=True,
            padding=False,
            max_length=max_seq_len,
            return_overflowing_tokens=False,
            return_length=False,
        )

        return {"input_ids": outputs["input_ids"], "attention_mask": outputs["attention_mask"]}

    tokenized_dataset = dataset.map(
        tokenize,
        batched=False,
        remove_columns=dataset.column_names,
        num_proc=1,
    )

    return tokenized_dataset

def build_chat_dataloader(
    cfg: DictConfig,
    tokenizer: PreTrainedTokenizerBase,
    device_batch_size: int) -> DataSpec:

    # Use EOS as the pad token if none exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = build_from_hf(
        cfg.dataset,
        max_seq_len=cfg.dataset.max_seq_len,
        tokenizer=tokenizer,
    )
    
    # masks out the user inputs in the labels (sets to -100)
    response_template_ids = tokenizer.encode(cfg.dataset.response_template, add_special_tokens=False)[2:]
    instruction_template_ids = tokenizer.encode(cfg.dataset.instruction_template, add_special_tokens=False)[2:]
    collate_fn = DataCollatorForCompletionOnlyLM(
        response_template=response_template_ids,
        instruction_template=instruction_template_ids,
        tokenizer=tokenizer,
    )
    dataloader_batch_size = device_batch_size

    dl = DataLoader(
        dataset,
        collate_fn=collate_fn,
        batch_size=dataloader_batch_size,
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
