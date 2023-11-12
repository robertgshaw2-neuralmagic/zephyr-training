import os, random

from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase, default_data_collator
import datasets as hf_datasets
from trl.trainer.utils import ConstantLengthDataset

from composer.core.data_spec import DataSpec
from composer.utils import dist
from llmfoundry.data.finetuning.tasks import dataset_constructor
from llmfoundry.data.text_data import get_tokens_per_batch_func
from omegaconf import DictConfig

def apply_chat_template(example, tokenizer):
    messages = example["messages"]
    # We add an empty system message if there is none
    if messages[0]["role"] != "system":
        messages.insert(0, {"role": "system", "content": ""})

    example["text"] = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
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
    
    dataset = dataset.map(apply_chat_template, fn_kwargs={"tokenizer": tokenizer})
    for index in random.sample(range(len(dataset)), 3):
        print(f"Sample {index} of the processed training set:\n\n{dataset[index]['text']}")

    # build packed dataset (note: tokenized here)
    return ConstantLengthDataset(
        tokenizer,
        dataset,
        dataset_text_field="text",
        formatting_func=None,
        seq_length=max_seq_len,
        infinite=False,
        num_of_sequences=1024,
        chars_per_token=3.6,
        eos_token_id=tokenizer.eos_token_id,
    )

def build_packed_chat_dataloader(
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

    # put into a dataloader
    dl = DataLoader(
        dataset,
        collate_fn=default_data_collator,
        batch_size=device_batch_size,
        # drop_last=cfg.drop_last,
        # sampler=dist.get_sampler(dataset, drop_last=cfg.drop_last, shuffle=cfg.dataset.shuffle),
        # num_workers=cfg.num_workers,
        # pin_memory=cfg.get('pin_memory', True),
        # prefetch_factor=cfg.get('prefetch_factor', 2),
        # persistent_workers=cfg.get('persistent_workers', True),
        timeout=cfg.get('timeout', 0),
    )

    token_counting_func = get_tokens_per_batch_func(
        pad_token_id=tokenizer.pad_token_id)

    return DataSpec(dataloader=dl, get_num_tokens_in_batch=token_counting_func)
