import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from functools import partial

def add_system_prompt(batch):
    system_prompt = {
        "content": "You are a friendly chatbot",
        "role": "system"
    }

    updated_messages = []
    for element in batch["messages"]:
        updated_messages.append([system_prompt] + element)

    return {"messages_with_sys_prompt": updated_messages}

def apply_chat_template(tokenizer, messages_col, batch):
    strs = []
    for example in batch[messages_col]:
        strs.append(tokenizer.apply_chat_template(example, tokenize=False))

    return strs

def train(
    model_id = "mistralai/Mistral-7B-v0.1",
    dataset_id = "HuggingFaceH4/ultrachat_200k",
    output_dir = "./sft-results",
):
    torch_dtype = torch.bfloat16

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch_dtype, 
        bnb_4bit_use_double_quant=True, 
        bnb_4bit_quant_type="nf4"
    )

    device_map = {"": Accelerator().local_process_index}

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map=device_map,
        torch_dtype=torch_dtype,
    )

    LORA_R = 8
    LORA_ALPHA = 16

    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    dataset = load_dataset(dataset_id)
    train_dataset = dataset["train_sft"]
    eval_dataset = dataset["test_sft"]

    train_dataset = train_dataset.map(
        add_system_prompt,
        batched=True,
        num_proc=32,
        batch_size=1000,
    )

    eval_dataset = eval_dataset.map(
        add_system_prompt,
        batched=True,
        num_proc=32,
        batch_size=1000,
    )

    RESPONSE_TEMPLATE = "/n<|assistant|>"
    INSTRUCTION_TEMPLATE = "/n<|user|>"

    chat_tokenizer_id = "HuggingFaceH4/mistral-7b-sft-beta"
    tokenizer = AutoTokenizer.from_pretrained(chat_tokenizer_id)
    response_template_ids = tokenizer.encode(RESPONSE_TEMPLATE, add_special_tokens=False)[2:]
    instruction_template_ids = tokenizer.encode(INSTRUCTION_TEMPLATE, add_special_tokens=False)[2:]

    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template_ids,
        instruction_template=instruction_template_ids,
        tokenizer=tokenizer,
    )

    BATCH_SIZE = 1
    GRADIENT_ACCUMULATION_STEPS = 1
    LEARNING_RATE = 2e-05
    LOGGING_STEPS = 1000
    EVAL_STEPS = 5000
    EPOCHS = 2

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=EPOCHS,
        report_to="tensorboard",
        gradient_checkpointing=False,
        do_train=True,
        do_eval=True,
        evaluation_strategy="steps",
        eval_steps=EVAL_STEPS,
        logging_steps=LOGGING_STEPS,
    )

    NUM_SAMPLES_TRAIN = 20000
    NUM_SAMPLES_EVAL = 1000
    MAX_SEQ_LEN = 512

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset.select(range(NUM_SAMPLES_TRAIN)),
        eval_dataset=eval_dataset.select(range(NUM_SAMPLES_EVAL)),
        data_collator=collator,
        formatting_func=partial(apply_chat_template, tokenizer, "messages_with_sys_prompt"),
        max_seq_length=MAX_SEQ_LEN,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(script_args.output_dir)

if __name__ == "__main__":
    train()