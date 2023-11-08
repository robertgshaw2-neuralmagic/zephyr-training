import torch, argparse, time
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from datasets import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--num_prompts", type=int, required=False, default=3)

def generate_prompts(tokenizer, dataset, messages_column_name="messages", num_prompts=3):
    messages = dataset.shuffle().select(range(num_prompts))[messages_column_name]

    system_prompt = {
        "content": "You are a helpful chatbot",
        "role": "system"
    }
    prompts = []
    for message in messages:
        convo = [system_prompt] + message[:-1]
        prompt = tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)
    return prompts

@torch.inference_mode()
def run_generation(
    model,
    tokenizer,
    dataset,
    messages_column_name="messages",
    num_prompts=3,
):
    prompts = generate_prompts(
        tokenizer,
        dataset,
        messages_column_name=messages_column_name,
        num_prompts=num_prompts,
    )

    for prompt in prompts:
        inps = tokenizer(prompt, return_tensors="pt")
        for key in inps:
            inps[key] = inps[key].to("cuda")
        
        generation_config = GenerationConfig(
            use_cache=True,
            do_sample=False, 
            max_new_tokens=1024, 
            pad_token_id=tokenizer.eos_token_id
        )
        
        start = time.perf_counter()
        output_tokens = model.generate(**inps, generation_config=generation_config)
        torch.cuda.synchronize()
        end = time.perf_counter()

        print(tokenizer.batch_decode(output_tokens)[0])
        print("--------------------------------------------------------------------------------------------------------------")
        start_tokens = inps["input_ids"].shape[1]
        total_tokens = output_tokens.shape[1]
        print(f"tok/sec: {(total_tokens - start_tokens) / (end - start)}")
        print("\n\n")

def main(model_path, dataset_id, num_prompts=3):
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        device_map={'':0}, 
        torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    dataset = load_dataset(dataset_id, split="train_sft")
    run_generation(
        model,
        tokenizer,
        dataset,
        messages_column_name="messages",
        num_prompts=num_prompts,
    )

if __name__ == "__main__":
    args = parser.parse_args()
    
    main(
        model_path=args.model_path, 
        dataset_id="HuggingFaceH4/ultrachat_200k", 
        num_prompts=args.num_prompts
    )