import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from typing import List, Tuple, Dict

def calculate_scale_and_zero_point(tensor: torch.Tensor, qmin=0, qmax=255) -> Tuple[float, int]:
    # Calculate scale for quantization based on tensor's min and max values.
    min_val, max_val = tensor.min().item(), tensor.max().item()
    max_val, min_val = max(max_val, 0.0), min(min_val, 0.0)
    scale = (max_val - min_val) / (qmax - qmin)
    # Clamping the value within the valid range
    zero_point = int(qmin - round(min_val / scale))
    zero_point = max(qmin, min(qmax, zero_point)) 
    return scale, zero_point

def analyze_activation(activation) -> Dict[str, float]:
    # Basic statistics
    range_val = torch.max(activation).item() - torch.min(activation).item()
    mean_val = torch.mean(activation).item()
    std_dev = torch.std(activation).item()

    # Analyzing outliers by determining how many elements fall outside the mean ± 4*std_dev range
    lower_bound = mean_val - 4 * std_dev
    upper_bound = mean_val + 4 * std_dev
    outliers = torch.sum((activation < lower_bound) | (activation > upper_bound)).item()

    # Quantization effects
    scale, zero_point = calculate_scale_and_zero_point(activation)
    quantized = torch.quantize_per_tensor(activation, scale, zero_point, torch.quint8)
    dequantized = quantized.dequantize()
    l1_loss_change = torch.nn.functional.l1_loss(activation, dequantized).item()
    l1_norm_original = torch.norm(activation, p=1).item()
    relative_l1_change = l1_loss_change / l1_norm_original if l1_norm_original else 0

    layer_statistics = {
        "range": range_val,
        "mean": mean_val,
        "std_dev": std_dev,
        "l1_loss_change": l1_loss_change,
        "relative_l1_change": relative_l1_change,
        "num_outliers": outliers,
    }
    return layer_statistics

def get_forward_hook(layer_name):

    # Hook signature
    def forward_hook(module, input, _):
        # Hook to capture input activations of Linear layers.
        if isinstance(module, torch.nn.Linear):
            input_statistics[layer_name] = analyze_activation(input[0].detach())

    return forward_hook


def process_samples(model, tokenizer, samples: List[str]) -> Dict[str, Dict[str, float]]:
    # Process text samples to collect input activations and layer names.
    global input_statistics
    input_statistics = {}

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):  # Focus on Linear layers (where MatMul happens)
            hooks.append(module.register_forward_hook(get_forward_hook(name)))

    with torch.no_grad():
        inputs = tokenizer(samples, return_tensors="pt", padding=True)
        if model.device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        model(**inputs)

    for hook in hooks:
        hook.remove()

    return input_statistics


def print_dict(data: Dict[str, Dict]):
    # Print dictionary in a formatted table, sorted by value.

    # Extract headers from the keys of the inner dictionaries
    # We assume that all inner dictionaries have the same structure
    headers = ["Layer"] + list(next(iter(data.values())).keys())

    # Find the longest layer name for proper column width setting
    longest_key = max(len(key) for key in data)
    longest_sub_key = max(len(key) for key in list(next(iter(data.values())).keys()))
    print(longest_sub_key)
    column_widths = [longest_key] + [longest_sub_key] * (len(headers) - 1)

    # Create the header template and row template based on the calculated column widths
    header_template = " | ".join(f"{{:<{width}}}" for width in column_widths)
    row_template = " | ".join(f"{{:<{width-1}g}} " if i > 0 else f"{{:<{width}}}" for i, width in enumerate(column_widths))

    # Print the headers
    print(header_template.format(*headers))

    # Sort the data by l1_loss_change in descending order
    sorted_items = sorted(data.items(), key=lambda item: item[1]["l1_loss_change"], reverse=True)

    # Print each layer's statistics
    for key, stat in sorted_items:
        row_data = [key] + [stat[header] for header in headers[1:]]  # skip the first header as it's the layer name
        print(row_template.format(*row_data))


def main(model_name: str, dataset_name: str, dataset_config: str, num_samples: int):
    # Main function: setup, process samples, analyze and print results.
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(dataset_name, dataset_config)
    samples = []
    for idx, example in enumerate(dataset['train']):
        if idx >= num_samples:
            break
        samples.append(example['text'])

    input_statistics = process_samples(model, tokenizer, samples)
    print_dict(input_statistics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze transformer model activations.')
    parser.add_argument('--model', default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument('--dataset', default="wikitext")
    parser.add_argument('--dataset_config', default="wikitext-2-raw-v1")
    parser.add_argument('--num_samples', type=int, default=10)
    args = parser.parse_args()

    main(args.model, args.dataset, args.dataset_config, args.num_samples)
