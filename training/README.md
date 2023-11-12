Clone LLM Foundry:
```bash
git clone https://github.com/mosaicml/llm-foundry.git
```

Build Image:
```bash
docker build -t llmf .
```

Launch Container:
```bash
docker run \
    --shm-size 2gb --gpus all \
    -v $PWD/data:/data -v $PWD/scripts:/scripts \
    -e HF_TOKEN=$HF_TOKEN_READ -e HF_TOKEN_WRITE=$HF_TOKEN_WRITE -e HF_HOME="/data" -e WANDB_API_KEY=$WANDB_API_KEY \
    --network host -it --rm llmf
```

Launch Script:
```bash
composer /scripts/train.py /data/yamls/run2.yaml
```

Save to HF:
```bash
HUGGING_FACE_HUB_TOKEN=$HF_TOKEN_WRITE python /llm-foundry/scripts/inference/convert_composer_to_hf.py \
  --composer_path /data/runs/mistral-sft-50sparse-fp16-v0-run1/checkpoints/latest-rank0.pt \
  --hf_output_path mistral-sft-50sparse-run1 \
  --output_precision bf16 \
  --hf_repo_for_upload robertgshaw2/mistral-sft-50sparse-run1
```
