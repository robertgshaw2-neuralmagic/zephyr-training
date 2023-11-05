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
    -v $PWD/data:/data -v $PWD/scripts:/scripts -v $PWD/llm-foundry:/llm-foundry -e HF_HOME="/data" -e WANDB_API_KEY=$wb_key\
    --network host -it --rm llmf
```

Install LLM-foundry in the container:
```bash
pip install -e /llm-foundry
```

Launch Script:
```bash
composer /scripts/train.py /data/yamls/run0.yaml
```
