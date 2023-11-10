Install SparseML:
```bash
git clone s
pip install -e ./sparseml[transformers]
```

Run Pruning:
```bash
python3 prune.py \
    --model-path HuggingFaceH4/zephyr-7b-beta \
    --recipe-path ./recipe-50sparse.yaml \
    --save-path data/zephyr-beta-50sparse-fp16-v0
```

```bash
python3 prune.py \
    --model-path HuggingFaceH4/mistral-7b-sft-beta \
    --recipe-path ./recipe-50sparse.yaml \
    --save-path data/mistral-sft-50sparse-fp16-v0
```
