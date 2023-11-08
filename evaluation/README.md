## MT-Bench Evaluation

Install:

```bash
git clone https://github.com/rsnm2/FastChat.git
cd FastChat
pip install torch=2.0.1 pandas
pip install -e ".[model_worker,llm_judge]"
cd fastchat/llm_judge
```

Download pregenerated baseline and view scores for baseline models:
```bash
python3 download_mt_bench_pregenerated.py
python3 show_result.py
```

Results:
```bash
########## Average ##########
                                score
model                                
gpt-4                        8.990625
gpt-3.5-turbo                7.943750
claude-instant-v1            7.905660
claude-v1                    7.900000
vicuna-33b-v1.3              7.121875
wizardlm-30b                 7.009375
Llama-2-70b-chat             6.856250
Llama-2-13b-chat             6.650000
guanaco-33b                  6.528125
tulu-30b                     6.434375
guanaco-65b                  6.409375
oasst-sft-7-llama-30b        6.409375
palm-2-chat-bison-001        6.400000
mpt-30b-chat                 6.393750
vicuna-13b-v1.3              6.387500
wizardlm-13b                 6.353125
Llama-2-7b-chat              6.268750
vicuna-7b-v1.3               5.996875
baize-v2-13b                 5.750000
nous-hermes-13b              5.553459
mpt-7b-chat                  5.459119
gpt4all-13b-snoozy           5.452830
koala-13b                    5.350000
mpt-30b-instruct             5.218750
falcon-40b-instruct          5.168750
h2ogpt-oasst-open-llama-13b  4.625000
alpaca-13b                   4.531250
chatglm-6b                   4.500000
oasst-sft-4-pythia-12b       4.318750
rwkv-4-raven-14b             3.984375
dolly-v2-12b                 3.275000
fastchat-t5-3b               3.040625
stablelm-tuned-alpha-7b      2.753125
llama-13b                    2.606250
```


Generate answers for a new model:
```bash
python gen_model_answer.py --model-path robertgshaw2/mistral-sft-run1-dense --model-id mistral-sft-run1-dense
python gen_model_answer.py --model-path HuggingFaceH4/mistral-7b-sft-beta --model-id mistral-sft-beta-dense
python gen_model_answer.py --model-path HuggingFaceH4/zephyr-7b-beta --model-id zephry-beta-dense
python gen_model_answer.py --model-path ~/zephyr-training/pruning/data/mistral-sft-beta-50sparse-fp16-one-shot-v0  --model-id mistral-sft-beta-50sparse-fp16-one-shot-v0
```