#!/bin/bash

cd training

docker run \
    --shm-size 2gb --gpus all \
    -v $PWD/data:/data -v $PWD/scripts:/scripts \
    -e HUGGING_FACE_HUB_TOKEN=$HF_TOKEN_READ -e HF_TOKEN_WRITE=$HF_TOKEN_WRITE -e HF_HOME="/data" -e WANDB_API_KEY=$WANDB_API_KEY \
    --network host -it --rm llmf