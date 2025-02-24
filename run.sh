#!/bin/bash

export HF_TOKEN=''
export HF_HOME="/mnt/tmp/hf_cache/"
export HF_HUB_ENABLE_HF_TRANSFER=1
export WANDB_API_KEY=''
export WANDB_PROJECT="reasoning"

ACCELERATE_LOG_LEVEL=info TRANSFORMERS_VERBOSITY=info TRAINING_TYPE=SFT \
    accelerate launch --config_file accelerate/ds3.yaml main.py configs/sft/kolimo.yaml

ACCELERATE_LOG_LEVEL=info TRANSFORMERS_VERBOSITY=info TRAINING_TYPE=SFT \
    accelerate launch --config_file accelerate/ds3.yaml main.py configs/sft/limo.yaml