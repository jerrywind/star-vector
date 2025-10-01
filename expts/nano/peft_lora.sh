#!/bin/bash

#export HF_HOME=<path to the folder where you want to store the models>
#export HF_TOKEN=<your huggingface token>
#export WANDB_API_KEY=<your wandb token>
root_dir= $(dirname $(realpath $0))
export OUTPUT_DIR="${root_dir}/outputs"
train_script=${root_dir}/train_lora.py

torchrun \
  --standalone \
  --nproc-per-node=1 \
  --nnodes=1 \
  ${train_script} \
  config=configs/models/starvector-8b/im2svg-stack.yaml
