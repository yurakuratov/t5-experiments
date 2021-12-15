#!/bin/bash

# these commands are with example parameters, you might want to change them --train-subbatch-size 8 \

export CUDA_VISIBLE_DEVICES=2,3,4,5
export CUDA_LAUNCH_BLOCKING=1
cmd="python evaluate_model.py single \
        --pretrained-checkpoint ./runs/small_wiki_bs_128_adafactor/model_400000.pth \
        --task-config ./dp_configs/squad_from_no_mem_pretraining-t5-paper.json \
        --suffix wm0_t5-paper \
        --train-batch-size 32 \
        --start-from-batch -1
        --grad-acc-steps-per-batch 2"

echo $cmd
$cmd
