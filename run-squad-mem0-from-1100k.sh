#!/bin/bash

# these commands are with example parameters, you might want to change them --train-subbatch-size 8 \

export CUDA_VISIBLE_DEVICES=1,2,3,4,5,7,8,9
export CUDA_LAUNCH_BLOCKING=1
cmd="python evaluate_model.py single \
        --pretrained-checkpoint ./runs/small_wiki_bs_128/model_1100000.pth \
        --task-config ./dp_configs/squad_from_no_mem_pretraining.json \
        --suffix wm0_bs_128_lr_1e-03 \
        --train-batch-size 16 \
        --lr 1e-03
        --start-from-batch -1
        --grad-acc-steps-per-batch 1"

echo $cmd
$cmd
