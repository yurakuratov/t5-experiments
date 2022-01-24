#!/bin/bash

# these commands are with example parameters, you might want to change them --train-subbatch-size 8 \

export CUDA_VISIBLE_DEVICES=4,5,6,7
export CUDA_LAUNCH_BLOCKING=1
cmd="python evaluate_model.py single \
        --pretrained-checkpoint ./runs/wm_small_span_corruption_128_1100000_steps_from_660k/model_99000-Copy1.pth.tar \
        --task-config ./dp_configs/squad_wm.json \
        --suffix wm_bs_128_lr_1e-03_1100k \
        --train-batch-size 32 \
        --lr 1e-03 \
        --start-from-batch 99000 \
        --grad-acc-steps-per-batch 1"


echo $cmd
$cmd
