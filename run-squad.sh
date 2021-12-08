#!/bin/bash

# these commands are with example parameters, you might want to change them --train-subbatch-size 8 \

export CUDA_VISIBLE_DEVICES=4,5,6,7
export CUDA_LAUNCH_BLOCKING=1
cmd="python evaluate_model.py single \
        --pretrained-checkpoint ./runs/wm_small_span_corruption_128_1100000_steps/model_350000.pth \
        --task-config ./dp_configs/squad_wm.json \
        --suffix wm_bs_128_lr_5e-05 \
        --train-batch-size 32 \
        --lr 5e-05
        --start-from-batch -1"

echo $cmd
$cmd
