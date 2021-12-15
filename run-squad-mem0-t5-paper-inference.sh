#!/bin/bash

# these commands are with example parameters, you might want to change them --train-subbatch-size 8 \

export CUDA_VISIBLE_DEVICES=2,5
export CUDA_LAUNCH_BLOCKING=1
cmd="python evaluate_model.py single \
        --pretrained-checkpoint ./runs/small_wiki_bs_128_adafactor/model_31000-Copy1.pth.tar \
        --task-config ./dp_configs/squad_from_no_mem_pretraining-t5-paper-show-examples.json \
        --suffix wm0_t5-paper-inference \
        --train-batch-size 8 \
        --start-from-batch -1
        --grad-acc-steps-per-batch 1"

echo $cmd
$cmd
