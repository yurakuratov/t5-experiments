#!/bin/bash

# these commands are with example parameters, you might want to change them --train-subbatch-size 8 \

export CUDA_VISIBLE_DEVICES=0,1,2,4
export CUDA_LAUNCH_BLOCKING=1
cmd="python evaluate_model.py single \
        --pretrained-checkpoint ./runs/small_wiki_bs_128/model_1100000.pth \
        --task-config ./dp_configs/wmt/ende_wm.json \
        --suffix bs_64_wm_hvd/run_0 \
        --train-batch-size 16 \
        --lr 5e-05"

echo $cmd
$cmd
