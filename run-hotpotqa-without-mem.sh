#!/bin/bash

# these commands are with example parameters, you might want to change them --train-subbatch-size 8 \

export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_LAUNCH_BLOCKING=1
cmd="python evaluate_model.py single \
        --pretrained-checkpoint ./runs/small_wiki_bs_128/model_53000-Copy1.pth.tar \
        --task-config ./dp_configs/hotpotqa.json \
        --suffix bs_128_wm_hvd_4_gpu_dgx3_genlen_64_seq_len_512_lr_5e-05/run_hotpotqa_without_wm_from_53000 \
        --train-batch-size 32 \
        --lr 5e-05
        --start-from-batch 53000"

echo $cmd
$cmd
