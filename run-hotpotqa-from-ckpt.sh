#!/bin/bash

# these commands are with example parameters, you might want to change them --train-subbatch-size 8 \

export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_LAUNCH_BLOCKING=1
cmd="python evaluate_model.py single \
        --pretrained-checkpoint ./runs/small_wiki_bs_128/model_19000-Copy1.pth.tar \
        --task-config ./dp_configs/hotpotqa_wm.json \
        --suffix bs_128_wm_hvd_4_gpu_dgx3_genlen_64_seq_len_1024/run_hotpotqa_wm_10_mem_09_nucleus/continuing_from_ckpt_47000_test \
        --train-batch-size 4 \
        --train-subbatch-size 8 \
        --lr 5e-05
        --start-from-batch 47000"

echo $cmd
$cmd
