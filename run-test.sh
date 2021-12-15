#!/bin/bash

# these commands are with example parameters, you might want to change them --train-subbatch-size 8 \

export CUDA_VISIBLE_DEVICES=8,9
export CUDA_LAUNCH_BLOCKING=1
cmd="python evaluate_model.py single \
        --pretrained-checkpoint ./runs/small_wiki_bs_128/model_90000-Copy1.pth.tar \
        --task-config ./dp_configs/hotpotqa_wm_test.json \
        --suffix bs_128_wm_hvd_4_gpu_dgx3_genlen_64_seq_len_512_lr_5e-05/run_hotpotqa_wm_10_mem_09_nucleus_run_1_from_90000___testtttt_with_wm \
        --train-batch-size 2 \
        --lr 5e-05
        --start-from-batch 90000"

echo $cmd
$cmd
