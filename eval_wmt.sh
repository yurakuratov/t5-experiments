#!/bin/bash
export CUDA_VISIBLE_DEVICES=4,5,10,11,12,13,14,15
cmd="python evaluate_model.py single \
        --pretrained-checkpoint ./runs/base_wiki_bs_128_adafactor_mem_20_run_1/model_2000000.pth \
        --task-config ./dp_configs/wmt/ende.json \
        --suffix bs_128/run_1 \
        --train-batch-size 16"
        
echo $cmd
$cmd