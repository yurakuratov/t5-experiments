#!/bin/bash
export CUDA_VISIBLE_DEVICES=10,11
cmd="python evaluate_model.py single \
        --pretrained-checkpoint ./runs/base_wiki_bs_128_adafactor_run_1/model_2000000.pth \
        --task-config ./dp_configs/glue/ \
        --suffix bs_32_lr_5e-04/run_1 \
        --train-batch-size 16 \
        --lr 5e-04"
        
echo $cmd
$cmd