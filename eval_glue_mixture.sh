#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
cmd="python evaluate_model.py train-mixture \
        --pretrained-checkpoint ./runs/base_wiki_bs_128_adafactor_run_1/model_2000000.pth \
        --task-config ./dp_configs/glue/glue_mixture.json \
        --suffix bs_128/run_0 \
        --train-batch-size 32"
        
echo $cmd
$cmd

cmd="python evaluate_model.py mixture \
        --pretrained-checkpoint ./runs/base_wiki_bs_128_adafactor_run_1/model_2000000.pth \
        --checkpoint ./runs/base_wiki_bs_128_adafactor_run_1/glue/glue_mixture/bs_128/run_0 \
        --task-config ./dp_configs/glue/ \
        --eval-batch-size 64"
        
echo $cmd
$cmd