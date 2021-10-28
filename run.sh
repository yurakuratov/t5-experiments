#!/bin/bash

# these commands are with example parameters, you might want to change them

export CUDA_VISIBLE_DEVICES=4
export CUDA_LAUNCH_BLOCKING=1
cmd="python run_t5_pretraining.py \
        --batch_size 32 \
        --gradient_accumulation_steps 2 \
        --save_interval 100000 \
        --log_interval 500 \
        --iters 1100000 \
        --data_path ~/asc-trr/ThePile/Wikipedia/preprocessed_shards \
        --model_path ./runs/test_wm_128 \
        --input_seq_len 128 \
        --target_seq_len 128 \
        --lr 5e-05 \
        --weight_decay 1e-05 \
        --model_cfg ./t5configs/t5-small.json \
        --model_cls modeling_t5:T5WMForConditionalGeneration"

echo $cmd
$cmd
