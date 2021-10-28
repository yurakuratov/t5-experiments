#!/bin/bash

# these commands are with example parameters, you might want to change them

export CUDA_VISIBLE_DEVICES=0,3,4,5
export CUDA_LAUNCH_BLOCKING=1
cmd="horovodrun --gloo -np 4 python run_t5_pretraining.py \
        --batch_size 32 \
        --gradient_accumulation_steps 1 \
        --save_interval 10 \
        --log_interval 500 \
        --iters 1100000 \
        --data_path ./ThePile/Wikipedia/preprocessed_shards \
        --model_path ./runs/wm_128_test \
        --input_seq_len 128 \
        --target_seq_len 128 \
        --lr 5e-05 \
        --weight_decay 1e-05 \
        --model_cfg ./t5configs/t5-small-wm.json \
        --model_cls modeling_t5:T5WMForConditionalGeneration"

echo $cmd
$cmd
