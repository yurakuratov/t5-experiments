#!/bin/bash

# these commands are with example parameters, you might want to change them

export CUDA_VISIBLE_DEVICES=4,5,6,7
export CUDA_LAUNCH_BLOCKING=1
cmd="horovodrun --gloo -np 4 python run_t5_pretraining.py \
        --batch_size 32 \
        --gradient_accumulation_steps 1 \
        --save_interval 10000 \
        --log_interval 500 \
        --iters 1100000 \
        --init_checkpoint ./runs/wm_small_span_corruption_128_1100000_steps/model_130000.pth\
        --data_path ./ThePile/Wikipedia/preprocessed_shards \
        --model_path ./runs/wm_small_span_corruption_128_1100000_steps \
        --input_seq_len 128 \
        --target_seq_len 128 \
        --lr 5e-05 \
        --weight_decay 1e-05 \
        --model_cfg ./t5configs/t5-small-wm.json \
        --model_cls modeling_t5:T5WMForConditionalGeneration"

echo $cmd
$cmd
