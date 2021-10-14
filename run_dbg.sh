#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1
cmd="horovodrun --gloo -np 2 python run_t5_pretraining.py \
        --batch_size 4 \
        --gradient_accumulation_steps 1 \
        --save_interval 250000 \
        --log_interval 5000 \
        --iters 100000000 \
        --task span_corruption \
        --data_path /home/jovyan/data/Wikipedia/preprocessed_shards_train \
        --valid_data_path /home/jovyan/data/Wikipedia/preprocessed_shards_valid \
        --model_path ./runs/large_dbg \
        --vocab ./vocabs/sentencepiece.model \
        --input_seq_len 512 \
        --target_seq_len 192 \
        --optimizer Adafactor \
        --fp16 --apex_opt_lvl O2 \
        --scale_parameter --relative_step --warmup_init \
        --model_cfg ./t5configs/t5-large.json \
        --model_cls modeling_t5:T5ForConditionalGeneration \
        --save_best"
        
echo $cmd
$cmd
