#!/bin/bash
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
cmd="horovodrun --gloo -np 4 python run_t5_pretraining.py \
        --batch_size 32 \
        --gradient_accumulation_steps 1 \
        --save_interval 250000 \
        --log_interval 5000 \
        --iters 100000000 \
        --task prefix_lm \
        --data_path /home/jovyan/data/ods_shards/preprocessed_shards_train \
        --valid_data_path /home/jovyan/data/ods_shards/preprocessed_shards_valid \
        --model_path ./runs/small_ods_plm_bs_128_adafactor \
        --init_checkpoint ./runs/small_ods_plm_bs_128_adafactor/model_3500000.pth \
        --vocab ./vocabs/ods_data_50259_sp_full_spkr_tokens.model \
        --input_seq_len 128 \
        --target_seq_len 128 \
        --optimizer Adafactor \
        --scale_parameter --relative_step --warmup_init \
        --model_cfg ./t5configs/t5-small_voc_50359.json \
        --model_cls modeling_t5:T5ForConditionalGeneration \
        --save_best"
        
echo $cmd
$cmd