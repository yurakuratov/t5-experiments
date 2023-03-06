#!/usr/bin/env bash
# e.g.:
# CUDA_VISIBLE_DEVICES=1,2 NP=2 ./test_bert_sparse_pretrain_train_valid.sh
CUDA_LAUNCH_BLOCKING=1
set -e
cd ..

TBS=256
BS=8

WD=1e-04
LR=2e-05
FP16=O2

INPUT_SEQ_LEN=3992
# INPUT_SEQ_LEN=512
INPUT_SIZE=512  # segment length
# INPUT_SIZE=256
MAX_N_SEGMENTS=8
MEMORY_SIZE=10
BPTT=-1

rmt_params=seglen_${INPUT_SIZE}_len_${INPUT_SEQ_LEN}_maxnsegm_${MAX_N_SEGMENTS}_msz_${MEMORY_SIZE}_bptt${BPTT}
horovodrun --gloo -np $NP python run_rmt_pretraining.py \
        --data_path /home/jovyan/data/train_data/t2t_1000Gselect_train_text_sentence \
        --valid_data_path /home/jovyan/data/train_data/t2t_valid_text_sentence \
        --tokenizer /home/jovyan/dnalm/data/tokenizers/t2t_1000h_multi_32k \
        --model_cfg ./bert_configs/L12-H768-A12-V32k-preln-lastln.json \
        --backbone_cls modeling_bert:BertForMaskedLM \
        --backbone_checkpoint /home/jovyan/t5-experiments/runs/bert_base_512_lastln_t2t_1000G_bs256_lr_1e-04_linear_fp16/model_2000000.pth \
        --model_cls modeling_rmt:RMTEncoderForMaskedLM \
        --use_nsp 0 \
        --model_path ./runs/rmt_bert_base_lastln_t2t_1000G_${rmt_params}_bs${TBS}_lr_${LR}_wd_${WD}_fp16_${FP16} \
        --input_seq_len $INPUT_SEQ_LEN \
        --input_size $INPUT_SIZE \
        --num_mem_tokens $MEMORY_SIZE \
        --max_n_segments $MAX_N_SEGMENTS \
        --segment_alignment center \
        --bptt_depth -1 \
        --batch_size $BS --gradient_accumulation_steps $(($TBS/($BS*$NP))) \
        --data_n_epochs 1 --data_skip_warmup --iters 4000000 \
        --optimizer FusedAdam --weight_decay $WD \
	--lr $LR --lr_scheduler constant_with_warmup --num_warmup_steps 50000 \
        --data_n_workers 8 --short_seq_prob 0.1 \
        --log_interval 500 --save_interval 10000 --valid_interval 10000 \
        --save_best \
        --fp16 --apex_opt_lvl $FP16 --clip_grad_norm 1.0
echo "run_bert_pretraining.py done"
echo "done"
