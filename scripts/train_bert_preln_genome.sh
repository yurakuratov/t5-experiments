#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=1,2 NP=2 ./test_bert_sparse_pretrain_train_valid.sh
set -e
cd ..

TBS=256
BS=32

horovodrun --gloo -np $NP python run_bert_pretraining.py \
--data_path /home/jovyan/data/human_genome/human_train_text_sentence \
--valid_data_path /home/jovyan/data/human_genome/human_valid_text_sentence \
--tokenizer ./vocabs/genom_32k/ --model_cfg ./bert_configs/L12-H768-A12-V32k-preln.json \
--model_cls modeling_bert:BertForPreTraining \
--model_path ./runs/bert_base_512_bs256_lr_1e-04_fp16 \
--input_seq_len 512 \
--batch_size $BS --gradient_accumulation_steps $(($TBS/($BS*$NP))) \
--data_n_epochs 25 --save_best --iters 4000000 \
--lr 1e-04 --lr_scheduler constant_with_warmup --num_warmup_steps 10000 \
--data_n_workers 8 \
--log_interval 500 --save_interval 100000 --valid_interval 25000 \
--fp16 --apex_opt_lvl O2
echo "run_bert_pretraining.py done"
echo "done"
