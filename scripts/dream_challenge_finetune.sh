#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=1,2 NP=2 ./test_bert_sparse_pretrain_train_valid.sh
set -e
cd ..

CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1

LR=1e-04
SCHEDULER=constant_with_warmup
ITERS=100000
BASE_CKPT=model_1000000

for LR in 3e-04 1e-04 5e-05 1e-05
do
for SCHEDULER in constant_with_warmup linear
do
for N in 1 2 3 4 5
do
horovodrun --gloo -np $NP python run_bert_dream_challenge.py \
        --data_path /home/kuratov/data/genomes/downstream_tasks/DREAM_challenge/split_$N/train \
        --valid_data_path /home/kuratov/data/genomes/downstream_tasks/DREAM_challenge/split_$N/valid \
        --test_data_path /home/kuratov/data/genomes/downstream_tasks/DREAM_challenge/split_$N/test \
        --tokenizer ./vocabs/genom_32k/ --model_cfg ./bert_configs/L12-H768-A12-V32k-preln.json \
        --model_cls modeling_bert:BertForSequenceClassification \
        --model_path ./runs/bert_base_512_bs256_lr_1e-04_fp16/dream_challenge/${BASE_CKPT}/lr${LR}_${SCHEDULER}_it${ITERS}_adamw_fp32/run_$N \
        --batch_size 128 --gradient_accumulation_steps 1 \
        --save_best --iters $ITERS \
        --optimizer AdamW \
        --lr $LR --lr_scheduler $SCHEDULER --num_warmup_steps 5000 \
        --reset_lr --reset_optimizer --reset_iteration \
        --data_n_workers 4 \
        --log_interval 250 --valid_interval 5000 \
        --optimize_metric pearsonr2 --optimize_mode max \
        --clip_grad_value 5.0 \
        --init_checkpoint ./runs/bert_base_512_bs256_lr_1e-04_fp16/${BASE_CKPT}.pth
done
done
done
echo "run_bert_pretraining.py done"
echo "done"