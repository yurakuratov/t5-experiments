#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=1,2 NP=2 ./test_bert_sparse_pretrain_train_valid.sh
set -e
cd ..

CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1

for N in {1..5}
do
        horovodrun --gloo -np $NP python run_bert_epdnewpromoter.py \
                --data_path /home/kuratov/data/genomes/downstream_tasks/epdnew_promoter/len_2000/split_$N/train \
                --valid_data_path /home/kuratov/data/genomes/downstream_tasks/epdnew_promoter/len_2000/split_$N/valid \
                --test_data_path /home/kuratov/data/genomes/downstream_tasks/epdnew_promoter/len_2000/split_$N/test \
                --tokenizer ./vocabs/genom_32k/ --model_cfg ./bert_configs/L12-H768-A12-V32k-preln.json \
                --model_cls modeling_bert:BertForSequenceClassification \
                --model_path ./runs/bert_base_512_bs256_lr_1e-04_fp16/epdnew_promoter_2k/model_1000000/lr1e-04_adamw_fp32/run_$N \
                --input_seq_len 512 \
                --batch_size 8 --gradient_accumulation_steps 8 \
                --save_best --iters 1500 \
                --optimizer AdamW \
                --lr 1e-04 --lr_scheduler constant_with_warmup --num_warmup_steps 50 \
                --reset_lr --reset_optimizer --reset_iteration \
                --data_n_workers 2 \
                --log_interval 50 --save_interval 1000000 --valid_interval 50 \
                --optimize_metric f1 --optimize_mode max \
                --clip_grad_value 5.0 \
                --init_checkpoint ./runs/bert_base_512_bs256_lr_1e-04_fp16/model_1000000.pth
done
echo "run_bert_pretraining.py done"
echo "done"