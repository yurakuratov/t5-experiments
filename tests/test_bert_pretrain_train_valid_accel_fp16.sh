#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=1,2 NP=2 ./test_bert_pretrain_train_valid.sh
set -e
cd ..

MODEL_PATH=./tests/runs/test_bert_pretrain_accel_fp16

accelerate launch --num_processes $NP --mixed_precision fp16 --config_file ./tests/accelerate.yaml run_bert_pretraining_accel.py \
--data_path ./data/toy_wiki/train_text_sentence --valid_data_path ./data/toy_wiki/valid_text_sentence \
--tokenizer ./vocabs/bert-base-uncased/ --model_cfg ./models_configs/bert_configs/bert_base_uncased-4L.json \
--model_path $MODEL_PATH \
--batch_size 16 --gradient_accumulation_steps 2 \
--data_n_epochs 1 --lr 3e-04 --save_best --iters 100 --log_interval 25 --save_interval 50 --valid_interval 50
echo "run_bert_pretraining.py done"
echo "cleaning..."
rm -rf $MODEL_PATH
echo "done"
