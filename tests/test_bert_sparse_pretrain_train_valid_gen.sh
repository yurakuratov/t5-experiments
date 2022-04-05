#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=1,2 NP=2 ./test_bert_sparse_pretrain_train_valid.sh
set -e
cd ..

horovodrun --gloo -np $NP python run_bert_pretraining.py \
--data_path ./data/toy_genom/train_text_sentence --valid_data_path ./data/toy_genom/valid_text_sentence \
--tokenizer ./vocabs/genom_32k/ --model_cfg ./bert_configs/L4-H768-A12-V32k-L512_sparse_rel_preln.json \
--model_cls modeling_bert:BertForPreTraining \
--model_path ./tests/runs/test_bert_pretrain \
--batch_size 16 --gradient_accumulation_steps 2 \
--data_n_epochs 1 --lr 3e-04 --save_best --iters 100 --log_interval 25 --save_interval 50 --valid_interval 50 \
--fp16 --apex_opt_lvl O2
echo "run_bert_pretraining.py done"
echo "cleaning..."
rm -rf ./tests/runs/test_bert_pretrain
echo "done"
