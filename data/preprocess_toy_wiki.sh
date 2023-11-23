#!/usr/bin/env bash

set -e

echo build data preprocessing ops
cd ../megatron/data
make
cd ..

for SPLIT in train valid
do
    echo start preprocessing: $SPLIT
    python preprocess_data.py \
        --input ../data/toy_wiki/${SPLIT} \
        --output-prefix ../data/toy_wiki/${SPLIT} \
        --dataset-impl mmap \
        --tokenizer-type HFTokenizer \
        --tokenizer-name-or-path ../vocabs/bert-base-uncased \
        --split-sentences --workers 8
done