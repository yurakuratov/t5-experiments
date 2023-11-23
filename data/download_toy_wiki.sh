#!/usr/bin/env bash

set -e

mkdir -p toy_wiki
mkdir -p toy_wiki/train
mkdir -p toy_wiki/valid

echo downloading...
wget https://huggingface.co/datasets/yurakuratov/toy_wiki/resolve/main/train/shard_000.jsonl -P ./toy_wiki/train
wget https://huggingface.co/datasets/yurakuratov/toy_wiki/resolve/main/train/shard_001.jsonl -P ./toy_wiki/train
wget https://huggingface.co/datasets/yurakuratov/toy_wiki/resolve/main/valid/5k_docs.jsonl -P ./toy_wiki/valid
echo done