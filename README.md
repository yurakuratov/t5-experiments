# t5-experiments
This repo is based on 🤗 Transfomers implementation of the T5 model and BERT.
T5 data processing pipeline is used from the original [T5 repository](https://github.com/google-research/text-to-text-transfer-transformer) for pre-training (span corruption, prefix-lm) and fine-tuning. BERT data processing pipeline is used from [Megatron-LM](https://github.com/NVIDIA/Megatron-LM).

Multi-gpu and multi-node training with Horovod is supported. APEX/torch.cuda.amp is used for FP16 and mixed-precision training. Sparse Attention from [DeepSpeed](https://www.deepspeed.ai/tutorials/sparse-attention/) is used.

BERT model supports such additional features as pre-attention layer norm, sparse attention, relative position and rotary embeddings.

T5 and BERT pre-training is implemented in `run_(model_type)_pretraining.py` scripts.

Training tools, such as Trainer, are in `lm_experiments_tools` package.

## Installation
There are two main parts in the repository:
- `lm_experiments_tools` module
- training scripts (like bert/t5 pretraining) that use `lm_experiments_tools`

### Install only lm_experiments_tools
`lm_experiments_tools` include Trainers with multi-gpu/node support:
- Trainer with Horovod (distributed training engine) and APEX torch.cuda.amp FP16 for models.
- TrainerAccelerate with Accelerate. Accelerate provides features from PyTorch distributed and DeepSpeed.

To train models with `lm_experiments_tools` trainers, models should be compatible with HF interface (e.g., return loss
if labels are passed).
Most of the scripts in the repo use Trainer from `lm_experiments_tools`.

> note: install torch and optionally horovod, accelerate and deepspeed according to your setup before `lm_experiments_tools` installation.

#### Trainer with Horovod:
Depending on your setup just `pip install horovod` might work.

Building Horovod with NCCL for PyTorch:
```bash
HOROVOD_NCCL_HOME=path/to/nccl HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_WITH_PYTORCH=1 pip install --no-cache-dir horovod[pytorch] --no-binary=horovod
```
check installation with
```bash
horovodrun --check-build
```
For further details check Horovod documentation: https://horovod.readthedocs.io/en/stable/install_include.html

You can get NCCL from https://developer.nvidia.com/nccl

#### Trainer with Accelerate:
requirements:
```
python>=3.8
accelerate==0.24.1
torch>=1.10.0
```
DeepSpeed installation is optional. DeepSpeed versions requirements:
```
deepspeed>=0.9.3,<=0.9.5 and any (?) triton version or no triton installed at all
deepspeed>=0.10.0 with no triton installed or installed triron==2.1.0
```
and finally install `lm_experiments_tools`:
```bash
pip install -e .
```

`lm_experiments_tools` Trainer supports gradient accumulation, logging to tensorboard, saving the best models
based on metrics, custom metrics and data transformations support.
>todo: customize `setup.py` to automate support of different environment setups: [all, hvd, accelerate]

### Install requirements for all experiments from this repository
Full requirements for all experiments are specified in requirements.txt. Install requirements after cloning the repo:
```bash
grep -v "^#" requirements.txt | xargs -n 1 -L 1 pip install
```
> note: installation order is such that t5 package will be installed last

### Install APEX
Install APEX https://github.com/NVIDIA/apex#quick-start
```
git clone https://github.com/NVIDIA/apex
cd apex
# most recent commits may fail to build
git checkout 2386a912164b0c5cfcd8be7a2b890fbac5607c82
# if pip >= 23.1 (ref: https://pip.pypa.io/en/stable/news/#v23-1) which supports multiple `--config-settings` with the same key... 
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
# otherwise
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

apex.amp is moved to torch.cuda.amp https://github.com/NVIDIA/apex/issues/818, but:

speed: `APEX O1` < `torch.cuda.amp` < `APEX O2`

resources (unordered):
 - https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html
 - https://pytorch.org/docs/stable/notes/amp_examples.html
 - https://spell.ml/blog/mixed-precision-training-with-pytorch-Xuk7YBEAACAASJam
 - https://pytorch.org/blog/accelerating-training-on-nvidia-gpus-with-pytorch-automatic-mixed-precision/
 - https://github.com/horovod/horovod/issues/1089
 - https://github.com/NVIDIA/apex/issues/818

### Install DeepSpeed for Sparse Ops
DeepSpeed Sparse attention supports only GPUs with compute compatibility >= 7 (V100, T4, A100), CUDA 10.1, 10.2, 11.0, or 11.1 and runs only in FP16 mode (as of DeepSpeed 0.6.0).

PyTorch>=1.7.1,<=1.10.1 wheels with CUDA 10.2/11.0/11.1 from [pytorch.org](https://pytorch.org/get-started/previous-versions/) can be used.
However, using Sparse Ops with CUDA 11.1 PyTorch wheels would require CUDA 11.3/11.4 to be installed on the system.
Sparse Ops could also be used with PyTorch==1.12.1 CUDA 11.3 wheels, but running DeepSpeed Sparse Ops tests would require modifying them as they check for Torch CUDA version <=11.1.
DeepSpeed fork for Triton 1.1.1 already has updated tests.

Triton 1.0.0 and 1.1.1 requires python<=3.9.

```bash
pip install triton==1.0.0
DS_BUILD_SPARSE_ATTN=1 pip install deepspeed==0.6.0 --global-option="build_ext" --global-option="-j8" --no-cache
```
and check installation with
```bash
ds_report
```
#### Triton 1.1.1
Triton 1.1.1 brings x2 speed-up to sparse operations on A100, but DeepSpeed (0.6.5) currently supports only triton 1.0.0.
DeepSpeed fork with triton 1.1.1 support could be used in the cases where such speed-up is needed:
```bash
pip install triton==1.1.1
git clone https://github.com/yurakuratov/DeepSpeed.git
cd DeepSpeed
DS_BUILD_SPARSE_ATTN=1 pip install -e . --global-option="build_ext" --global-option="-j8" --no-cache
```
and run sparse ops tests with
```bash
cd tests/unit
pytest -v test_sparse_attention.py
```

## BERT pretraining
Data preprocessing [readme](megatron/README.md).

Python script: `run_bert_pretraining.py`

## FP16 for pretraining
The Trainer argument `--fp16` will enable torch.cuda.amp FP16 mixed precision. Adding `--apex_opt_lvl O1` or `--apex_opt_lvl O2` will enable mixed precision with APEX FP16. Check APEX docs for the details https://nvidia.github.io/apex/amp.html#opt-levels.

## Adafactor optimizer
[Adafactor](https://arxiv.org/abs/1804.04235) was used to train such models as T5, BigBird, PaLM and others. Adafactor lowers required memory by keeping moving average of per-parameter second moments factorized.

Adafactor parameters:
- `scale_parameter` - lr is scaled by root mean square of parameter: lr * RMS(p)
- `relative_step` - lr = 1/sqrt(step)
- `warmup_init` - linear warm up from 1e-06 to 0.01 at 10k steps, works only in combination with `relative_step`

Adafactor can be used with constant lr / lr schedulers. In this case, `relative_step` and `warmup_init` should be set to False. `scale_parameter` is does not depend on learning rate schedules and can be used with external learning rates.

example for pretraining scripts:
```bash
--optimizer Adafactor --lr 1e-03 --scale_parameter \
--lr_scheduler constant_with_warmup --num_warmup_steps 10000
```

e.g. for DP config
```json
"optimizer": "Adafactor",
"optimizer_parameters": {
        "lr": 1e-03,
        "weight_decay": 0.0,
        "scale_parameter": true,
        "relative_step": false,
        "warmup_init": false
}
```

## Sparse Attention
BERT model training supports sparse attentions from DeepSpeed.

DeepSpeed Sparse attention docpage -- https://www.deepspeed.ai/tutorials/sparse-attention.

### Configure Sparse Attention
SparseAttention parameters are passed to the model with HF model configuration file:
```json
"sparse_config_cls": "deepspeed.ops.sparse_attention:BigBirdSparsityConfig",
"sparse_attention": {
  "num_heads": 12,
  "block": 16,
  "different_layout_per_head": true,
  "num_sliding_window_blocks": 1,
  "num_global_blocks": 1,
  "num_random_blocks": 1
}
```
You can also check `bert_base_uncased-4L_sparse.json` config example in `models_configs/bert_configs` folder.

## T5 Pre-training
### T5-small baseline
```bash
export CUDA_VISIBLE_DEVICES=4,5; horovodrun --gloo -np 2 python run_t5_pretraining.py \
        --batch_size 32 \
        --gradient_accumulation_steps 2 \
        --save_interval 100000 \
        --log_interval 500 \
        --iters 1100000 \
        --data_path ~/data/ThePile/Wikipedia/preprocessed_shards \
        --model_path ./runs/small_wiki_bs_128 \
        --input_seq_len 512 \
        --target_seq_len 192 \
        --lr 5e-05 \
        --weight_decay 1e-05 \
        --model_cfg ./models_configs/t5_configs/t5-small.json \
        --model_cls modeling_t5:T5ForConditionalGeneration
```

### T5-base with custom layers:
and continue interrupted training
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3; horovodrun --gloo -np 4 python run_t5_pretraining.py \
        --batch_size 8 \
        --gradient_accumulation_steps 4 \
        --save_interval 75000 \
        --log_interval 500 \
        --iters 1000000 --data_path ~/data/ThePile/Wikipedia/preprocessed_shards \
        --model_path ./runs/base_wiki_enc_only_cdq_fixed_pos_wo_tanh \
        --input_seq_len 512 \
        --target_seq_len 192 \
        --lr 5e-05 \
        --weight_decay 1e-05 \
        --model_cls modeling_t5:T5ForConditionalGeneration \
        --model_cfg models_configs/t5_configs/t5-base-only-cdQ.json \
        --init_checkpoint ./runs/base_wiki_enc_only_cdq_fixed_pos_wo_tanh/model_150000.pth
```