import logging
import os
from pathlib import Path

from megatron.data.dataset_utils import get_indexed_dataset_
from megatron.data.bert_dataset import BertDataset
from megatron.tokenizer.tokenizer import _HFAutoTokenizer

import horovod.torch as hvd
from dotenv import load_dotenv
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
from transformers import HfArgumentParser

from lm_experiments_tools import Trainer, TrainerArgs
from lm_experiments_tools.data import MixtureDataset

load_dotenv()

logger_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(format=logger_fmt, level=logging.INFO)
logger = logging.getLogger()

# if CUDA_VISIBLE_DEVICES is not set make all gpus visible
if os.environ.get('CUDA_VISIBLE_DEVICES', None) is None:
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in range(torch.cuda.device_count())])

logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
# first call to torch.cuda.device_count() sets visible gpus, following calls will not change the result
logger.info(f"CUDA DEVICE COUNT: {torch.cuda.device_count()}")

hvd.init()

import transformers  # noqa: E402
from transformers import AutoConfig  # noqa: E402

from lm_experiments_tools.utils import prepare_run, get_cls_by_name, get_optimizer  # noqa: E402
import lm_experiments_tools.optimizers as optimizers  # noqa: E402

# limit # of CPU threads to be used per pytorch worker, otherwise it might use all cpus and throttle gpus
# > 2 fails cause of https://github.com/pytorch/pytorch/issues/56615
# need to upgrade to torch>1.8.1
torch.set_num_threads(2)
# all gpus set with CUDA_VISIBLE_DEVICES are visible to process, indexing from 0 to ...
torch.cuda.set_device(hvd.local_rank())

parser = HfArgumentParser(TrainerArgs)
parser.add_argument('--data_path', type=str, nargs='*', help='path with the indexed data in bin format or multiple '
                    'paths. Datasets would be merged into single dataset.')
parser.add_argument('--valid_data_path', type=str, help='path with the indexed data in bin format')
parser.add_argument('--validate_only', action='store_true', default=False,
                    help='Skip training and run only validation. (default: False)')
parser.add_argument('--working_dir', type=str, default='.',
                    help='working dir, should be a dir with t5-experiments repo (default: .)')
parser.add_argument('--seed', type=int, default=42, help='random seed')

# bert data args
parser.add_argument('--data_impl', type=str, default='mmap', choices=['lazy', 'cached', 'mmap'],
                    help='type of dataset produced by preprocess_data.py')
parser.add_argument('--data_name', type=str, default='', help='used to save/load samples mapping .npy index')
parser.add_argument('--data_n_epochs', type=int, default=None, help='pre-generate samples for data_epochs')
parser.add_argument('--data_n_samples', type=int, default=None, help='pre-generate data_n_samples')
parser.add_argument('--data_skip_warmup', action='store_true', default=False,
                    help='skip dataset warmup (default: False)')
parser.add_argument('--input_seq_len', type=int, default=128, help='input sequnce length (default: 128).')
parser.add_argument('--mlm_prob', type=float, default=0.15, help='MLM task prob (default: 0.15)')
parser.add_argument('--short_seq_prob', type=float, default=0.1, help='short sequence prob (default: 0.1)')
parser.add_argument('--use_nsp', type=int, default=1, help='use next sentence prediction task (default: 1)')
parser.add_argument('--data_n_workers', type=int, default=2, help='number of dataloader workers (default: 2)')

# model args
parser.add_argument('--model_cfg', type=str, help='path to model configuration file (default: None)')
parser.add_argument('--model_cls', type=str, default='transformers:BertForPreTraining',
                    help='model class name to use (default: transformers:BertForPreTraining)')

# rmt args
parser.add_argument('--backbone_cls', type=str, default=None,
                    help='backbone class name to use for RMT')
parser.add_argument('--backbone_checkpoint', type=str,
                    help='pre-trained backbone checkpoint (default: None).')
parser.add_argument('--input_size', type=int, default=None, help='maximal input size of the backbone model')
parser.add_argument('--num_mem_tokens', type=int, default=None, help='number of memory tokens.')
parser.add_argument('--max_n_segments', type=int, default=1, help='maximal segment number')
parser.add_argument('--bptt_depth', type=int, default=-1, help='max number of previous segments in gradient computation.')
parser.add_argument('--segment_alignment', type=str, help='segment alignment', default='right',
                    choices=['right', 'left', 'center'])

# tokenizer
# todo: add wordpiece tokenizers support?
parser.add_argument('--tokenizer', type=str, default=None, help='path or name of pre-trained HF Tokenizer')

# optimizer args
parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer name: AdamW, Adafactor. (default: AdamW)')
parser.add_argument('--weight_decay', type=float, default=0.0, help='optimizer weight decay (default: 0.0)')
parser.add_argument('--scale_parameter', action='store_true', default=False,
                    help='Adafactor scale_parameter (default: False)')
parser.add_argument('--relative_step', action='store_true', default=False,
                    help='Adafactor relative_step (default: False)')
parser.add_argument('--warmup_init', action='store_true', default=False,
                    help='Adafactor warmup_init (default: False)')


def main():
    args = parser.parse_args()
    # set current working dir
    args.working_dir = str(Path(args.working_dir).expanduser().absolute())
    os.chdir(args.working_dir)

    if args.use_nsp:
        raise NotImplementedError

    prepare_run(args, logger, logger_fmt)

    if hvd.rank() == 0:
        logger.info(f'hvd size: {hvd.size()}')
        logger.info(f'FP16: {args.fp16}')

    tokenizer = _HFAutoTokenizer(args.tokenizer)
    # get train dataset
    train_datasets = []
    for d_path in args.data_path:
        d_path = Path(d_path).expanduser().absolute()
        if hvd.rank() == 0:
            logger.info(f'preparing training data from: {d_path}')
        train_data_index = get_indexed_dataset_(str(d_path), args.data_impl, skip_warmup=args.data_skip_warmup)
        train_datasets += [BertDataset(indexed_dataset=train_data_index, masked_lm_prob=args.mlm_prob,
                                       short_seq_prob=args.short_seq_prob, binary_head=args.use_nsp,
                                       tokenizer=tokenizer, name=args.data_name, data_prefix=str(d_path),
                                       num_epochs=args.data_n_epochs, max_num_samples=args.data_n_samples,
                                       max_seq_length=args.input_seq_len, mask_label_id=-100, seed=args.seed)]
    if len(train_datasets) > 1:
        if hvd.rank() == 0:
            logger.info('building mixture dataset...')
        train_dataset = MixtureDataset(train_datasets)
    else:
        train_dataset = train_datasets[0]
    # shuffle train data each epoch (one loop over train_dataset) & drop last batch
    train_sampler = DistributedSampler(train_dataset, rank=hvd.rank(), num_replicas=hvd.size(), shuffle=True,
                                       drop_last=True, seed=args.seed)

    per_worker_batch_size = args.batch_size * args.gradient_accumulation_steps
    global_batch_size = per_worker_batch_size * hvd.size()
    kwargs = {'pin_memory': True, 'num_workers': args.data_n_workers}
    # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
    # issues with Infiniband implementations that are not fork-safe
    if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
            mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
        # kwargs['multiprocessing_context'] = 'forkserver'
        ...
    train_dataloader = DataLoader(train_dataset, batch_size=per_worker_batch_size, sampler=train_sampler, **kwargs)
    # get validation dataset
    if args.valid_data_path:
        if hvd.rank() == 0:
            logger.info(f'preparing validation data from: {args.valid_data_path}')
        valid_data_path = Path(args.valid_data_path).expanduser().absolute()
        valid_data_index = get_indexed_dataset_(str(valid_data_path), args.data_impl, skip_warmup=args.data_skip_warmup)
        valid_dataset = BertDataset(indexed_dataset=valid_data_index, masked_lm_prob=args.mlm_prob,
                                    short_seq_prob=args.short_seq_prob, binary_head=args.use_nsp, tokenizer=tokenizer,
                                    name=args.data_name, data_prefix=str(valid_data_path),
                                    num_epochs=1, max_num_samples=None,  # take all validation data
                                    max_seq_length=args.input_seq_len, mask_label_id=-100, seed=args.seed)
        valid_sampler = DistributedSampler(valid_dataset, rank=hvd.rank(), num_replicas=hvd.size(), shuffle=False)
        valid_dataloader = DataLoader(valid_dataset, batch_size=per_worker_batch_size, sampler=valid_sampler, **kwargs)
        if args.valid_interval is None:
            args.valid_interval = args.log_interval
    else:
        valid_dataloader = None
        if hvd.rank() == 0:
            logger.info('No validation data is used.')

    # define model
    model_cfg = AutoConfig.from_pretrained(args.model_cfg)
    # todo: get model class from model_cfg?
    model_cls = get_cls_by_name(args.backbone_cls)
    if hvd.rank() == 0:
        logger.info(f'Using backbone model class: {model_cls}')
    model = model_cls(config=model_cfg)

    if args.backbone_checkpoint is not None:
        if hvd.rank() == 0:
            logger.info(f'loading pre-trained backbone from {args.backbone_checkpoint}')
        checkpoint = torch.load(args.backbone_checkpoint, map_location='cpu')
        missing_k, unexpected_k = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if len(missing_k) != 0 and hvd.rank() == 0:
            logger.info(f'{missing_k} were not loaded from checkpoint! These parameters were randomly initialized.')
        if len(unexpected_k) != 0 and hvd.rank() == 0:
            logger.info(f'{unexpected_k} were found in checkpoint, but model is not expecting them!')

    # rmt with memory
    if args.num_mem_tokens is not None:
        rmt_config = {
            'num_mem_tokens': args.num_mem_tokens,
            'max_n_segments': args.max_n_segments,
            'segment_alignment': args.segment_alignment,
            'input_size': args.input_size,
            'bptt_depth': args.bptt_depth,
            'sum_loss': True,
            'tokenizer': tokenizer.tokenizer,
        }
        rmt_cls = get_cls_by_name(args.model_cls)
        if hvd.rank() == 0:
            logger.info(f'Wrapping in: {rmt_cls}')
        model = rmt_cls(model, **rmt_config)

        if args.input_seq_len / model.segment_size > rmt_config['max_n_segments']:
            raise RuntimeError(f"Input sequence does not fully fit into selected number of segments: "
                               f"{args.input_seq_len} / {model.segment_size} > {rmt_config['max_n_segments']}")

    # define optimizer
    # todo: move to trainer?
    optimizer_cls = get_optimizer(args.optimizer)
    if optimizer_cls is None:
        raise RuntimeError(f'{args.optimizer} was not found in optimizers, torch.optim, transformers.optimization')

    if hvd.rank() == 0:
        logger.info(f'Using optimizer class: {optimizer_cls}')

    # todo: group optimizer params
    if optimizer_cls in [transformers.optimization.Adafactor, optimizers.Adafactor]:
        # https://github.com/huggingface/transformers/pull/9751/files -> transformers 4.3.0
        optimizer = optimizer_cls(model.parameters(), lr=args.lr,
                                  scale_parameter=args.scale_parameter,
                                  relative_step=args.relative_step,
                                  warmup_init=args.warmup_init,
                                  weight_decay=args.weight_decay)
    else:
        optimizer = optimizer_cls(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def batch_transform_fn(batch):
        b = {
            'input_ids': batch['text'],
            'token_type_ids': batch['types'],
            'attention_mask': batch['padding_mask'],
            'labels': batch['labels']}
        if args.use_nsp:
            b['next_sentence_label'] = batch['is_random']
        return b

    def batch_metrics_fn(batch, output):
        # output - result of model(batch) call
        # only stateless metrics could be get in such way - metrics are averaged over batches
        # loss is a default metric, this function should be used if other metrics than loss should be logged
        metrics = {}
        for k in output:
            if 'loss' in k:
                metrics[k] = output[k].detach()
        # losses
        if 'mlm_loss' in output:
            metrics['loss_mlm'] = output['mlm_loss']
        if 'nsp_loss' in output:
            metrics['loss_nsp'] = output['nsp_loss']
        # compute batch-lvl mlm accuracy, cause exact mlm accuracy would require to store [n_steps, bs, seq_len] x 2
        # which could be too large

        logits_segm = output['logits_segm'][0]
        labels_segm = output['labels_segm'][0]
        y_rmt, p_rmt = [], []
        for i in range(len(logits_segm)):
            y_segm = labels_segm[i]
            if y_segm is None:
                continue
            p_segm = torch.argmax(logits_segm[i].detach(), dim=-1)
            n = (y_segm != -100).sum()
            metrics[f'accuracy_mlm_segm_{i}'] = (y_segm == p_segm).sum() / n if n != 0 else torch.tensor(0.0)
            y_rmt += [y_segm.clone()]
            p_rmt += [p_segm]

        y_rmt = torch.cat(y_rmt)
        p_rmt = torch.cat(p_rmt)
        n = (y_rmt != -100).sum()
        metrics['accuracy_mlm'] = (y_rmt == p_rmt).sum() / n if n != 0 else torch.tensor(0.0)
        return metrics

    def keep_for_metrics_fn(batch, output):
        # select data from batch and model output that would be used to compute metrics
        data = {}
        if 'seq_relationship_logits' in output:
            data['next_sentence_label'] = batch['next_sentence_label']
            data['nsp_predictions'] = torch.argmax(output['seq_relationship_logits'].detach(), dim=-1)
        return data

    def metrics_fn(data):
        # compute metrics based on stored labels, predictions, ...
        metrics = {}
        # nsp accuracy
        if 'next_sentence_label' in data:
            y, p = data['next_sentence_label'], data['nsp_predictions']
            metrics['accuracy_nsp'] = (p == y).sum() / len(y)
        return metrics

    trainer = Trainer(args, model, optimizer, train_dataloader, valid_dataloader, train_sampler,
                      batch_transform_fn, batch_metrics_fn, keep_for_metrics_fn, metrics_fn)

    if not args.validate_only:
        # train loop
        trainer.train()
    else:
        trainer.validate(valid_dataloader)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.exception(e)
