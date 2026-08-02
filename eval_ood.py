#!/usr/bin/env python3
"""Post-hoc OOD length evaluation for finished MAD training runs (Option A).

Train as usual at length L, then evaluate a saved checkpoint on test data with
seq_len L' > L. Writes ``ood_results.csv`` beside the training ``results.csv``
(does not modify training artifacts).

Example:
  uv run python -m eval_ood \\
      --log-path logs/.../t-M_..._model-... \\
      --layers bdlru-sel-wd1-d128-h128 mlp \\
      --layer-overrides implementation=triton_auto \\
      --eval-seq-len 128 512 1024
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from dataclasses import fields

import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from mad.configs import MADConfig, MADModelConfig
from mad.data import generate_or_load_dataset
from mad.helpers import infer_dim_from_layer_names, parse_layer_overrides
from mad.model import PLModelWrap
from mad.registry import layer_registry, validate_layer_names


def get_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        '--log-path', type=str, required=True,
        help='finished training run directory (must contain results.csv and checkpoints/)',
    )
    p.add_argument(
        '--layers', nargs='+', required=True,
        help='layer names used when training (architecture is not stored in the checkpoint)',
    )
    p.add_argument(
        '--eval-seq-len', type=int, nargs='+', default=[128, 512, 1024],
        help='one or more evaluation sequence lengths (default: 128 512 1024)',
    )
    p.add_argument(
        '--checkpoint', type=str, default=None,
        help='checkpoint filename or path (default: checkpoints/best.ckpt, else last.ckpt)',
    )
    p.add_argument('--backbone', type=str, default='language-model')
    p.add_argument('--dim', type=int, default=128)
    p.add_argument(
        '--layer-overrides', nargs='*', default=[],
        help='override layer YAML fields, e.g. implementation=triton_auto',
    )
    p.add_argument('--batch-size', type=int, default=None, help='override eval batch size')
    p.add_argument('--num-test-examples', type=int, default=None, help='override number of OOD test examples')
    p.add_argument('--data-path', type=str, default=None, help='override data root (default: from checkpoint MADConfig)')
    p.add_argument('--num-data-workers', type=int, default=0)
    p.add_argument('--devices', type=str, default='0,')
    p.add_argument('--precision', type=str, default=None)
    p.add_argument('--accelerator', type=str, default='cuda')
    p.add_argument(
        '--allow-incomplete', action='store_true',
        help='allow evaluating runs that do not yet have results.csv',
    )
    p.add_argument(
        '--force', action='store_true',
        help='re-evaluate eval lengths already present in ood_results.csv',
    )
    p.add_argument(
        '--prime-only', action='store_true',
        help='only generate/load OOD test caches; skip model evaluation (safe before parallel eval)',
    )
    p.add_argument(
        '--output', type=str, default=None,
        help='results csv path (default: <log-path>/ood_results.csv)',
    )
    return vars(p.parse_args())


def resolve_checkpoint(log_path: str, checkpoint: str | None) -> str:
    if checkpoint is not None:
        path = checkpoint if os.path.isabs(checkpoint) or os.path.dirname(checkpoint) else os.path.join(
            log_path, 'checkpoints', checkpoint
        )
        if not os.path.isfile(path):
            # also accept a bare relative path as given
            if os.path.isfile(checkpoint):
                return checkpoint
            raise FileNotFoundError(f'Checkpoint not found: {path}')
        return path

    for name in ('best.ckpt', 'last.ckpt'):
        path = os.path.join(log_path, 'checkpoints', name)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f'No checkpoints/best.ckpt or checkpoints/last.ckpt under "{log_path}"'
    )


def load_mad_config_from_checkpoint(ckpt_path: str) -> tuple[MADConfig, dict]:
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    hp = ckpt.get('hyper_parameters') or {}
    mad_config = hp.get('mad_config')
    if mad_config is None:
        raise KeyError(f'Checkpoint "{ckpt_path}" has no hyper_parameters.mad_config')
    if not isinstance(mad_config, MADConfig):
        # older / re-hydrated dict form
        cfg = MADConfig()
        cfg.update_from_kwargs(dict(mad_config))
        mad_config = cfg
    return mad_config, ckpt


def clone_mad_config(mad_config: MADConfig, **overrides) -> MADConfig:
    cfg = MADConfig()
    for field in fields(MADConfig):
        setattr(cfg, field.name, copy.deepcopy(getattr(mad_config, field.name)))
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise AttributeError(f'MADConfig has no field {key!r}')
        setattr(cfg, key, value)
    return cfg


def build_model(
    layers: list[str],
    dim: int,
    vocab_size: int,
    max_length: int,
    backbone: str,
    layer_overrides: dict | None,
):
    model_config = MADModelConfig(
        layers=layers,
        backbone=backbone,
        dim=dim,
        vocab_size=vocab_size,
        max_length=max_length,
        layer_overrides=layer_overrides or None,
    )
    return model_config.build_model_from_registry()


def load_weights(wrapped: PLModelWrap, ckpt: dict, strict: bool = True) -> None:
    state_dict = ckpt['state_dict']
    try:
        wrapped.load_state_dict(state_dict, strict=strict)
    except RuntimeError as exc:
        raise RuntimeError(
            'Failed to load checkpoint weights into the eval model. '
            'If this is a max_length / positional-embedding size mismatch, '
            'either evaluate with a length-agnostic backbone or retrain with '
            f'max_length >= eval_seq_len.\nOriginal error: {exc}'
        ) from exc


def already_done(output_csv: str, eval_seq_len: int, ckpt_name: str) -> bool:
    if not os.path.isfile(output_csv):
        return False
    df = pd.read_csv(output_csv)
    if df.empty or 'eval_seq_len' not in df.columns:
        return False
    mask = df['eval_seq_len'].astype(int) == int(eval_seq_len)
    if 'checkpoint' in df.columns:
        mask &= df['checkpoint'].astype(str) == str(ckpt_name)
    return bool(mask.any())


def append_result(output_csv: str, row: dict) -> None:
    df_new = pd.DataFrame([row])
    if os.path.isfile(output_csv):
        df = pd.read_csv(output_csv)
        df = pd.concat([df, df_new], ignore_index=True)
    else:
        df = df_new
    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
    df.to_csv(output_csv, index=False)


def evaluate_one(
    *,
    mad_config_train: MADConfig,
    ckpt: dict,
    ckpt_path: str,
    layers: list[str],
    dim: int,
    backbone: str,
    layer_overrides: dict,
    eval_seq_len: int,
    batch_size: int,
    num_test_examples: int,
    data_path: str,
    num_data_workers: int,
    devices: str,
    precision: str,
    accelerator: str,
    prime_only: bool,
) -> dict | None:
    train_seq_len = mad_config_train.seq_len
    if eval_seq_len < train_seq_len:
        print(
            f'WARNING: eval_seq_len={eval_seq_len} < train_seq_len={train_seq_len} '
            '(still running; this is in-distribution / shorter, not OOD-longer).'
        )

    eval_cfg = clone_mad_config(
        mad_config_train,
        seq_len=eval_seq_len,
        num_test_examples=num_test_examples,
        data_path=data_path,
        batch_size=batch_size,
        num_data_workers=num_data_workers,
        devices=devices,
        precision=precision,
        accelerator=accelerator,
        compile=False,
    )

    instance_kwargs = dict(eval_cfg.instance_fn_kwargs)
    # Own RNG so we do not mutate the train config's generator state.
    instance_kwargs['rng'] = __import__('numpy').random.default_rng(eval_cfg.seed)

    test_ds = generate_or_load_dataset(
        instance_fn=eval_cfg.instance_fn,
        instance_fn_kwargs=instance_kwargs,
        data_path=eval_cfg.test_dataset_path,
        num_examples=eval_cfg.num_test_examples,
        num_workers=eval_cfg.num_data_workers,
        is_training=False,
        verbose=True,
    )
    print(
        f'OOD test data ready: path={eval_cfg.test_dataset_path} '
        f'shape={tuple(test_ds.inputs.shape)}'
    )

    if prime_only:
        return None

    model = build_model(
        layers=layers,
        dim=dim,
        vocab_size=eval_cfg.vocab_size,
        max_length=eval_seq_len,
        backbone=backbone,
        layer_overrides=layer_overrides,
    )
    wrapped = PLModelWrap(model=model, mad_config=eval_cfg)
    load_weights(wrapped, ckpt, strict=True)

    test_dl = DataLoader(
        dataset=test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_data_workers,
        persistent_workers=num_data_workers > 0,
    )

    torch.set_float32_matmul_precision('high')
    trainer = pl.Trainer(
        accelerator=accelerator if torch.cuda.is_available() else 'cpu',
        devices=devices,
        precision=precision,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
    )
    metrics = trainer.validate(model=wrapped, dataloaders=test_dl, weights_only=False)[0]
    param_num = sum(p.numel() for p in wrapped.parameters() if p.requires_grad)

    return {
        'train_seq_len': train_seq_len,
        'eval_seq_len': eval_seq_len,
        'test_acc': metrics['test/Accuracy_epoch'],
        'test_ppl': metrics['test/Perplexity_epoch'],
        'test_loss': metrics['test/Loss_epoch'],
        'model_size': param_num,
        'checkpoint': os.path.basename(ckpt_path),
        'task': mad_config_train.task,
        'seed': mad_config_train.seed,
    }


def main() -> None:
    args = get_args()
    log_path = os.path.abspath(args['log_path'])
    if not os.path.isdir(log_path):
        raise FileNotFoundError(f'--log-path does not exist: {log_path}')

    results_csv = os.path.join(log_path, 'results.csv')
    if not os.path.isfile(results_csv) and not args['allow_incomplete']:
        raise FileNotFoundError(
            f'No results.csv in "{log_path}". Refusing to eval an unfinished run '
            '(pass --allow-incomplete to override).'
        )

    validate_layer_names(args['layers'])
    inferred_dim = infer_dim_from_layer_names(args['layers'])
    dim = inferred_dim if inferred_dim is not None else args['dim']
    if inferred_dim is not None and inferred_dim != args['dim']:
        print(f'Using dim={inferred_dim} implied by layer name(s) (CLI had {args["dim"]}).')

    backbone = args['backbone']
    layer_overrides = parse_layer_overrides(args['layer_overrides'])

    ckpt_path = resolve_checkpoint(log_path, args['checkpoint'])
    mad_config_train, ckpt = load_mad_config_from_checkpoint(ckpt_path)
    print(f'Loaded checkpoint: {ckpt_path}')
    print(
        f'Train config: task={mad_config_train.task} seq_len={mad_config_train.seq_len} '
        f'vocab_size={mad_config_train.vocab_size} seed={mad_config_train.seed}'
    )

    if mad_config_train.task == 'compression' and backbone != 'autoencoder':
        print('Setting backbone to "autoencoder" (required for compression).')
        backbone = 'autoencoder'

    batch_size = args['batch_size'] or mad_config_train.batch_size
    num_test_examples = args['num_test_examples'] or mad_config_train.num_test_examples
    data_path = args['data_path'] or mad_config_train.data_path
    precision = args['precision'] or mad_config_train.precision
    output_csv = args['output'] or os.path.join(log_path, 'ood_results.csv')
    ckpt_name = os.path.basename(ckpt_path)

    rows = []
    for eval_seq_len in args['eval_seq_len']:
        if not args['force'] and not args['prime_only'] and already_done(output_csv, eval_seq_len, ckpt_name):
            print(f'SKIP eval_seq_len={eval_seq_len} (already in {output_csv}; use --force to redo)')
            continue

        print(f'\n===== OOD eval: train_seq_len={mad_config_train.seq_len} -> eval_seq_len={eval_seq_len} =====')
        row = evaluate_one(
            mad_config_train=mad_config_train,
            ckpt=ckpt,
            ckpt_path=ckpt_path,
            layers=args['layers'],
            dim=dim,
            backbone=backbone,
            layer_overrides=layer_overrides,
            eval_seq_len=eval_seq_len,
            batch_size=batch_size,
            num_test_examples=num_test_examples,
            data_path=data_path,
            num_data_workers=args['num_data_workers'],
            devices=args['devices'],
            precision=precision,
            accelerator=args['accelerator'],
            prime_only=args['prime_only'],
        )
        if row is None:
            continue
        append_result(output_csv, row)
        rows.append(row)
        print(
            f"eval_seq_len={eval_seq_len}  test_acc={row['test_acc']:.4f}  "
            f"test_ppl={row['test_ppl']:.4f}  test_loss={row['test_loss']:.4f}"
        )

    if args['prime_only']:
        print('Prime-only done (no evaluation).')
        return

    if rows:
        print(f'\nWrote {len(rows)} row(s) to {output_csv}')
    else:
        print(f'No new evaluations. Existing results (if any): {output_csv}')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise
