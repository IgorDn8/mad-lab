"""Benchmark the training (or inference) speed of MAD architectures.

Times forward/backward/optimizer-step throughput of a model across a sweep of a
single task variable (by default: sequence length, up to 2**14), on a *cached*
synthetic sequence-modelling dataset. Speed is measured independently of task
convergence, so no real MAD data is required.

Example (LSTM on selective-copying, vocab 32, scaling sequence length):

    uv run python speed_benchmark.py \
        --layers lstm-d128-h128 mlp \
        --task selective-copying --vocab-size 32 \
        --sweep-key seq_len \
        --sweep-values 256 512 1024 2048 4096 8192 16384 \
        --precision 32

Compare architectures by re-running with different --layers and combining the
resulting CSVs.
"""

import os
import csv
import argparse
import typing as tp

import torch

from mad.registry import layer_registry, model_registry
from mad.configs import load_yml
from mad.paths import get_base_path
from mad.speed import get_or_create_synthetic_dataset, time_model


# powers of two from 256 up to 2**14 (16384)
DEFAULT_SWEEP_VALUES = [2 ** e for e in range(8, 15)]


def get_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # model:
    parser.add_argument('--layers', nargs='+', required=True,
                        help='layer names composing the model (see mad/registry.py)')
    parser.add_argument('--dim', type=int, default=128, help='model width')
    parser.add_argument('--backbone', type=str, default=None,
                        choices=[None, 'language-model', 'autoencoder'],
                        help='override model backbone (default: inferred from task)')
    parser.add_argument('--layer-overrides', nargs='*', default=[], metavar='KEY=VALUE',
                        help='override layer-config fields for every layer, e.g. '
                             '`--layer-overrides implementation=hopscan`')

    # task / sweep:
    parser.add_argument('--task', type=str, default='selective-copying',
                        help='task name (only used to pick a backbone; data is synthetic)')
    parser.add_argument('--vocab-size', type=int, default=32, help='token vocabulary size')
    parser.add_argument('--sweep-key', type=str, default='seq_len',
                        choices=['seq_len', 'batch_size', 'dim', 'vocab_size'],
                        help='which variable to sweep')
    parser.add_argument('--sweep-values', nargs='+', type=int, default=DEFAULT_SWEEP_VALUES,
                        help='values for the swept variable (default: powers of two up to 2**14)')

    # timing:
    parser.add_argument('--batch-size', type=int, default=32, help='batch size (base value; may be reduced on OOM)')
    parser.add_argument('--precision', type=str, default='bf16', choices=['bf16', 'fp16', '32'],
                        help='compute precision (use 32 for the fork LSTM/BD-LRU/H-LRU layers)')
    parser.add_argument('--warmup', type=int, default=5, help='untimed warmup steps')
    parser.add_argument('--iters', type=int, default=20, help='timed steps to average over')
    parser.add_argument('--train', action=argparse.BooleanOptionalAction, default=True,
                        help='if set, time fwd+bwd+optimizer; otherwise fwd only (inference)')
    parser.add_argument('--oom-retry', action=argparse.BooleanOptionalAction, default=True,
                        help='on CUDA OOM, halve the batch size and retry (down to 1)')

    # data caching:
    parser.add_argument('--num-samples', type=int, default=512,
                        help='number of synthetic sequences to cache per (vocab, seq_len)')
    parser.add_argument('--data-path', type=str, default='./speed_data',
                        help='root dir for the cached synthetic datasets')
    parser.add_argument('--seed', type=int, default=0, help='seed for synthetic data generation')

    # output:
    parser.add_argument('--out', type=str, default='./speed_results.csv',
                        help='CSV path; results are appended if the file exists')

    return parser.parse_args()


def _parse_override_value(raw: str):
    """Coerce a KEY=VALUE string value to int/float/bool, else leave as str."""
    low = raw.lower()
    if low in ('true', 'false'):
        return low == 'true'
    for cast in (int, float):
        try:
            return cast(raw)
        except ValueError:
            pass
    return raw


def parse_layer_overrides(items: tp.List[str]) -> dict:
    overrides = {}
    for item in items:
        if '=' not in item:
            raise ValueError(f"--layer-overrides expects KEY=VALUE, got {item!r}")
        key, value = item.split('=', 1)
        overrides[key] = _parse_override_value(value)
    return overrides


def build_model(layers: tp.List[str], dim: int, vocab_size: int, max_length: int,
                backbone: str, overrides: tp.Optional[dict] = None):
    layer_mods = [layer_registry[l]['module'] for l in layers]
    missing = [l for l, m in zip(layers, layer_mods) if m is None]
    if missing:
        raise RuntimeError(
            f"layer(s) {missing} are unavailable in this environment "
            f"(install the required extra, e.g. `uv sync --extra cuda`)."
        )
    layer_cfgs = [load_yml(os.path.join(get_base_path(), layer_registry[l]['cfg'])) for l in layers]
    for cfg in layer_cfgs:
        cfg['dim'] = dim
        cfg['max_length'] = max_length
        # only override keys the layer config actually defines, so we don't inject
        # unrelated kwargs into layers that don't accept them.
        for key, value in (overrides or {}).items():
            if key in cfg:
                cfg[key] = value
    return model_registry[backbone](
        dim=dim, vocab_size=vocab_size, layers=layer_mods,
        layer_cfgs=layer_cfgs, max_length=max_length,
    )


def run_one(args, value: int) -> dict:
    """Build + time the model for a single swept value, with OOM retry."""
    # resolve per-run parameters from the swept variable
    seq_len = value if args.sweep_key == 'seq_len' else 256
    dim = value if args.sweep_key == 'dim' else args.dim
    vocab_size = value if args.sweep_key == 'vocab_size' else args.vocab_size
    batch_size = value if args.sweep_key == 'batch_size' else args.batch_size

    backbone = args.backbone or ('autoencoder' if args.task == 'compression' else 'language-model')

    # cached synthetic dataset (generated once, reused afterwards)
    inputs, targets, dataset_dir, created = get_or_create_synthetic_dataset(
        data_path=args.data_path, vocab_size=vocab_size, seq_len=seq_len,
        num_samples=max(args.num_samples, batch_size), seed=args.seed,
    )
    print(f"[{args.sweep_key}={value}] dataset {'created' if created else 'loaded'}: {dataset_dir}")

    overrides = parse_layer_overrides(args.layer_overrides)
    while batch_size >= 1:
        try:
            model = build_model(args.layers, dim, vocab_size, seq_len, backbone, overrides)
            result = time_model(
                model, inputs, targets, batch_size=batch_size,
                precision=args.precision, warmup=args.warmup, iters=args.iters,
                train=args.train,
            )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return result
        except torch.cuda.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if not args.oom_retry or batch_size == 1:
                return {'status': 'OOM', 'batch_size': batch_size}
            batch_size //= 2
            print(f"  OOM -> retrying with batch_size={batch_size}")
        except RuntimeError as e:
            if 'out of memory' in str(e).lower() and args.oom_retry and batch_size > 1:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                batch_size //= 2
                print(f"  OOM -> retrying with batch_size={batch_size}")
                continue
            return {'status': f'error: {type(e).__name__}: {str(e)[:120]}', 'batch_size': batch_size}


def main():
    args = get_args()
    overrides = parse_layer_overrides(args.layer_overrides)
    model_id = '-'.join(layer_registry[l]['shorthand'] for l in args.layers)
    # distinguish runs that differ only by a layer override (e.g. implementation)
    override_suffix = ''.join(f'[{k}={v}]' for k, v in overrides.items())
    model_id += override_suffix
    print(f"Benchmarking model '{model_id}' on '{args.task}', sweeping {args.sweep_key} over {args.sweep_values}\n")

    rows = []
    for value in args.sweep_values:
        result = run_one(args, value) or {'status': 'unknown'}
        result.update({
            'model': model_id,
            'layers': ' '.join(args.layers),
            'overrides': override_suffix,
            'task': args.task,
            'precision': args.precision,
            'train': args.train,
            args.sweep_key: value,
        })
        rows.append(result)
        _print_row(args.sweep_key, value, result)

    _write_csv(args.out, rows)
    print(f"\nWrote {len(rows)} rows to {args.out}")


def _print_row(sweep_key: str, value: int, r: dict) -> None:
    if r.get('status') == 'ok':
        print(
            f"  {sweep_key}={value:>6}  bs={r['batch_size']:>4}  "
            f"step={r['step_ms']:8.2f} ms  {r['tokens_per_s']:>12.0f} tok/s  "
            f"{r['samples_per_s']:8.1f} samp/s  peak_mem={r['peak_mem_mb']:8.1f} MB  "
            f"params={r['params']/1e6:.3f}M"
        )
    else:
        print(f"  {sweep_key}={value:>6}  -> {r.get('status')}")


def _write_csv(path: str, rows: tp.List[dict]) -> None:
    fieldnames = sorted({k for r in rows for k in r})
    write_header = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()
