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

Outputs are organized per benchmark: pass ``--name <benchmark>`` and results go
to ``results/<benchmark>/results.csv`` with figures auto-emitted to
``figures/<benchmark>/<metric>.png`` (disable with ``--no-plot``).
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

# per-benchmark output roots (kept out of the repo root; see .gitignore)
RESULTS_ROOT = 'results'
FIGURES_ROOT = 'figures'
# metrics auto-plotted at the end of a run (all log-log)
PLOT_METRICS = ['tokens_per_s', 'step_ms', 'peak_mem_mb']


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
                             '`--layer-overrides implementation=hopscan_custom`')

    # task / sweep:
    parser.add_argument('--task', type=str, default='selective-copying',
                        help='task name (only used to pick a backbone; data is synthetic)')
    parser.add_argument('--vocab-size', type=int, default=32, help='token vocabulary size')
    parser.add_argument('--sweep-key', type=str, default='seq_len',
                        choices=['seq_len', 'batch_size', 'dim', 'vocab_size',
                                 'hidden_dim', 'window_dim'],
                        help='which variable to sweep (hidden_dim/window_dim are '
                             'applied as per-run layer-config overrides)')
    parser.add_argument('--sweep-values', nargs='+', type=int, default=DEFAULT_SWEEP_VALUES,
                        help='values for the swept variable (default: powers of two up to 2**14)')
    parser.add_argument('--fixed-seq-len', type=int, default=256,
                        help='sequence length held constant when sweeping a non-seq_len variable '
                             '(batch_size/dim/vocab_size); ignored when sweeping seq_len')

    # timing:
    parser.add_argument('--batch-size', type=int, default=32, help='batch size (base value; may be reduced on OOM)')
    parser.add_argument('--precision', type=str, default='bf16', choices=['bf16', 'fp16', '32'],
                        help='compute precision (use 32 for the fork LSTM/BD-LRU/H-LRU layers)')
    parser.add_argument('--warmup', type=int, default=5, help='untimed warmup steps')
    parser.add_argument('--iters', type=int, default=20, help='timed steps per measurement group')
    parser.add_argument('--repeats', type=int, default=5,
                        help='measurement groups; latency reported as median +/- IQR over these')
    parser.add_argument('--train', action=argparse.BooleanOptionalAction, default=True,
                        help='if set, time fwd+bwd+optimizer; otherwise fwd only (inference)')
    parser.add_argument('--oom-retry', action=argparse.BooleanOptionalAction, default=True,
                        help='on CUDA OOM, halve the batch size and retry (down to 1)')
    parser.add_argument('--step-ceiling-ms', type=float, default=None,
                        help='abort a cell early if a step exceeds this (ms) and record it as '
                             'capped_step:>Nms with the ceiling as an off-chart sentinel latency; '
                             'used to stop non-competitive sequential loops (e.g. orig at long T)')

    # data caching:
    parser.add_argument('--num-samples', type=int, default=512,
                        help='number of synthetic sequences to cache per (vocab, seq_len)')
    parser.add_argument('--data-path', type=str, default='./speed_data',
                        help='root dir for the cached synthetic datasets')
    parser.add_argument('--seed', type=int, default=0, help='seed for synthetic data generation')

    # output:
    parser.add_argument('--name', type=str, default=None,
                        help='benchmark name; results go to results/<name>/results.csv '
                             'and figures to figures/<name>/ (default: derived from the model)')
    parser.add_argument('--out', type=str, default=None,
                        help='explicit CSV path (overrides --name); appended if it exists')
    parser.add_argument('--results-root', type=str, default=RESULTS_ROOT,
                        help=f'root dir for per-benchmark result CSVs (default: {RESULTS_ROOT})')
    parser.add_argument('--figures-root', type=str, default=FIGURES_ROOT,
                        help=f'root dir for per-benchmark figures (default: {FIGURES_ROOT})')
    parser.add_argument('--plot', action=argparse.BooleanOptionalAction, default=True,
                        help='auto-generate figures into figures/<name>/ after the sweep')
    parser.add_argument('--run-tag', type=str, default='',
                        help='stable identifier written to every row (used by the suite to '
                             'skip already-completed work when resuming/extending a benchmark)')

    return parser.parse_args()


def _slugify(text: str) -> str:
    """Make a filesystem-friendly benchmark folder name from a model id."""
    keep = []
    for ch in text:
        keep.append(ch if (ch.isalnum() or ch in '-_') else '-')
    slug = ''.join(keep).strip('-')
    while '--' in slug:
        slug = slug.replace('--', '-')
    return slug or 'benchmark'


def resolve_output(args, model_id: str) -> tp.Tuple[str, str]:
    """Return (benchmark_name, csv_path) honoring --out > --name > model default."""
    name = args.name or _slugify(model_id)
    if args.out:
        return name, args.out
    out_dir = os.path.join(args.results_root, name)
    os.makedirs(out_dir, exist_ok=True)
    return name, os.path.join(out_dir, 'results.csv')


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
    seq_len = value if args.sweep_key == 'seq_len' else args.fixed_seq_len
    dim = value if args.sweep_key == 'dim' else args.dim
    vocab_size = value if args.sweep_key == 'vocab_size' else args.vocab_size
    batch_size = value if args.sweep_key == 'batch_size' else args.batch_size

    # hidden_dim / window_dim are layer-config fields, so a sweep over them is
    # applied as a per-run override on top of any explicit --layer-overrides.
    swept_overrides = {}
    if args.sweep_key in ('hidden_dim', 'window_dim'):
        swept_overrides[args.sweep_key] = value

    backbone = args.backbone or ('autoencoder' if args.task == 'compression' else 'language-model')

    # cached synthetic dataset (generated once, reused afterwards)
    inputs, targets, dataset_dir, created = get_or_create_synthetic_dataset(
        data_path=args.data_path, vocab_size=vocab_size, seq_len=seq_len,
        num_samples=max(args.num_samples, batch_size), seed=args.seed,
    )
    print(f"[{args.sweep_key}={value}] dataset {'created' if created else 'loaded'}: {dataset_dir}")

    overrides = {**parse_layer_overrides(args.layer_overrides), **swept_overrides}
    while batch_size >= 1:
        try:
            model = build_model(args.layers, dim, vocab_size, seq_len, backbone, overrides)
            result = time_model(
                model, inputs, targets, batch_size=batch_size,
                precision=args.precision, warmup=args.warmup, iters=args.iters,
                repeats=args.repeats, train=args.train,
                step_ceiling_ms=args.step_ceiling_ms,
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
        except Exception as e:  # noqa: BLE001 - record any build/timing failure as an error row
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
    name, out_path = resolve_output(args, model_id)
    print(f"Benchmarking model '{model_id}' on '{args.task}', sweeping {args.sweep_key} over {args.sweep_values}")
    print(f"  benchmark='{name}'  ->  {out_path}\n")

    rows = []
    for value in args.sweep_values:
        result = run_one(args, value) or {'status': 'unknown'}
        result.update({
            'model': model_id,
            'layers': ' '.join(args.layers),
            'overrides': override_suffix,
            'run_tag': args.run_tag,
            'task': args.task,
            'precision': args.precision,
            'train': args.train,
            args.sweep_key: value,
        })
        rows.append(result)
        _print_row(args.sweep_key, value, result)

    _write_csv(out_path, rows)
    print(f"\nWrote {len(rows)} rows to {out_path}")

    if args.plot:
        _plot_results(out_path, name, args.sweep_key, args.figures_root)


def _plot_results(csv_path: str, name: str, sweep_key: str, figures_root: str) -> None:
    """Auto-emit figures for a finished benchmark.

    Always writes the simple per-metric line plots. When the accumulated CSV holds
    two or more implementations (i.e. several runs share this --name), also emits
    the publication-quality `comparison` (and `block_comparison`) figures.
    """
    try:
        import pandas as pd
        from scripts.plot_speed import render
    except Exception as exc:  # noqa: BLE001 - plotting is best-effort, never fatal
        print(f"(skipping figures: {type(exc).__name__}: {exc})")
        return
    df = pd.read_csv(csv_path)
    for metric in PLOT_METRICS:
        path = render(df, name, metric, x=sweep_key, logx=True, logy=True,
                      figures_root=figures_root)
        if path:
            print(f"  figure: {path}")

    # comparison figures only make sense across >= 2 implementations
    try:
        from scripts.plot_compare import generate as generate_comparison
        for path in generate_comparison([csv_path], name, x=sweep_key,
                                        figures_root=figures_root, min_impls=2):
            print(f"  figure: {path}")
    except Exception as exc:  # noqa: BLE001 - comparison plot is best-effort
        print(f"(skipping comparison figure: {type(exc).__name__}: {exc})")


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
    """Append rows to `path`, keeping column order stable across invocations.

    When the file already exists we reuse ITS header order (dropping any keys not
    present, filling missing ones with '') so appended rows always line up with the
    header -- otherwise a sweep whose rows carry a different key set (e.g. all-OOM,
    or a capped row) would recompute a different column order and misalign the CSV.
    """
    if not rows:
        return
    if os.path.exists(path):
        with open(path, newline='') as f:
            fieldnames = next(csv.reader(f))
        write_header = False
    else:
        fieldnames = sorted({k for r in rows for k in r})
        write_header = True
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore', restval='')
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()
