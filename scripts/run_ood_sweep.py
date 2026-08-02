#!/usr/bin/env python3
"""Run ``eval_ood`` over every finished job in a MAD sweep directory.

Maps ``runs/<name>/`` folders to the (layer, swiglu) x2 stack used by the
mem/FR iso1m launches, primes OOD test caches once per (seed, eval_seq_len),
then evaluates each checkpoint.

Example:
  uv run python -m scripts.run_ood_sweep --sweep-dir logs_mem_iso1m
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# run-folder name -> recurrent layer registry key (paired with swiglu x2)
NAME_TO_LAYER = {
    'lstm': 'lstm-d128-iso1m',
    'bdlru-wd1': 'bdlru-sel-wd1-d128-iso1m',
    'bdlru-wd2': 'bdlru-sel-wd2-d128-iso1m',
    'bdlru-wd4': 'bdlru-sel-wd4-d128-iso1m',
    'hlru-wd1': 'hlru-sel-wd1-d128-iso1m',
    'hlru-wd2': 'hlru-sel-wd2-d128-iso1m',
    'hlru-wd4': 'hlru-sel-wd4-d128-iso1m',
    'deltanet': 'dnet-d128-iso1m',
    'deltaprod2': 'dproduct-hh2-d128-iso1m',
    'deltaprod4': 'dproduct-hh4-d128-iso1m',
}


def layers_for(name: str) -> list[str]:
    if name not in NAME_TO_LAYER:
        raise KeyError(
            f'Unknown run name {name!r}. Add it to NAME_TO_LAYER in '
            f'{os.path.basename(__file__)}.'
        )
    layer = NAME_TO_LAYER[name]
    return [layer, 'swiglu', layer, 'swiglu']


def discover_runs(sweep_dir: str) -> list[tuple[str, str]]:
    """Return (name, log_path) for every finished run with a checkpoint."""
    runs = []
    for res in sorted(glob.glob(os.path.join(sweep_dir, 'runs', '*', '*', 'results.csv'))):
        log_path = os.path.dirname(res)
        name = log_path.split(os.sep)[-2]
        ckpt_dir = os.path.join(log_path, 'checkpoints')
        if not (os.path.isfile(os.path.join(ckpt_dir, 'best.ckpt'))
                or os.path.isfile(os.path.join(ckpt_dir, 'last.ckpt'))):
            print(f'SKIP {log_path}: no checkpoint')
            continue
        runs.append((name, log_path))
    return runs


def seed_from_path(log_path: str) -> int | None:
    m = re.search(r'_s-(\d+)_', os.path.basename(log_path))
    return int(m.group(1)) if m else None


def run_cmd(cmd: list[str], env: dict | None = None) -> None:
    print('+ ' + ' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT, env=env)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--sweep-dir', required=True)
    p.add_argument('--eval-seq-len', type=int, nargs='+', default=[128, 512, 1024])
    p.add_argument('--devices', type=str, default='0,')
    p.add_argument('--batch-size', type=int, default=None,
                   help='override eval batch size (default: from each checkpoint)')
    p.add_argument('--layer-overrides', nargs='*', default=[],
                   help='e.g. implementation=triton_auto (default: use YAML / training impl)')
    p.add_argument('--force', action='store_true')
    p.add_argument('--prime-only', action='store_true')
    p.add_argument('--names', nargs='*', default=None,
                   help='optional subset of run folder names')
    args = p.parse_args()

    sweep_dir = os.path.abspath(os.path.join(ROOT, args.sweep_dir))
    runs = discover_runs(sweep_dir)
    if args.names:
        want = set(args.names)
        runs = [(n, p) for n, p in runs if n in want]
    if not runs:
        raise SystemExit(f'No finished runs found under {sweep_dir}/runs/')

    print(f'Found {len(runs)} runs under {sweep_dir}')
    print(f'Eval lengths: {args.eval_seq_len}')

    # Prime once per unique (seed, ...) by walking runs; eval_ood --prime-only is
    # idempotent and skips existing caches.
    primed_seeds: set[int] = set()
    for name, log_path in runs:
        seed = seed_from_path(log_path)
        if seed is not None and seed in primed_seeds:
            continue
        layers = layers_for(name)
        cmd = [
            sys.executable, '-m', 'eval_ood',
            '--log-path', log_path,
            '--layers', *layers,
            '--eval-seq-len', *[str(x) for x in args.eval_seq_len],
            '--devices', args.devices,
            '--prime-only',
        ]
        if args.layer_overrides:
            cmd += ['--layer-overrides', *args.layer_overrides]
        if args.batch_size is not None:
            cmd += ['--batch-size', str(args.batch_size)]
        run_cmd(cmd)
        if seed is not None:
            primed_seeds.add(seed)

    if args.prime_only:
        print('Prime-only done.')
        return

    failed = []
    for i, (name, log_path) in enumerate(runs, 1):
        layers = layers_for(name)
        print(f'\n[{i}/{len(runs)}] {name}  {os.path.basename(log_path)}')
        cmd = [
            sys.executable, '-m', 'eval_ood',
            '--log-path', log_path,
            '--layers', *layers,
            '--eval-seq-len', *[str(x) for x in args.eval_seq_len],
            '--devices', args.devices,
        ]
        if args.layer_overrides:
            cmd += ['--layer-overrides', *args.layer_overrides]
        if args.batch_size is not None:
            cmd += ['--batch-size', str(args.batch_size)]
        if args.force:
            cmd.append('--force')
        try:
            run_cmd(cmd)
        except subprocess.CalledProcessError as exc:
            print(f'FAIL {name}: {exc}', flush=True)
            failed.append(log_path)

    print(f'\nDone. failures={len(failed)}/{len(runs)}')
    if failed:
        for path in failed:
            print(f'  - {path}')
        raise SystemExit(1)


if __name__ == '__main__':
    main()
