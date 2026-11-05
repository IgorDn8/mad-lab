"""Generate the synthetic datasets of the paper for one or more seeds.

The dataset settings come from ``configs/tasks/<task>.yml`` (see
``mad/task_configs.py``): the baseline of each task plus its variations. Datasets
are written to the canonical path layout used by ``train.py``
(``mad.paths.make_dataset_path``), so a later ``train.py`` call with the same task
arguments and seed loads them instead of regenerating.

Examples:
    # every task that has a config file (MAD tasks + group-S), three seeds
    uv run python -m scripts.generate_synth_data --seeds 42 43 12345

    # only the three main token-manipulation tasks
    uv run python -m scripts.generate_synth_data --tasks in-context-recall selective-copying compression

    # the permutation task (group-S), one seed
    uv run python -m scripts.generate_synth_data --tasks group-S --seeds 42

    # print the expanded settings without generating anything
    uv run python -m scripts.generate_synth_data --list

    # one custom dataset (overrides the config files)
    uv run python -m scripts.generate_synth_data \\
        --task in-context-recall --vocab-size 16 --seq-len 128 \\
        --num-train-examples 20000 --num-test-examples 1280 --seeds 42
"""

from __future__ import annotations

import argparse
import os

from mad.configs import MADConfig
from mad.data import generate_data
from mad.paths import make_dataset_path
from mad.task_configs import load_task_configs, paper_tasks
from scripts.generate_group_data import TASK_TO_KIND, generate_for_seed as generate_group_for_seed


def generate_one(spec: dict, seed: int, data_path: str, num_workers: int, overwrite: bool) -> str:
    """Generate one dataset (train + test split) for `spec` and `seed`; returns its path."""
    if spec['task'] in TASK_TO_KIND:
        # group tasks: build the group once and draw disjoint train/test sequences
        return generate_group_for_seed(
            task=spec['task'], vocab_size=spec['vocab_size'], seq_len=spec['seq_len'],
            num_train=spec['num_train_examples'], num_test=spec.get('num_test_examples', 1280),
            seed=seed, data_path=data_path, overwrite=overwrite,
        )
    cfg = MADConfig(data_path=data_path, seed=seed, **spec)
    dataset_path = make_dataset_path(cfg)
    if os.path.isdir(cfg.train_dataset_path) and os.path.isdir(cfg.test_dataset_path) and not overwrite:
        print(f'Skip existing: {dataset_path}')
        return dataset_path
    print(f'\n=== seed {seed} === {dataset_path}')
    generate_data(
        instance_fn=cfg.instance_fn,
        instance_fn_kwargs=cfg.instance_fn_kwargs,
        train_data_path=cfg.train_dataset_path,
        test_data_path=cfg.test_dataset_path,
        num_train_examples=cfg.num_train_examples,
        num_test_examples=cfg.num_test_examples,
        num_workers=num_workers,
    )
    print(f'  wrote {dataset_path}')
    return dataset_path


def get_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--tasks', type=str, nargs='+', default=None,
                        help='tasks to generate from configs/tasks/ (default: every task with a config file)')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 12345],
                        help='random seeds (one dataset directory per seed)')
    parser.add_argument('--data-path', type=str, default='./data')
    parser.add_argument('--num-data-workers', type=int, default=0,
                        help='parallel workers for generation (0 = sequential)')
    parser.add_argument('--overwrite', action='store_true',
                        help='regenerate even if train/ and test/ already exist')
    parser.add_argument('--list', action='store_true',
                        help='print the expanded dataset settings and exit')
    # single custom dataset (overrides the config files when --task is given)
    parser.add_argument('--task', type=str, default=None)
    parser.add_argument('--vocab-size', type=int, default=None)
    parser.add_argument('--seq-len', type=int, default=None)
    parser.add_argument('--num-train-examples', type=int, default=None)
    parser.add_argument('--num-test-examples', type=int, default=None)
    return parser.parse_args()


def resolve_specs(args) -> list[dict]:
    if args.task is not None:
        spec = dict(task=args.task)
        for key in ('vocab_size', 'seq_len', 'num_train_examples', 'num_test_examples'):
            value = getattr(args, key)
            if value is not None:
                spec[key] = value
        return [spec]
    tasks = args.tasks or paper_tasks()
    specs = []
    for task in tasks:
        specs.extend(load_task_configs(task))
    return specs


def _describe(spec: dict) -> str:
    keys = ('task', 'vocab_size', 'seq_len', 'num_train_examples', 'num_test_examples',
            'k_motif_size', 'v_motif_size', 'frac_noise', 'noise_vocab_size', 'num_tokens_to_copy')
    return '  '.join(f'{k}={spec[k]}' for k in keys if k in spec)


if __name__ == '__main__':
    args = get_args()
    specs = resolve_specs(args)
    if args.list:
        for spec in specs:
            print(_describe(spec))
        print(f'\n{len(specs)} dataset settings x {len(args.seeds)} seeds')
        raise SystemExit(0)
    for seed in args.seeds:
        for spec in specs:
            generate_one(spec, seed, args.data_path, args.num_data_workers, args.overwrite)
    print('\nDone.')
