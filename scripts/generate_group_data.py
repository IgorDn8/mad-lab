# Fast, multi-seed generator for the group (word-problem) tasks.
#
# Builds the chosen symmetry group ONCE and reuses it (plus a precomputed Cayley
# table) to generate train/test datasets for one or more seeds, writing them to
# disk in the exact layout MAD's training pipeline expects. Because train.py
# loads a dataset if it already exists at the derived path, datasets produced
# here are picked up automatically on the next training run with matching args.
#
# Example:
#   python -m scripts.generate_group_data \
#       --task group-S --vocab-size 5 --seq-len 16 \
#       --num-train-examples 100000 --num-test-examples 1280 \
#       --seeds 12345 42 43
#
# This mirrors word-problem/src/generate_data.py + convert.py, but without the
# intermediate CSV step and with the group built a single time per seed.

import os
import argparse

import numpy as np

from mad.configs import MADConfig
from mad.paths import make_dataset_path
from mad.data.instances import get_cached_group, prefix_products

# map task name -> group kind expected by get_cached_group
TASK_TO_KIND = {
    'group-S': 'S',
    'group-Z': 'Z',
    'group-A': 'A',
}


def sample_unique_sequences(
    num_elements: int,
    seq_len: int,
    num_sequences: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample `num_sequences` distinct length-`seq_len` sequences of element indices.

    Sequences are drawn without replacement so that any subsequent train/test
    split is guaranteed disjoint (avoiding data leakage). If more sequences are
    requested than exist, all unique sequences are returned.
    """
    num_unique = num_elements ** seq_len
    if num_sequences >= num_unique:
        print(
            f'Warning: requested {num_sequences} sequences but only '
            f'{num_unique} unique ones exist; using all {num_unique}.'
        )
        num_sequences = num_unique

    seqs = set()
    while len(seqs) < num_sequences:
        batch = rng.integers(
            0, num_elements, size=(num_sequences - len(seqs), seq_len)
        )
        for row in batch:
            seqs.add(tuple(row.tolist()))
    return np.array(list(seqs), dtype=np.int64)


def save_split(path: str, inputs: np.ndarray, targets: np.ndarray) -> None:
    """Write inputs.npy / targets.npy into `path` (MAD MemoryDataset layout)."""
    os.makedirs(path, exist_ok=True)
    np.save(os.path.join(path, 'inputs.npy'), inputs)
    np.save(os.path.join(path, 'targets.npy'), targets)


def generate_for_seed(
    task: str,
    vocab_size: int,
    seq_len: int,
    num_train: int,
    num_test: int,
    seed: int,
    data_path: str,
    overwrite: bool = False,
) -> str:
    """Generate and write train/test data for a single seed. Returns the dataset dir."""
    kind = TASK_TO_KIND[task]

    # A MADConfig lets us reuse MAD's canonical dataset path naming, so training
    # with the same arguments will find and load this data automatically.
    mad_config = MADConfig(
        task=task,
        vocab_size=vocab_size,
        seq_len=seq_len,
        num_train_examples=num_train,
        num_test_examples=num_test,
        seed=seed,
        data_path=data_path,
    )
    dataset_path = make_dataset_path(mad_config)
    train_path = os.path.join(dataset_path, 'train')
    test_path = os.path.join(dataset_path, 'test')

    if os.path.exists(train_path) and os.path.exists(test_path) and not overwrite:
        print(f'Skip existing dataset (use --overwrite to regenerate): {dataset_path}')
        return dataset_path

    group_data = get_cached_group(kind, vocab_size)
    num_elements = group_data['num_elements']
    print(
        f'\n=== seed {seed} === {task} |G|={num_elements}, seq_len={seq_len}, '
        f'train={num_train}, test={num_test}'
    )

    rng = np.random.default_rng(seed)
    all_inputs = sample_unique_sequences(
        num_elements, seq_len, num_train + num_test, rng
    )
    rng.shuffle(all_inputs)

    train_inputs = all_inputs[:num_train]
    test_inputs = all_inputs[num_train:num_train + num_test]

    train_targets = prefix_products(train_inputs, group_data)
    test_targets = prefix_products(test_inputs, group_data)

    save_split(train_path, train_inputs, train_targets)
    save_split(test_path, test_inputs, test_targets)

    print(f'  train: {train_inputs.shape}  test: {test_inputs.shape}')
    print(f'  wrote {dataset_path}')
    return dataset_path


def get_args():
    parser = argparse.ArgumentParser(
        description='Fast multi-seed generator for MAD group tasks.'
    )
    parser.add_argument(
        '--task', type=str, default='group-S', choices=list(TASK_TO_KIND.keys()),
        help='which group task to generate data for'
    )
    parser.add_argument(
        '--vocab-size', type=int, default=5,
        help='degree n of the group (e.g. 5 for S5/A5, or |Z_n| for group-Z)'
    )
    parser.add_argument('--seq-len', type=int, default=16, help='sequence length')
    parser.add_argument('--num-train-examples', type=int, default=100_000)
    parser.add_argument('--num-test-examples', type=int, default=1_280)
    parser.add_argument(
        '--seeds', type=int, nargs='+', default=[12345, 42, 43],
        help='one or more random seeds; a dataset is written for each'
    )
    parser.add_argument(
        '--data-path', type=str, default='./data',
        help='root directory for generated datasets'
    )
    parser.add_argument(
        '--overwrite', action='store_true',
        help='regenerate datasets even if they already exist'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    for seed in args.seeds:
        generate_for_seed(
            task=args.task,
            vocab_size=args.vocab_size,
            seq_len=args.seq_len,
            num_train=args.num_train_examples,
            num_test=args.num_test_examples,
            seed=seed,
            data_path=args.data_path,
            overwrite=args.overwrite,
        )
    print('\nDone.')
