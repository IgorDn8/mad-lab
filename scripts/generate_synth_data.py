# Generate MAD synthetic task datasets for one or more seeds, writing them to the
# canonical path layout used by train.py (via mad.paths.make_dataset_path).
#
# Example — all specs below with three seeds:
#   python -m scripts.generate_synth_data --seeds 42 43 12345
#
# Example — one task, custom sizes:
#   python -m scripts.generate_synth_data \
#       --task in-context-recall --vocab-size 16 --seq-len 128 \
#       --num-train-examples 20000 --num-test-examples 1280 \
#       --seeds 42 43 12345

from __future__ import annotations

import argparse
import os

from mad.configs import MADConfig
from mad.data import generate_data
from mad.paths import make_dataset_path

# Default kwargs shared by the user's requested dataset names (ntc-16, mq-1, etc.).
COMMON = dict(
    num_test_examples=1280,
    k_motif_size=1,
    v_motif_size=1,
    multi_query=True,
    frac_noise=0.0,
    noise_vocab_size=0,
    num_tokens_to_copy=16,
)

# Each entry becomes one dataset directory per seed (path ends with _s-<seed>).
REQUESTED_SPECS: list[dict] = [
    dict(task="selective-copying", vocab_size=16, seq_len=64, num_train_examples=20000),
    dict(task="selective-copying", vocab_size=16, seq_len=128, num_train_examples=20000),
    dict(task="selective-copying", vocab_size=32, seq_len=64, num_train_examples=20000),
    dict(task="selective-copying", vocab_size=16, seq_len=64, num_train_examples=10000),
    dict(task="compression", vocab_size=16, seq_len=64, num_train_examples=20000),
    dict(task="compression", vocab_size=16, seq_len=128, num_train_examples=20000),
    dict(task="compression", vocab_size=32, seq_len=64, num_train_examples=20000),
    dict(task="compression", vocab_size=16, seq_len=64, num_train_examples=10000),
    dict(task="in-context-recall", vocab_size=16, seq_len=64, num_train_examples=20000),
    dict(task="in-context-recall", vocab_size=16, seq_len=128, num_train_examples=20000),
    dict(task="in-context-recall", vocab_size=32, seq_len=64, num_train_examples=20000),
    dict(task="in-context-recall", vocab_size=16, seq_len=64, num_train_examples=10000),
    dict(task="fuzzy-in-context-recall", vocab_size=32, seq_len=64, num_train_examples=20000),
    dict(task="memorization", vocab_size=4096, seq_len=64, num_train_examples=20000),
]


def _build_config(spec: dict, seed: int, data_path: str) -> MADConfig:
    cfg = MADConfig(
        data_path=data_path,
        seed=seed,
        **COMMON,
        **spec,
    )
    return cfg


def generate_one(spec: dict, seed: int, data_path: str, num_workers: int, overwrite: bool) -> str:
    cfg = _build_config(spec, seed, data_path)
    dataset_path = make_dataset_path(cfg)
    train_path = cfg.train_dataset_path
    test_path = cfg.test_dataset_path

    if (
        os.path.isdir(train_path)
        and os.path.isdir(test_path)
        and not overwrite
    ):
        print(f"Skip existing: {dataset_path}")
        return dataset_path

    print(f"\n=== seed {seed} === {dataset_path}")
    generate_data(
        instance_fn=cfg.instance_fn,
        instance_fn_kwargs=cfg.instance_fn_kwargs,
        train_data_path=train_path,
        test_data_path=test_path,
        num_train_examples=cfg.num_train_examples,
        num_test_examples=cfg.num_test_examples,
        num_workers=num_workers,
    )
    print(f"  wrote {dataset_path}")
    return dataset_path


def get_args():
    parser = argparse.ArgumentParser(
        description="Generate MAD synthetic datasets for multiple seeds."
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[42, 43, 12345],
        help="random seeds (one dataset directory per seed)",
    )
    parser.add_argument("--data-path", type=str, default="./data")
    parser.add_argument(
        "--num-data-workers", type=int, default=0,
        help="parallel workers for generation (0 = sequential)",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="regenerate even if train/ and test/ already exist",
    )
    parser.add_argument(
        "--preset", choices=["requested", "none"], default="requested",
        help="use the built-in spec list from the benchmark sweep (default: requested)",
    )
    # Optional single-spec mode (overrides --preset when any is set).
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--num-train-examples", type=int, default=None)
    parser.add_argument("--num-test-examples", type=int, default=None)
    return parser.parse_args()


def _resolve_specs(args) -> list[dict]:
    if args.task is not None:
        spec = dict(task=args.task)
        if args.vocab_size is not None:
            spec["vocab_size"] = args.vocab_size
        if args.seq_len is not None:
            spec["seq_len"] = args.seq_len
        if args.num_train_examples is not None:
            spec["num_train_examples"] = args.num_train_examples
        if args.num_test_examples is not None:
            spec["num_test_examples"] = args.num_test_examples
        return [spec]
    if args.preset == "requested":
        return REQUESTED_SPECS
    raise SystemExit("No specs: use --preset requested or pass --task ...")


if __name__ == "__main__":
    args = get_args()
    specs = _resolve_specs(args)
    for seed in args.seeds:
        for spec in specs:
            generate_one(
                spec,
                seed=seed,
                data_path=args.data_path,
                num_workers=args.num_data_workers,
                overwrite=args.overwrite,
            )
    print("\nDone.")
