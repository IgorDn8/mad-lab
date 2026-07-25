#!/usr/bin/env python3
"""Train H-LRU orig vs hopscan_custom on the compression task and compare.

Both implementations now share the same companion recurrence (hopscan /
triton transpose the orig layout so ``h <- A @ h + b`` matches ``h <- h @ A + b``).

Usage:
  python scripts/compare_hlru_compression.py \\
      --vocab-size 16 --seq-len 128 --layers hlru-sel-wd4-d128-h128 mlp
"""

from __future__ import annotations

import argparse
import os
import sys

import yaml

# repo root on sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

import random  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from mad.configs import MADConfig, MADModelConfig  # noqa: E402
from mad.paths import get_base_path, make_log_path  # noqa: E402
from mad.registry import layer_registry, model_registry  # noqa: E402
from train import train  # noqa: E402


def load_yml(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(layers: list[str], dim: int, vocab_size: int, max_length: int,
                backbone: str, implementation: str):
    layer_configs = []
    for layer in layers:
        cfg = load_yml(os.path.join(get_base_path(), layer_registry[layer]["cfg"]))
        cfg["dim"] = dim
        cfg["max_length"] = max_length
        if "implementation" in cfg:
            cfg["implementation"] = implementation
        layer_configs.append(cfg)
    return model_registry[backbone](
        dim=dim,
        vocab_size=vocab_size,
        layers=[layer_registry[l]["module"] for l in layers],
        layer_cfgs=layer_configs,
        max_length=max_length,
    )


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--layers", nargs="+", default=["hlru-sel-wd4-d128-h128", "mlp"])
    p.add_argument("--vocab-size", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--precision", type=str, default="bf16")
    p.add_argument("--early-stop", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--stop-patience", type=int, default=20)
    p.add_argument("--log-base-path", type=str, default="./logs/hlru_comp_compare")
    p.add_argument("--impls", nargs="+", default=["orig", "hopscan_custom"])
    p.add_argument("--no-checkpoints", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    backbone = "autoencoder"
    results = {}

    for impl in args.impls:
        print(f"\n===== Training H-LRU implementation={impl} =====", flush=True)
        set_seed(args.seed)

        mad_kwargs = dict(
            task="compression",
            vocab_size=args.vocab_size,
            seq_len=args.seq_len,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            precision=args.precision,
            early_stop=args.early_stop,
            stop_patience=args.stop_patience,
            accelerator="cuda",
            devices="0,",
            save_checkpoints=not args.no_checkpoints,
        )
        mad_config = MADConfig()
        mad_config.update_from_kwargs(mad_kwargs)

        model_config = MADModelConfig(
            layers=args.layers,
            backbone=backbone,
            dim=args.dim,
            vocab_size=args.vocab_size,
            max_length=args.seq_len,
        )
        model = build_model(
            args.layers, args.dim, args.vocab_size, args.seq_len, backbone, impl
        )
        model_id = "-".join(layer_registry[l]["shorthand"] for l in args.layers) + f"-{impl}"

        log_path = make_log_path(
            base_path=args.log_base_path,
            mad_config=mad_config,
            model_id=model_id,
        )
        print(f"log_path={log_path}", flush=True)

        df = train(
            model=model,
            mad_config=mad_config,
            log_path=log_path,
            log_to_csv=True,
            log_to_wandb=False,
            save_checkpoints=not args.no_checkpoints,
        )
        results[impl] = df.iloc[0].to_dict()
        print(f"[{impl}] test_acc={results[impl]['test_acc']:.4f} "
              f"test_ppl={results[impl]['test_ppl']:.4f} "
              f"test_loss={results[impl]['test_loss']:.4f}", flush=True)

    print("\n===== Summary =====")
    for impl, r in results.items():
        print(
            f"{impl:16s}  train_acc={r['train_acc']:.4f}  test_acc={r['test_acc']:.4f}  "
            f"test_ppl={r['test_ppl']:.4f}  test_loss={r['test_loss']:.4f}"
        )
    if set(args.impls) >= {"orig", "hopscan_custom"}:
        o, h = results["orig"], results["hopscan_custom"]
        better = "hopscan_custom" if h["test_acc"] > o["test_acc"] else "orig"
        if abs(h["test_acc"] - o["test_acc"]) < 1e-4:
            better = "tie"
        print(f"\nBetter by test accuracy: {better} "
              f"(Δacc={h['test_acc'] - o['test_acc']:+.4f})")


if __name__ == "__main__":
    main()
