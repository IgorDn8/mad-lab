"""Count sequence-mixer parameters at a fixed model width (default d=128).

Iso-parameter normalization (a core MAD principle) compares architectures at a
matched parameter budget. Token-embedding and unembedding params are identical
across models for a fixed (vocab, dim), so we count only the *sequence-mixer
layer* params -- i.e. exactly what differs between architectures.

Usage:
    uv run python -m scripts.param_count                 # all buildable layers
    uv run python -m scripts.param_count --families bdlru dproduct lstm
    uv run python -m scripts.param_count --group --tol 0.10   # iso-param buckets

The script instantiates each layer as `layer(**cfg)` (the same call the
LanguageModel backbone makes) and sums trainable parameters.
"""

import os
import argparse
import typing as tp
from collections import defaultdict

import torch

from mad.registry import layer_registry
from mad.configs import load_yml
from mad.paths import get_base_path


# families of interest for the speed/accuracy benchmark set (prefix -> label)
FAMILIES = {
    'lstm-': 'LSTM',
    'bdlru-sel-': 'BD-LRU',
    'hlru-sel-': 'H-LRU',
    'pdssm-': 'PDSSM',
    'mamba2-fla-': 'Mamba2 (fla)',
    'mamba': 'Mamba',
    'dnet-': 'DeltaNet',
    'dproduct-': 'DeltaProduct',
    'attention': 'Attention',
    'mh-attention': 'Attention (MH)',
    'hyena': 'Hyena',
    'gated-linear-attention': 'GLA',
    'linear-attention': 'LinAttn',
}


def _family_of(name: str) -> tp.Optional[str]:
    # longest-prefix match so 'mh-attention' beats 'attention', etc.
    best = None
    for prefix, label in FAMILIES.items():
        if name.startswith(prefix):
            if best is None or len(prefix) > len(best[0]):
                best = (prefix, label)
    return best[1] if best else None


def count_layer_params(name: str, dim: int, max_length: int,
                       overrides: tp.Optional[dict] = None) -> int:
    """Instantiate the mixer layer and return its trainable-parameter count."""
    entry = layer_registry[name]
    module = entry['module']
    if module is None:
        raise RuntimeError('layer unavailable (missing extra/dependency)')
    cfg = load_yml(os.path.join(get_base_path(), entry['cfg']))
    cfg['dim'] = dim
    cfg['max_length'] = max_length
    for k, v in (overrides or {}).items():
        if k in cfg:
            cfg[k] = v
    layer = module(**cfg)
    return sum(p.numel() for p in layer.parameters() if p.requires_grad)


def get_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dim', type=int, default=128, help='model width (embedding dim)')
    p.add_argument('--max-length', type=int, default=2048,
                   help='max sequence length passed to the layer (rarely affects param count)')
    p.add_argument('--families', nargs='*', default=None,
                   help='restrict to these family labels (case-insensitive substring)')
    p.add_argument('--layers', nargs='*', default=None,
                   help='explicit layer names to count (overrides family discovery)')
    p.add_argument('--group', action='store_true',
                   help='also print iso-parameter buckets')
    p.add_argument('--tol', type=float, default=0.10,
                   help='relative tolerance for iso-parameter grouping (default 10%%)')
    return p.parse_args()


def main():
    args = get_args()
    names = args.layers if args.layers else sorted(layer_registry.keys())

    rows = []
    for name in names:
        fam = _family_of(name)
        if args.layers is None:
            if fam is None:
                continue
            if args.families and not any(f.lower() in fam.lower() for f in args.families):
                continue
        try:
            n = count_layer_params(name, args.dim, args.max_length)
        except Exception as e:  # noqa: BLE001
            rows.append((name, fam or '?', None, f'{type(e).__name__}: {str(e)[:60]}'))
            continue
        rows.append((name, fam or '?', n, ''))

    ok = [r for r in rows if r[2] is not None]
    ok.sort(key=lambda r: r[2])

    print(f"\nSequence-mixer parameters at d={args.dim} (max_length={args.max_length})")
    print(f"{'layer':44s} {'family':16s} {'params':>12s}  {'M':>7s}")
    print('-' * 84)
    for name, fam, n, _ in ok:
        print(f"{name:44s} {fam:16s} {n:12,d}  {n/1e6:7.3f}")

    skipped = [r for r in rows if r[2] is None]
    if skipped:
        print(f"\nskipped ({len(skipped)}):")
        for name, fam, _, err in skipped:
            print(f"  {name:44s} {fam:16s} {err}")

    if args.group:
        # greedy bucketing: sort by params, start a new bucket when the next
        # value exceeds (1+tol) * bucket anchor.
        print(f"\nIso-parameter buckets (relative tol = {args.tol:.0%}):")
        buckets: list[list] = []
        for name, fam, n, _ in ok:
            if buckets and n <= buckets[-1][0] * (1 + args.tol):
                buckets[-1][1].append((name, fam, n))
            else:
                buckets.append([n, [(name, fam, n)]])
        for anchor, members in buckets:
            fams = {m[1] for m in members}
            if len(fams) < 2:
                continue  # only interesting when >=2 families coincide
            lo = min(m[2] for m in members)
            hi = max(m[2] for m in members)
            print(f"\n  ~{(lo+hi)/2/1e6:.3f}M  ({lo/1e6:.3f}-{hi/1e6:.3f}M, {len(fams)} families):")
            for name, fam, n in members:
                print(f"    {name:44s} {fam:16s} {n/1e6:7.3f}M")


if __name__ == '__main__':
    main()
