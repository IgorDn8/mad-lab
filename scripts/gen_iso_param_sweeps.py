"""Solve every family's config for each iso-PARAMETER tier and write configs.

For a fixed model width `d` and a target mixer-parameter budget, this solves the
size knob of each family so its parameter count hits the budget, for tiers:
    d=128  -> 0.33M, 1M
    d=1024 -> 1M, 10M, 33M, 100M

Families and the knob that is solved (all monotonic in the knob):
    LSTM           : hidden_dim
    BD-LRU / H-LRU : hidden_dim, at blocks m in {1,2,4,8,16}
    PDSSM          : hidden_dim
    Mamba2         : expand (state_size=128, head_dim=64 fixed)
    DeltaNet       : head_dim (num_heads=8 fixed)
    DeltaProduct   : head_dim (num_heads=8 fixed), at householders r in {2,4,8}
                     -- rank changes params but NOT state; the 2/4/8 variants are
                     the intended #householders-vs-block-size baselines.

Writes one YAML per config under configs/layers/ plus an index at
configs/iso_param_sweeps.json that mad/registry.py reads to register them.

    uv run python -m scripts.gen_iso_param_sweeps           # solve + write
    uv run python -m scripts.gen_iso_param_sweeps --check    # solve + print only
"""

import os
import json
import argparse

import yaml

from mad.model.layers import BDLRU_sel, HLRU_sel, LSTM, PDSSM, Mamba2fla, dnet, dproduct
from mad.paths import get_base_path


# (dim, target_params, tag)
TIERS = [
    (128, 330_000, 'iso033m'),
    (128, 1_000_000, 'iso1m'),
    (1024, 1_000_000, 'iso1m'),
    (1024, 10_000_000, 'iso10m'),
    (1024, 33_000_000, 'iso33m'),
    (1024, 100_000_000, 'iso100m'),
]
BLOCKS = [1, 2, 4, 8, 16]
DPROD_RANKS = [2, 4, 8]
MAX_LENGTH = 2048
INDEX_PATH = 'configs/iso_param_sweeps.json'


def _params(layer) -> int:
    return sum(p.numel() for p in layer.parameters() if p.requires_grad)


def _solve(count_fn, target, lo, step):
    """Grid value lo+step*i (i>=0) whose count is closest to target; count is monotonic up."""
    cache = {}

    def c(v):
        if v not in cache:
            cache[v] = count_fn(v)
        return cache[v]

    def val(i):
        return lo + i * step

    i_hi = 1
    while c(val(i_hi)) < target and i_hi < 5_000_000:
        i_hi *= 2
    i_lo = i_hi // 2
    while i_lo < i_hi:
        mid = (i_lo + i_hi) // 2
        if c(val(mid)) < target:
            i_lo = mid + 1
        else:
            i_hi = mid
    cands = [i for i in (i_lo - 1, i_lo) if i >= 0] or [0]
    best = min(cands, key=lambda i: abs(c(val(i)) - target))
    return val(best), c(val(best))


# ---- per-family builders + config emitters -------------------------------
def _family_configs(dim, target, tag):
    """Yield (name, family, cfg_dict, params) for every family at this tier."""
    out = []

    # LSTM: solve hidden_dim
    n, p = _solve(lambda N: _params(LSTM(dim=dim, hidden_dim=N, max_length=MAX_LENGTH)),
                  target, lo=1, step=1)
    out.append((f'lstm-d{dim}-{tag}', 'lstm',
                {'dim': dim, 'hidden_dim': n}, p))

    # BD-LRU / H-LRU: solve hidden_dim per block
    for fam, mod in (('bdlru', BDLRU_sel), ('hlru', HLRU_sel)):
        for m in BLOCKS:
            n, p = _solve(lambda N, m=m, mod=mod: _params(
                mod(dim=dim, hidden_dim=N, window_dim=m, implementation='orig', max_length=MAX_LENGTH)),
                target, lo=1, step=1)
            out.append((f'{fam}-sel-wd{m}-d{dim}-{tag}', fam,
                        {'dim': dim, 'hidden_dim': n, 'window_dim': m, 'implementation': 'orig'}, p))

    # PDSSM: solve hidden_dim
    def _pdssm(N):
        return _params(PDSSM(dim=dim, hidden_dim=N, dictionary_size=8, hidden_D_multiple=2,
                             dropout_rate=0.01, implementation='associative_scan', max_length=MAX_LENGTH))
    n, p = _solve(_pdssm, target, lo=1, step=1)
    out.append((f'pdssm-d{dim}-{tag}', 'pdssm',
                {'dim': dim, 'hidden_dim': n, 'dictionary_size': 8, 'hidden_D_multiple': 2,
                 'dropout_rate': 0.01, 'implementation': 'associative_scan'}, p))

    # Mamba2: solve expand (state_size=128, head_dim=64 fixed). Skip tiers whose
    # budget is below Mamba2's floor (expand=1) by a wide margin -- e.g. d=1024/1M,
    # where the smallest Mamba2 is already ~3.4M.
    def _mamba2(E):
        return _params(Mamba2fla(dim=dim, head_dim=64, state_size=128, expand=E,
                                 n_groups=1, conv_kernel=4, chunk_size=256, backend='triton'))
    if _mamba2(1) <= target * 1.25:
        e, p = _solve(_mamba2, target, lo=1, step=1)
        out.append((f'mamba2-fla-d{dim}-{tag}', 'mamba2',
                    {'dim': dim, 'head_dim': 64, 'state_size': 128, 'expand': e,
                     'n_groups': 1, 'conv_kernel': 4, 'chunk_size': 256, 'backend': 'triton'}, p))

    # DeltaNet: fixed kernel-safe head_dim=16, solve num_heads (fine granularity)
    n, p = _solve(lambda NH: _params(dnet(dim=dim, head_dim=16, num_heads=NH, expand_v=1,
                                          gated=True, negative=True, max_length=MAX_LENGTH)),
                  target, lo=1, step=1)
    out.append((f'dnet-d{dim}-{tag}', 'dnet',
                {'dim': dim, 'head_dim': 16, 'num_heads': n, 'expand_v': 1,
                 'gated': True, 'negative': True}, p))

    # DeltaProduct: fixed head_dim=16, solve num_heads per householder count r.
    # rank changes params (not state); the 2/4/8 variants are matched to budget.
    for r in DPROD_RANKS:
        n, p = _solve(lambda NH, r=r: _params(dproduct(dim=dim, head_dim=16, num_heads=NH, expand_v=1,
                                                      gated=True, negative=True, rank=r, max_length=MAX_LENGTH)),
                      target, lo=1, step=1)
        out.append((f'dproduct-hh{r}-d{dim}-{tag}', 'dproduct',
                    {'dim': dim, 'head_dim': 16, 'num_heads': n, 'expand_v': 1,
                     'gated': True, 'negative': True, 'rank': r}, p))

    return out


def get_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--check', action='store_true', help='solve and print only, do not write')
    return p.parse_args()


def main():
    args = get_args()
    base = get_base_path()
    index = []

    print(f"{'layer':40s} {'family':10s} {'params(M)':>10s} {'err%':>7s}")
    print('-' * 72)
    for dim, target, tag in TIERS:
        for name, fam, cfg, params in _family_configs(dim, target, tag):
            err = 100.0 * (params - target) / target
            print(f"{name:40s} {fam:10s} {params/1e6:10.3f} {err:7.1f}")
            cfg_path = f'configs/layers/{name}.yml'
            index.append({'name': name, 'family': fam, 'cfg_path': cfg_path})
            if not args.check:
                with open(os.path.join(base, cfg_path), 'w') as f:
                    yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)

    if not args.check:
        with open(os.path.join(base, INDEX_PATH), 'w') as f:
            json.dump(index, f, indent=2)
        print(f"\nwrote {len(index)} configs + index {INDEX_PATH}")


if __name__ == '__main__':
    main()
