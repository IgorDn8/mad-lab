#!/usr/bin/env python3
"""When is a *parallel* scan preferential? (the crossover question, Q#6)

Given a benchmark that sweeps TWO axes (e.g. sequence length x block size, or
sequence length x batch size) across all BD-LRU scan implementations, this
splits the implementations into two families

    PARALLEL family   : work-efficient / associative scans whose *depth* is
                        sub-linear -- triton_parallel_blelloch,
                        affine_scan_torch_impl, hopscan_custom
    SEQUENTIAL family : linear-depth per-row kernels --
                        triton_sequential, triton_persistent (and `orig`)

and answers, for every point of the (x, y) grid, which family is faster and by
how much. It renders one figure with two panels into ``figures/<name>/``:

  (a) crossover.png -- heatmap of log2(best-parallel / best-sequential)
      throughput. Red => parallel wins, blue => sequential wins; the white band
      (ratio = 1) is the crossover boundary. Each cell is annotated ``P 1.8x`` /
      ``S 2.3x`` with the winning family and speedup.

  (b) the same grid colored by the single best implementation (winner-take-all),
      so you can see *which* kernel owns each regime.

A one-line verdict (where parallel starts winning) is printed and used as the
subtitle.

Usage:
    uv run python -m scripts.plot_crossover --name blocksize \
        --csv results/blocksize/results.csv --x seq_len --y window_dim
"""

from __future__ import annotations

import os
import re
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from scripts.plot_compare import _impl_of, _block_of, _pretty, _style, FIGURES_ROOT


# Family membership. "parallel" = sub-linear depth; "sequential" = O(T) depth.
PARALLEL = {'triton_parallel_blelloch', 'affine_scan_torch_impl', 'hopscan_custom',
            'custom_hopscan_autotune', 'triton_chunked', 'triton_auto',
            'triton_auto_v2', 'triton_auto_v2_compile', 'triton_fused_gates',
            'affine_scan', 'hopscan_opt'}
SEQUENTIAL = {'triton_sequential', 'triton_persistent', 'orig'}


def _family(impl: str) -> str | None:
    if impl in PARALLEL:
        return 'parallel'
    if impl in SEQUENTIAL:
        return 'sequential'
    return None


def _ensure_axis(df: pd.DataFrame, y: str) -> pd.DataFrame:
    """Guarantee a numeric `y` column, deriving it from overrides if needed.

    window_dim can also come from the model shorthand; any override-driven grid
    axis (e.g. hidden_dim) is parsed out of the `overrides` string ``y=<int>``.
    """
    if y in df.columns and df[y].notna().any():
        return df
    if y == 'window_dim':
        df[y] = df.apply(_block_of, axis=1)
        return df
    pat = re.compile(rf'{re.escape(y)}=(\d+)')

    def parse(ov):
        if isinstance(ov, str):
            m = pat.search(ov)
            if m:
                return int(m.group(1))
        return np.nan
    df[y] = df.get('overrides').map(parse) if 'overrides' in df.columns else np.nan
    return df


def load(csv_paths: list[str], x: str, y: str) -> pd.DataFrame:
    df = pd.concat([pd.read_csv(c) for c in csv_paths], ignore_index=True)
    if 'status' in df.columns:
        df = df[df['status'] == 'ok'].copy()
    df['impl'] = df.apply(_impl_of, axis=1)
    df['family'] = df['impl'].map(_family)
    df = _ensure_axis(df, y)
    if x not in df.columns:
        raise SystemExit(f'x-axis column {x!r} not in CSV.')
    df = df.dropna(subset=['tokens_per_s', x, y])
    if df.empty:
        raise SystemExit('No successful rows to analyze.')
    return df


def _grid(df: pd.DataFrame, x: str, y: str):
    xs = sorted(df[x].dropna().unique())
    ys = sorted(df[y].dropna().unique())
    return xs, ys


def _best(df, x, y, xv, yv, family=None):
    """(best throughput, best impl) at a grid cell, optionally within a family."""
    sub = df[(df[x] == xv) & (df[y] == yv)]
    if family is not None:
        sub = sub[sub['family'] == family]
    sub = sub.dropna(subset=['tokens_per_s'])
    if sub.empty:
        return None, None
    row = sub.loc[sub['tokens_per_s'].idxmax()]
    return float(row['tokens_per_s']), str(row['impl'])


def _panel_ratio(ax, df, xs, ys, x, y):
    ratio = np.full((len(ys), len(xs)), np.nan)
    annot = [['' for _ in xs] for _ in ys]
    for iy, yv in enumerate(ys):
        for ix, xv in enumerate(xs):
            par, _ = _best(df, x, y, xv, yv, 'parallel')
            seq, _ = _best(df, x, y, xv, yv, 'sequential')
            if par and seq:
                r = par / seq
                ratio[iy, ix] = np.log2(r)
                fam, sp = ('P', r) if r >= 1 else ('S', 1 / r)
                annot[iy][ix] = f'{fam} {sp:.1f}x'

    vmax = np.nanmax(np.abs(ratio)) if np.isfinite(ratio).any() else 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    im = ax.imshow(ratio, origin='lower', aspect='auto', cmap='RdBu_r', norm=norm)
    for iy in range(len(ys)):
        for ix in range(len(xs)):
            if annot[iy][ix]:
                ax.text(ix, iy, annot[iy][ix], ha='center', va='center', fontsize=7,
                        color='black')
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels([str(int(v)) for v in xs], rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(ys)))
    ax.set_yticklabels([str(int(v)) for v in ys], fontsize=8)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title('(a) log2(best parallel / best sequential)\nred = parallel wins, blue = sequential wins', fontsize=10)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('log2 speedup')


def _panel_winner(ax, df, xs, ys, x, y):
    impls = sorted(df['impl'].unique())
    code = {impl: i for i, impl in enumerate(impls)}
    grid = np.full((len(ys), len(xs)), np.nan)
    for iy, yv in enumerate(ys):
        for ix, xv in enumerate(xs):
            _, impl = _best(df, x, y, xv, yv)
            if impl is not None:
                grid[iy, ix] = code[impl]

    # discrete colormap from each impl's canonical style color
    from matplotlib.colors import ListedColormap, BoundaryNorm
    colors = [_style(i)['color'] for i in impls]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, len(impls) + 0.5, 1), cmap.N)
    ax.imshow(grid, origin='lower', aspect='auto', cmap=cmap, norm=norm)
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels([str(int(v)) for v in xs], rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(ys)))
    ax.set_yticklabels([str(int(v)) for v in ys], fontsize=8)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title('(b) Fastest implementation per regime', fontsize=10)
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=_style(i)['color'], label=_pretty(i)) for i in impls]
    ax.legend(handles=handles, fontsize=7, loc='center left', bbox_to_anchor=(1.02, 0.5),
              framealpha=0.9)


def verdict(df, xs, ys, x, y) -> str:
    """Summarize where the parallel family starts winning, per y-row."""
    lines = []
    for yv in ys:
        crossover = None
        for xv in xs:
            par, _ = _best(df, x, y, xv, yv, 'parallel')
            seq, _ = _best(df, x, y, xv, yv, 'sequential')
            if par and seq and par > seq:
                crossover = xv
                break
        if crossover is None:
            lines.append(f'{y}={int(yv)}: sequential wins everywhere')
        elif crossover == xs[0]:
            lines.append(f'{y}={int(yv)}: parallel wins everywhere')
        else:
            lines.append(f'{y}={int(yv)}: parallel wins for {x}>={int(crossover)}')
    return ' | '.join(lines)


def build_figure(df, xs, ys, x, y, subtitle) -> plt.Figure:
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 10, 'figure.dpi': 120})
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))
    _panel_ratio(axes[0], df, xs, ys, x, y)
    _panel_winner(axes[1], df, xs, ys, x, y)
    fig.suptitle('When is a parallel scan preferential? \u2014 ' + subtitle,
                 fontsize=13, fontweight='bold', y=1.0)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    return fig


def get_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--csv', nargs='+', required=True)
    p.add_argument('--name', type=str, required=True, help='benchmark sub-folder under figures/')
    p.add_argument('--x', type=str, default='seq_len', help='x-axis grid column (default: seq_len)')
    p.add_argument('--y', type=str, default='window_dim',
                   help='y-axis grid column (e.g. window_dim, batch_size, hidden_dim)')
    p.add_argument('--figures-root', type=str, default=FIGURES_ROOT)
    p.add_argument('--subtitle', type=str, default=None)
    return p.parse_args()


def generate(csv_paths, name, *, x='seq_len', y='window_dim',
             figures_root=FIGURES_ROOT, subtitle=None) -> list[str]:
    df = load(csv_paths, x, y)
    xs, ys = _grid(df, x, y)
    if len(xs) < 1 or len(ys) < 2:
        return []  # need a genuine 2-D grid to talk about crossover
    if not ((df['family'] == 'parallel').any() and (df['family'] == 'sequential').any()):
        return []  # need both families present
    subtitle = subtitle or (' / '.join(sorted(df['layers'].dropna().unique())) or name)
    v = verdict(df, xs, ys, x, y)
    print('Verdict:', v)
    fig = build_figure(df, xs, ys, x, y, f'{subtitle}\n{v}')
    out_dir = os.path.join(figures_root, name)
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for ext in ('png', 'pdf'):
        path = os.path.join(out_dir, f'crossover.{ext}')
        fig.savefig(path, dpi=200, bbox_inches='tight')
        written.append(path)
    plt.close(fig)
    return written


def main():
    args = get_args()
    written = generate(args.csv, args.name, x=args.x, y=args.y,
                       figures_root=args.figures_root, subtitle=args.subtitle)
    if not written:
        print('Nothing written: need a 2-D grid with both parallel and sequential families.')
    for p in written:
        print(f'Wrote {p}')


if __name__ == '__main__':
    main()
