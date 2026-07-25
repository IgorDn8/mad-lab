#!/usr/bin/env python3
"""Full-grid throughput bar charts for MAD speed benchmarks.

Companion to ``plot_compare.build_block_figure`` (which draws throughput-by-block
bars at a *single* sequence length). This renders the same style of grouped
throughput histogram but faceted over a full 2-D grid: one sub-panel per
(row-var, col-var) cell -- by default ``seq_len`` (rows) x ``batch_size``
(columns) -- with one bar per scan implementation inside each cell.

Reads one or more result CSVs (e.g. ``results/batch/results.csv``) and writes
``figures/<name>/throughput_grid.png`` / ``.pdf``.

Because OOM auto-reduces the batch size on the heaviest cells, the *nominal*
batch (the one requested by the suite) is recovered from ``run_tag`` so every
implementation lands in the column it was asked to run at; throughput
(tokens/s) is already batch-normalised, so bars stay comparable.

Example:
    uv run python -m scripts.plot_grid \
        --csv results/batch/results.csv --name batch
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

from scripts.plot_compare import (
    FIGURES_ROOT, IMPL_ORDER, _impl_of, _pretty, _style, _save,
)

_NOMINAL_RE = {
    'batch_size': re.compile(r'batch_size=(\d+)'),
    'seq_len': re.compile(r'seq_len=(\d+)'),
    'hidden_dim': re.compile(r'hidden_dim=(\d+)'),
    'window_dim': re.compile(r'window_dim=(\d+)'),
}

_AXIS_LABEL = {
    'seq_len': 'T', 'batch_size': 'B', 'hidden_dim': 'H', 'window_dim': 'm',
}


def _nominal(row, var: str):
    """Grid coordinate for ``var``: prefer the value baked into ``run_tag``
    (so OOM-reduced runs stay in their requested column), else the raw column.
    """
    tag = row.get('run_tag')
    pat = _NOMINAL_RE.get(var)
    if isinstance(tag, str) and pat is not None:
        m = pat.search(tag)
        if m:
            return int(m.group(1))
    val = row.get(var)
    try:
        return int(val) if pd.notna(val) else None
    except (TypeError, ValueError):
        return None


def load(csv_paths: list[str], row_var: str, col_var: str) -> pd.DataFrame:
    df = pd.concat([pd.read_csv(c) for c in csv_paths], ignore_index=True)
    if 'status' in df.columns:
        df = df[df['status'] == 'ok'].copy()
    if df.empty:
        raise SystemExit('No successful rows to plot (check the CSV / status column).')
    df['impl'] = df.apply(_impl_of, axis=1)
    df['row_val'] = df.apply(lambda r: _nominal(r, row_var), axis=1)
    df['col_val'] = df.apply(lambda r: _nominal(r, col_var), axis=1)
    df['tps_m'] = df['tokens_per_s'] / 1e6
    return df.dropna(subset=['row_val', 'col_val'])


def _impls_present(df: pd.DataFrame) -> list[str]:
    present = list(df['impl'].unique())
    ordered = [i for i in IMPL_ORDER if i in present]
    ordered += [i for i in present if i not in ordered]
    return ordered


def build_grid_figure(df: pd.DataFrame, *, row_var: str, col_var: str,
                      subtitle: str, share_y: bool = True) -> plt.Figure:
    plt.rcParams.update({
        'font.size': 9, 'axes.titlesize': 9,
        'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 120,
    })
    impls = _impls_present(df)
    rows = sorted(df['row_val'].unique())
    cols = sorted(df['col_val'].unique())
    rlab, clab = _AXIS_LABEL.get(row_var, row_var), _AXIS_LABEL.get(col_var, col_var)

    nr, nc = len(rows), len(cols)
    fig, axes = plt.subplots(nr, nc, figsize=(2.7 * nc + 1.5, 2.35 * nr + 1.2),
                             squeeze=False, sharey=share_y)

    ymax = df['tps_m'].max()
    ymin = max(df['tps_m'].min(), ymax / 1e5)  # keep the log floor sane
    xb = np.arange(len(impls))

    for i, rv in enumerate(rows):
        for j, cv in enumerate(cols):
            ax = axes[i][j]
            cell = df[(df['row_val'] == rv) & (df['col_val'] == cv)]
            heights, colors, present_any = [], [], False
            for impl in impls:
                sub = cell[cell['impl'] == impl]
                h = sub['tps_m'].median() if not sub.empty else np.nan
                heights.append(h)
                colors.append(_style(impl)['color'])
                present_any = present_any or not sub.empty
            ax.bar(xb, [0 if np.isnan(h) else h for h in heights], width=0.82,
                   color=colors, edgecolor='white', linewidth=0.4)
            ax.set_yscale('log')
            ax.set_xticks([])
            ax.grid(True, axis='y', which='major', alpha=0.25)
            ax.margins(x=0.02)
            if share_y:
                ax.set_ylim(ymin * 0.6, ymax * 1.6)
            if i == 0:
                ax.set_title(f'{clab}={cv}', fontsize=10, fontweight='bold')
            if j == 0:
                ax.set_ylabel(f'{rlab}={rv}\nM tok/s', fontsize=9)
            if not present_any:
                ax.text(0.5, 0.5, 'n/a', transform=ax.transAxes,
                        ha='center', va='center', color='#aaaaaa', fontsize=9)

    handles = [plt.Rectangle((0, 0), 1, 1, color=_style(im)['color']) for im in impls]
    fig.legend(handles, [_pretty(im) for im in impls], ncol=min(len(impls), 6),
               loc='lower center', fontsize=9, framealpha=0.9,
               bbox_to_anchor=(0.5, 0.0))

    fig.suptitle(f'Throughput grid ({rlab} \u00d7 {clab}) \u2014 {subtitle}',
                 fontsize=13, fontweight='bold', y=0.995)
    fig.text(0.5, 0.955,
             f'bars = scan implementations   |   rows: {row_var}   columns: {col_var}'
             + ('   |   shared log y-axis' if share_y else ''),
             ha='center', fontsize=9, color='#555555')
    fig.tight_layout(rect=(0, 0.05, 1, 0.945))
    return fig


def generate(csv_paths: list[str], name: str, *, row_var: str = 'seq_len',
             col_var: str = 'batch_size', figures_root: str = FIGURES_ROOT,
             subtitle: str = None, share_y: bool = True) -> list[str]:
    df = load(csv_paths, row_var, col_var)
    subtitle = subtitle or ' / '.join(sorted(df['layers'].dropna().unique())) or name
    fig = build_grid_figure(df, row_var=row_var, col_var=col_var,
                            subtitle=subtitle, share_y=share_y)
    return _save(fig, os.path.join(figures_root, name), 'throughput_grid')


def get_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--csv', nargs='+', required=True, help='one or more result CSV files')
    p.add_argument('--name', type=str, required=True, help='benchmark name = sub-folder under figures/')
    p.add_argument('--row-var', type=str, default='seq_len',
                   choices=list(_AXIS_LABEL), help='variable mapped to grid rows')
    p.add_argument('--col-var', type=str, default='batch_size',
                   choices=list(_AXIS_LABEL), help='variable mapped to grid columns')
    p.add_argument('--figures-root', type=str, default=FIGURES_ROOT)
    p.add_argument('--subtitle', type=str, default=None)
    p.add_argument('--independent-y', action='store_true',
                   help='give each cell its own y-scale (default: shared)')
    return p.parse_args()


def main():
    args = get_args()
    for path in generate(args.csv, args.name, row_var=args.row_var,
                         col_var=args.col_var, figures_root=args.figures_root,
                         subtitle=args.subtitle, share_y=not args.independent_y):
        print(f'Wrote {path}')


if __name__ == '__main__':
    main()
