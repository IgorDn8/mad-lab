#!/usr/bin/env python3
"""Visualize length-OOD evaluation results for a MAD sweep.

Reads ``ood_results.csv`` written by ``eval_ood.py`` under
``<sweep>/runs/<name>/<run>/`` and plots mean±std test accuracy vs evaluation
sequence length. Optionally overlays the in-distribution test accuracy from
each run's ``results.csv`` at the training length.

Example:
  uv run python -m scripts.plot_ood --sweep-dir logs_mem_iso1m --name mem-iso1m-ood
"""

from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter

FIGURES_ROOT = 'figures'

# Keep colours aligned with plot_task_parity.py
FAMILY_COLORS = [
    ('lstm', '#6c757d'),
    ('bdlru', '#0f75bc'),
    ('hlru', '#d1495b'),
    ('deltanet', '#e9a13b'),
    ('deltaprod2', '#c47a1a'),
    ('deltaprod4', '#8a4b12'),
    ('deltaprod', '#b5651d'),
]


def _match_family(name: str):
    hits = [(fam, col, i) for i, (fam, col) in enumerate(FAMILY_COLORS)
            if name == fam or name.startswith(fam)]
    if not hits:
        return None
    return max(hits, key=lambda t: len(t[0]))


def _family_key(name: str) -> tuple:
    hit = _match_family(name)
    if hit is None:
        return (len(FAMILY_COLORS), 0)
    fam, _, i = hit
    if 'wd' in name:
        digits = ''.join(c for c in name.split('wd')[-1] if c.isdigit())
    else:
        digits = ''.join(c for c in name[len(fam):] if c.isdigit())
    return (i, int(digits) if digits else 0)


def _color(name: str) -> str:
    hit = _match_family(name)
    return hit[1] if hit else '#495057'


def _marker(name: str) -> str:
    if name.startswith('bdlru'):
        return 'o'
    if name.startswith('hlru'):
        return 's'
    if name == 'lstm':
        return 'D'
    if name.startswith('delta'):
        return '^'
    return 'o'


def load_ood_runs(sweep_dir: str, include_id: bool = True) -> pd.DataFrame:
    rows = []
    pattern = os.path.join(sweep_dir, 'runs', '*', '*', 'ood_results.csv')
    for path in sorted(glob.glob(pattern)):
        name = path.split(os.sep)[-3]
        run_dir = os.path.dirname(path)
        seed_m = re.search(r'_s-(\d+)_', os.path.basename(run_dir))
        seed = int(seed_m.group(1)) if seed_m else -1
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            rows.append(dict(
                model=name,
                seed=seed,
                train_seq_len=int(r['train_seq_len']),
                eval_seq_len=int(r['eval_seq_len']),
                test_acc=float(r['test_acc']),
                test_ppl=float(r['test_ppl']),
                test_loss=float(r['test_loss']),
                source='ood',
            ))

        if include_id:
            res_path = os.path.join(run_dir, 'results.csv')
            if os.path.isfile(res_path):
                final = pd.read_csv(res_path).iloc[-1]
                # recover train length from any ood row, else from dirname
                if len(df):
                    train_sl = int(df.iloc[0]['train_seq_len'])
                else:
                    m = re.search(r'_sl-(\d+)_', os.path.basename(run_dir))
                    train_sl = int(m.group(1)) if m else -1
                # avoid duplicating if ood already evaluated at train length
                already = any(
                    (row['model'] == name and row['seed'] == seed
                     and row['eval_seq_len'] == train_sl)
                    for row in rows
                )
                if not already and train_sl > 0:
                    rows.append(dict(
                        model=name,
                        seed=seed,
                        train_seq_len=train_sl,
                        eval_seq_len=train_sl,
                        test_acc=float(final['test_acc']),
                        test_ppl=float(final['test_ppl']),
                        test_loss=float(final['test_loss']),
                        source='id',
                    ))

    if not rows:
        raise SystemExit(f'no ood_results.csv found under {sweep_dir}/runs/*/')
    out = pd.DataFrame(rows)
    return out.sort_values(['model', 'seed', 'eval_seq_len']).reset_index(drop=True)


def aggregate(runs: pd.DataFrame) -> pd.DataFrame:
    g = runs.groupby(['model', 'eval_seq_len', 'train_seq_len'], sort=False)
    df = g.agg(
        test_acc=('test_acc', 'mean'),
        test_std=('test_acc', 'std'),
        test_ppl=('test_ppl', 'mean'),
        n=('seed', 'nunique'),
    ).reset_index()
    df['test_std'] = df['test_std'].fillna(0.0)
    df['_ord'] = df['model'].map(_family_key)
    df = df.sort_values(['_ord', 'eval_seq_len']).drop(columns='_ord')
    return df.reset_index(drop=True)


def build_figure(df: pd.DataFrame, *, subtitle: str, chance: float | None,
                 chance_label: str = 'chance') -> plt.Figure:
    plt.rcParams.update({
        'font.size': 10, 'axes.titlesize': 11,
        'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 120,
    })
    fig, ax = plt.subplots(figsize=(7.2, 4.4))

    models = sorted(df['model'].unique(), key=_family_key)
    for name in models:
        sub = df[df['model'] == name].sort_values('eval_seq_len')
        x = sub['eval_seq_len'].to_numpy()
        y = sub['test_acc'].to_numpy()
        e = sub['test_std'].to_numpy()
        color = _color(name)
        ax.plot(x, y, color=color, marker=_marker(name), ms=6.5,
                lw=1.8, label=name, markerfacecolor=color,
                markeredgecolor='white', markeredgewidth=0.9)
        if np.any(e > 0):
            ax.fill_between(x, y - e, y + e, color=color, alpha=0.18, linewidth=0)

    if chance is not None and chance > 0:
        ax.axhline(chance, ls='--', lw=1.1, color='#495057',
                   label=f'{chance_label}={chance:.3g}')

    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_locator(LogLocator(base=2))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{int(v)}'))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel('evaluation sequence length')
    ax.set_ylabel('test accuracy')
    ax.set_title(subtitle, loc='left')

    ymin = max(0.0, float(df['test_acc'].min() - df['test_std'].max()) - 0.05)
    ymax = min(1.05, float(df['test_acc'].max() + df['test_std'].max()) + 0.08)
    if ymax - ymin < 0.15:
        mid = 0.5 * (ymin + ymax)
        ymin, ymax = mid - 0.08, mid + 0.08
    ax.set_ylim(ymin, ymax)

    train_lens = sorted(df['train_seq_len'].unique())
    for tl in train_lens:
        ax.axvline(tl, ls=':', lw=1.1, color='#adb5bd', zorder=0)
    if train_lens:
        ax.text(train_lens[0], ymax, f'  train L={train_lens[0]}',
                va='top', ha='left', fontsize=8, color='#6c757d')

    ax.legend(frameon=False, fontsize=8.5, ncol=2, loc='best')
    fig.tight_layout()
    return fig


def build_heatmap(df: pd.DataFrame, *, subtitle: str) -> plt.Figure:
    """Models × eval lengths, mean test accuracy."""
    models = sorted(df['model'].unique(), key=_family_key)
    lengths = sorted(df['eval_seq_len'].unique())
    mat = np.full((len(models), len(lengths)), np.nan)
    for i, name in enumerate(models):
        for j, L in enumerate(lengths):
            hit = df[(df['model'] == name) & (df['eval_seq_len'] == L)]
            if len(hit):
                mat[i, j] = float(hit['test_acc'].iloc[0])

    plt.rcParams.update({
        'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
        'figure.dpi': 120,
    })
    fig_w = max(5.5, 0.85 * len(lengths) + 2.8)
    fig_h = max(3.6, 0.45 * len(models) + 1.6)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(mat, aspect='auto', cmap='viridis', vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(lengths)))
    ax.set_xticklabels([str(L) for L in lengths])
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.set_xlabel('evaluation sequence length')
    ax.set_title(subtitle, loc='left')
    for i in range(len(models)):
        for j in range(len(lengths)):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f'{mat[i, j]:.2f}', ha='center', va='center',
                        color='white' if mat[i, j] < 0.55 else '#212529',
                        fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='test accuracy')
    fig.tight_layout()
    return fig


def _save(fig, out_dir, stem) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for ext in ('png', 'pdf'):
        p = os.path.join(out_dir, f'{stem}.{ext}')
        fig.savefig(p, dpi=200, bbox_inches='tight')
        paths.append(p)
    plt.close(fig)
    return paths


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--sweep-dir', required=True)
    p.add_argument('--name', required=True, help='sub-folder under figures/')
    p.add_argument('--figures-root', default=FIGURES_ROOT)
    p.add_argument('--subtitle', default=None)
    p.add_argument('--vocab-size', type=int, default=0,
                   help='marks chance at 1/(vocab_size-1) when > 1')
    p.add_argument('--chance', type=float, default=None)
    p.add_argument('--chance-label', default='chance')
    p.add_argument('--no-id', action='store_true',
                   help='do not overlay in-distribution results.csv points')
    p.add_argument('--summary-csv', default=None,
                   help='optional path for aggregated mean±std table '
                        '(default: <sweep-dir>/ood_summary.csv)')
    args = p.parse_args()

    runs = load_ood_runs(args.sweep_dir, include_id=not args.no_id)
    df = aggregate(runs)
    summary_csv = args.summary_csv or os.path.join(args.sweep_dir, 'ood_summary.csv')
    df.to_csv(summary_csv, index=False)
    print(f'wrote {summary_csv}')

    if args.chance is not None:
        chance = args.chance
    else:
        chance = 1.0 / (args.vocab_size - 1) if args.vocab_size > 1 else None

    train_sl = int(df['train_seq_len'].mode().iloc[0])
    subtitle = args.subtitle or (
        f'Length OOD — trained at L={train_sl} ({args.sweep_dir})'
    )
    out_dir = os.path.join(args.figures_root, args.name)

    fig = build_figure(df, subtitle=subtitle, chance=chance,
                       chance_label=args.chance_label)
    for path in _save(fig, out_dir, 'ood_length'):
        print(f'wrote {path}')

    heat = build_heatmap(df, subtitle=subtitle)
    for path in _save(heat, out_dir, 'ood_heatmap'):
        print(f'wrote {path}')

    print(f'\n{runs["seed"].nunique()} seed(s), {runs["model"].nunique()} models, '
          f'{runs["eval_seq_len"].nunique()} eval lengths')
    print(df.to_string(index=False,
                       formatters={'test_acc': '{:.4f}'.format,
                                   'test_std': '{:.4f}'.format,
                                   'test_ppl': '{:.3f}'.format}))


if __name__ == '__main__':
    main()
