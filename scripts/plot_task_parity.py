#!/usr/bin/env python3
"""Show whether a model sweep actually separates on a task, or merely ties.

Reads the per-run ``results.csv`` written by ``train.py`` for every job under a
sweep directory (as laid out by ``launch_mem_iso1m.sh``: ``<sweep>/runs/<name>/``)
and renders, into ``figures/<name>/``:

  * ``parity.png`` / ``.pdf`` -- two panels:
      (a) final train vs test accuracy per model, on the full 0..1 scale, with the
          task's chance level marked
      (b) the same test accuracies zoomed to their own range, with error bars from
          the epoch-to-epoch fluctuation of each run's own test accuracy over its
          last ``--tail-epochs`` epochs

Panel (a) carries the result -- identical train accuracy, visually identical test
accuracy -- and panel (b) quantifies how narrow the remaining band is. The error
bars describe how settled each individual run is; they are *not* a significance
test. Ordering models needs repeated seeds, which a single-seed sweep cannot
provide no matter how stable each run looks at its own plateau.

Example:
    uv run python -m scripts.plot_task_parity --sweep-dir logs_mem_iso1m \
        --name mem-iso1m --vocab-size 4096
"""

from __future__ import annotations

import os
import re
import glob
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


FIGURES_ROOT = 'figures'

# families in the order they should be drawn, with a colour each
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
    """Longest-prefix match so ``deltaprod2`` is not swallowed by ``deltaprod``."""
    hits = [(fam, col, i) for i, (fam, col) in enumerate(FAMILY_COLORS)
            if name == fam or name.startswith(fam)]
    if not hits:
        return None
    return max(hits, key=lambda t: len(t[0]))


def _family_key(name: str) -> tuple:
    """Sort key grouping by family (in FAMILY_COLORS order) then by size knob."""
    hit = _match_family(name)
    if hit is None:
        return (len(FAMILY_COLORS), 0)
    fam, _, i = hit
    # BD-/H-LRU use ``wd<m>``; DeltaProduct uses a trailing householder rank.
    if 'wd' in name:
        digits = ''.join(c for c in name.split('wd')[-1] if c.isdigit())
    else:
        digits = ''.join(c for c in name[len(fam):] if c.isdigit())
    return (i, int(digits) if digits else 0)


def _color(name: str) -> str:
    hit = _match_family(name)
    return hit[1] if hit else '#495057'


def load_runs(sweep_dir: str, tail_epochs: int) -> pd.DataFrame:
    """Collect final metrics + a within-run noise estimate for every job.

    ``results.csv`` holds only the final numbers, so the run's own ``metrics.csv``
    supplies the late-training spread of test accuracy used as the error bar.

    One row per run; a model trained at several seeds contributes several rows,
    which ``aggregate_seeds`` then collapses.
    """
    rows = []
    for res in sorted(glob.glob(os.path.join(sweep_dir, 'runs', '*', '*', 'results.csv'))):
        name = res.split(os.sep)[-3]
        seed_match = re.search(r'_s-(\d+)_', res.split(os.sep)[-2])
        final = pd.read_csv(res).iloc[-1]

        tail_std, best_test = np.nan, np.nan
        metrics = os.path.join(os.path.dirname(res), 'logs', 'metrics.csv')
        if os.path.exists(metrics):
            m = pd.read_csv(metrics)
            if 'test/Accuracy_epoch' in m:
                # train.py ends with two trainer.validate() passes (train_dl then
                # test_dl) that log into this same column, so the last two entries
                # are not epochs; the train-set pass in particular reads ~1.0
                per_epoch = m['test/Accuracy_epoch'].dropna().iloc[:-2]
                if len(per_epoch):
                    best_test = float(per_epoch.max())
                if len(per_epoch) > tail_epochs:
                    tail_std = float(per_epoch.tail(tail_epochs).std())

        rows.append(dict(model=name,
                         seed=int(seed_match.group(1)) if seed_match else -1,
                         train_acc=float(final['train_acc']),
                         test_acc=float(final['test_acc']), best_test=best_test,
                         params=int(final['model_size']), tail_std=tail_std))

    if not rows:
        raise SystemExit(f'no results.csv found under {sweep_dir}/runs/*/')
    df = pd.DataFrame(rows)
    return df.sort_values('model', key=lambda s: s.map(_family_key)).reset_index(drop=True)


def aggregate_seeds(runs: pd.DataFrame) -> tuple:
    """Collapse per-seed runs to one row per model.

    With replicates the error bar becomes the across-seed standard deviation,
    which is the only noise scale that licenses ranking models. With a single
    seed it falls back to the run's own late-training fluctuation, which measures
    how settled that one run is and nothing about seed variability.
    """
    n_seeds = int(runs.groupby('model')['seed'].nunique().max())
    if n_seeds < 2:
        return runs.assign(err=runs['tail_std']), n_seeds

    g = runs.groupby('model', sort=False)
    df = g.agg(train_acc=('train_acc', 'mean'), test_acc=('test_acc', 'mean'),
               test_std=('test_acc', 'std'), best_test=('best_test', 'mean'),
               params=('params', 'first'), tail_std=('tail_std', 'mean'),
               n=('seed', 'nunique')).reset_index()
    df = df.sort_values('model', key=lambda s: s.map(_family_key)).reset_index(drop=True)
    return df.assign(err=df['test_std']), n_seeds


def _panel_full_scale(ax, df, chance, chance_label='chance'):
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    x = np.arange(len(df))
    w = 0.38
    ax.bar(x - w / 2, df['train_acc'], w, color='#ced4da', edgecolor='white')
    ax.bar(x + w / 2, df['test_acc'], w, edgecolor='white',
           color=[_color(n) for n in df['model']])

    for xi, v in zip(x - w / 2, df['train_acc']):
        ax.text(xi, v + 0.02, f'{v:.2f}', ha='center', fontsize=8, color='#495057')
    for xi, v in zip(x + w / 2, df['test_acc']):
        ax.text(xi, v + 0.02, f'{v:.3f}', ha='center', fontsize=8)

    handles = [Patch(facecolor='#ced4da', label='train'),
               Patch(facecolor='#495057', label='test (colour = family)')]
    if chance:
        ax.axhline(chance, ls=':', lw=1.2, color='#495057')
        handles.append(Line2D([], [], ls=':', color='#495057',
                              label=f'{chance_label} = {chance:.4g}'))

    tr, te = df['train_acc'], df['test_acc']
    if tr.min() > 0.999:
        fit = f'all {len(df)} models fit the train set perfectly (1.000)'
    else:
        fit = f'train accuracy {tr.min():.3f}\u2013{tr.max():.3f}'
    ax.set_xticks(x)
    ax.set_xticklabels(df['model'], rotation=30, ha='right')
    ax.set_ylim(0, max(1.0, float(tr.max())) * 1.3)
    ax.set_ylabel('accuracy')
    ax.set_title(f'(a) {fit};\n'
                 f'test accuracy {te.min():.3f}\u2013{te.max():.3f}', loc='left')
    ax.legend(handles=handles, frameon=False, loc='upper center', ncol=3,
              fontsize=8.5)


def _panel_zoom(ax, df, tail_epochs, n_seeds):
    x = np.arange(len(df))
    acc = df['test_acc'].to_numpy()
    err = df['err'].to_numpy()
    spread_pp = (acc.max() - acc.min()) * 100
    noise_pp = float(np.nanmean(err)) * 100 if not np.all(np.isnan(err)) else np.nan

    ax.axhspan(acc.min(), acc.max(), color='#adb5bd', alpha=0.28, zorder=0,
               label=f'full spread across models: {spread_pp:.2f} pp')
    ax.axhline(acc.mean(), ls='--', lw=1.2, color='#495057', zorder=1,
               label=f'mean = {acc.mean():.4f}')

    # errorbar takes a single marker colour, so the per-model markers go on top
    err_label = (f'std across {n_seeds} seeds' if n_seeds > 1
                 else f'epoch-to-epoch spread, last {tail_epochs} epochs')
    ax.errorbar(x, acc, yerr=err, fmt='none', capsize=4, elinewidth=1.4,
                ecolor='#495057', zorder=2, label=err_label)
    ax.scatter(x, acc, s=60, c=[_color(n) for n in df['model']],
               edgecolors='white', linewidths=1.2, zorder=3,
               label='mean over seeds' if n_seeds > 1 else 'final epoch')

    # a best-epoch well above the final one means the comparison is being made
    # after the models have already started overfitting
    best = df['best_test'].to_numpy()
    finite_err = np.nan_to_num(err)
    lo, hi = float((acc - finite_err).min()), float((acc + finite_err).max())
    if not np.all(np.isnan(best)) and np.nanmax(best) - acc.max() > 0.2 * max(spread_pp / 100, 1e-6):
        ax.scatter(x, best, s=55, facecolors='none', zorder=3,
                   edgecolors=[_color(n) for n in df['model']], linewidths=1.4,
                   label='best epoch')
        lo, hi = min(lo, np.nanmin(best)), max(hi, np.nanmax(best))

    ax.set_xticks(x)
    ax.set_xticklabels(df['model'], rotation=30, ha='right')
    ax.set_xlim(-0.6, len(df) - 0.4)
    # pad below so extremes are off the frame, and generously above to keep the
    # legend clear of the markers
    span = max(hi - lo, 1e-4)
    ax.set_ylim(lo - 0.35 * span, hi + 1.0 * span)
    ax.set_ylabel('test accuracy')
    if n_seeds > 1 and not np.isnan(noise_pp):
        inside = spread_pp <= 2 * noise_pp
        verdict = ('inside seed noise: the ordering is not resolved'
                   if inside else 'larger than seed noise')
        ax.set_title(f'(b) between-model spread {spread_pp:.2f} pp vs '
                     f'{noise_pp:.2f} pp seed noise\n{verdict}', loc='left')
    else:
        ax.set_title(f'(b) zoomed to the test-accuracy band: {spread_pp:.2f} pp wide\n'
                     f'({spread_pp / acc.mean():.2f}% relative) \u2014 one seed, so read '
                     f'no ranking into it', loc='left')
    ax.legend(frameon=False, loc='upper right', fontsize=8)
    return spread_pp, noise_pp


def build_simple_figure(df, *, chance, n_seeds, chance_label='chance', title=None):
    """A single-panel, minimally annotated version of panel (a).

    The two-panel figure argues a point in prose; this one is meant to be
    dropped into a paper where the caption carries the wording, so everything
    that restates the data in words is gone: no suptitle, no verdict line, no
    per-panel summary, and no labels on the train bars (which are all 1.000).
    """
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    plt.rcParams.update({
        'font.size': 10, 'axes.titlesize': 10,
        'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 120,
    })
    # widen with the model count so 10-family FR/mem sweeps stay readable
    fig, ax = plt.subplots(figsize=(max(6.4, 0.72 * len(df) + 1.6), 3.4))

    x = np.arange(len(df))
    w = 0.38
    err = np.nan_to_num(df['err'].to_numpy()) if 'err' in df else None
    ax.bar(x - w / 2, df['train_acc'], w, color='#ced4da', edgecolor='white')
    ax.bar(x + w / 2, df['test_acc'], w, edgecolor='white',
           color=[_color(n) for n in df['model']],
           yerr=err, capsize=3, error_kw=dict(elinewidth=1.1, ecolor='#212529'))

    # A fixed 0..1 axis only reads well when something reaches the top of it. On a
    # task nobody solves, every bar would be a sliver along the floor, so the axis
    # follows the data instead, leaving headroom for the labels.
    data_max = max(float(df['train_acc'].max()),
                   float((df['test_acc'] + err).max()), chance or 0.0)
    top = 1.08 if data_max > 0.9 else data_max * 1.45

    # the std here is a fraction of a percentage point, so the error bar is barely
    # a pixel tall; the label carries the number that the bar cannot
    for xi, v, e in zip(x + w / 2, df['test_acc'], err):
        ax.text(xi, v + 0.032 * top, f'{v:.3f}\n$\\pm${e:.3f}', ha='center',
                fontsize=7, linespacing=1.35)

    handles = [Patch(facecolor='#ced4da', label='train'),
               Patch(facecolor='#495057', label='test')]
    if err is not None and np.any(err > 0):
        handles.append(Line2D([], [], color='#212529', lw=1.1,
                              label=f'$\\pm$std ({n_seeds} seeds)' if n_seeds > 1
                                    else '$\\pm$spread'))
    # a baseline far below the bars just draws a line along the axis floor, so
    # only mark it when it is actually distinguishable from zero
    if chance and chance > 0.02:
        ax.axhline(chance, ls=':', lw=1.2, color='#495057')
        handles.append(Line2D([], [], ls=':', color='#495057',
                              label=f'{chance_label} = {chance:.3g}'))

    ax.set_xticks(x)
    ax.set_xticklabels(df['model'], rotation=30, ha='right')
    ax.set_ylim(0, top)
    if top > 0.9:
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    else:
        ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.set_ylabel('accuracy')
    if title:
        ax.set_title(title, loc='left')
    # train bars reach the top of the axes, so the legend goes outside the frame
    ax.legend(handles=handles, frameon=False, ncol=len(handles), fontsize=8.5,
              loc='lower left', bbox_to_anchor=(0, 1.0), borderaxespad=0.2)
    fig.tight_layout()
    return fig


def build_figure(df, *, chance, tail_epochs, subtitle, n_seeds,
                 chance_label='chance'):
    plt.rcParams.update({
        'font.size': 10, 'axes.titlesize': 10,
        'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 120,
    })
    fig_w = max(13.0, 1.15 * len(df) + 4.0)
    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(fig_w, 5.4))
    _panel_full_scale(ax_full, df, chance, chance_label)
    spread_pp, noise_pp = _panel_zoom(ax_zoom, df, tail_epochs, n_seeds)
    if chance:
        # the zoom hides the baseline unless it is inside the plotted window, and
        # "all models at chance" is exactly the case the zoom must not obscure
        lo, hi = ax_zoom.get_ylim()
        if lo <= chance <= hi:
            ax_zoom.axhline(chance, ls=':', lw=1.2, color='#495057',
                            label=f'{chance_label} = {chance:.4g}')
            ax_zoom.legend(frameon=False, loc='upper right', fontsize=8)

    seeds = f'{n_seeds} seeds' if n_seeds > 1 else '1 seed'
    fig.suptitle(f'Model comparison \u2014 {subtitle}',
                 fontsize=13, fontweight='bold', y=0.99)
    verdict = (f'test accuracy {df["test_acc"].min():.4f}\u2013'
               f'{df["test_acc"].max():.4f} across all {len(df)} models '
               f'({seeds}), a band {spread_pp:.2f} pp wide')
    if chance:
        verdict += f'; {chance_label} = {chance:.4g}'
        if df['test_acc'].max() <= chance:
            verdict += ' \u2014 every model is at or below it'
    fig.text(0.5, 0.935, verdict, ha='center', fontsize=10, color='#d1495b')
    fig.tight_layout(rect=(0, 0.02, 1, 0.90))
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


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--sweep-dir', required=True,
                   help='directory holding runs/<name>/ (e.g. logs_mem_iso1m)')
    p.add_argument('--name', required=True, help='sub-folder under figures/')
    p.add_argument('--vocab-size', type=int, default=0,
                   help='task vocab size; marks chance at 1/(vocab_size-1), which '
                        'only holds when targets span the whole non-special vocab')
    p.add_argument('--chance', type=float, default=None,
                   help='explicit baseline, overriding --vocab-size. Needed for '
                        'tasks that supervise a sub-vocabulary: fuzzy in-context '
                        'recall draws its targets from the value half only')
    p.add_argument('--chance-label', default='chance')
    p.add_argument('--tail-epochs', type=int, default=20,
                   help='epochs of test accuracy used for the noise estimate')
    p.add_argument('--subtitle', default=None)
    p.add_argument('--figures-root', default=FIGURES_ROOT)
    p.add_argument('--simple', action='store_true',
                   help='also write parity_simple.{png,pdf}: the left panel only, '
                        'stripped of the explanatory text, for use with a caption')
    p.add_argument('--simple-title', default=None,
                   help='short title for the --simple figure (default: none)')
    args = p.parse_args()

    runs = load_runs(args.sweep_dir, args.tail_epochs)
    df, n_seeds = aggregate_seeds(runs)
    if args.chance is not None:
        chance = args.chance
    else:
        chance = 1.0 / (args.vocab_size - 1) if args.vocab_size > 1 else 0.0
    fig = build_figure(df, chance=chance, tail_epochs=args.tail_epochs,
                       subtitle=args.subtitle or args.sweep_dir, n_seeds=n_seeds,
                       chance_label=args.chance_label)
    out_dir = os.path.join(args.figures_root, args.name)
    for path in _save(fig, out_dir, 'parity'):
        print(f'wrote {path}')

    if args.simple:
        simple = build_simple_figure(df, chance=chance, n_seeds=n_seeds,
                                     chance_label=args.chance_label,
                                     title=args.simple_title)
        for path in _save(simple, out_dir, 'parity_simple'):
            print(f'wrote {path}')

    fmt = {c: '{:.4f}'.format for c in
           ('train_acc', 'test_acc', 'best_test', 'tail_std', 'test_std', 'err')}
    print(f'\n{len(runs)} runs, {n_seeds} seed(s) per model')
    print(df.to_string(index=False,
                       formatters={k: v for k, v in fmt.items() if k in df}))


if __name__ == '__main__':
    main()
