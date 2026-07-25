#!/usr/bin/env python3
"""Figures for the iso-parameter sweeps produced by ``run_comparison_suite``.

The iso regimes sweep a 4-D grid -- (family, block size ``m``, scan impl, batch)
x sequence length -- which the existing plot scripts do not cover: they compare
scan impls at a *single* shape. This one keeps their styling (``plot_compare``'s
``PRETTY`` / ``STYLE`` / ``_save``) and renders three views into
``figures/<name>/``:

  * ``scan_impls.png``   -- throughput vs T, one panel per (batch, block size),
                            one line per scan impl. The raw scaling picture.
  * ``scan_latency.png`` -- the same grid in wall-clock ms/step. Both grids share
                            a single y-axis across every panel, so cells are
                            comparable across batch and block size by eye.
  * ``scan_speedup.png`` -- heatmap of the hero impl's speedup over the best
                            baseline, block size x T, one panel per batch.
                            Diverging about 1.0, so red cells are regressions.
  * ``families.png`` /   -- cross-architecture iso-parameter comparison in
    ``families_latency``    throughput and in ms/step: batch across columns,
                            BD-LRU / H-LRU / baselines down rows, sharing one
                            y-axis. Each block size of an LRU family is its own
                            curve, since m trades state size against parallelism
                            and the comparison between block sizes is the point.

Rows are keyed off ``run_tag``, which the suite writes as
``{family}-wd{m}-{impl}|batch_size={B}`` for BD-LRU / H-LRU and ``{family}|
batch_size={B}`` for the single-impl baselines (lstm, mamba2, deltanet, ...).

Partial CSVs are expected: the suite is resumable and cells that OOM'd or errored
are recorded with a non-``ok`` status, so every panel drops missing points rather
than failing.

Examples:
    uv run python -m scripts.plot_iso --csv results/iso-d128-iso1m/results.csv \
        --name iso-d128-iso1m

    uv run python -m scripts.plot_iso --csv results/iso-d1024-iso100m/results.csv \
        --name iso-d1024-iso100m --family hlru --hero triton_auto
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
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter

from scripts.plot_compare import FIGURES_ROOT, _pretty, _style, _save
from scripts.run_comparison_suite import _ISO_DPROD_RANKS, _ISO_LRU_IMPL_LABELS

# The suite writes the short run_tag *label* ('hopscan'), not the layer's
# implementation name ('hopscan_custom'), so invert its map to recover the
# names plot_compare styles by.
_LABEL_TO_IMPL = {label: impl for impl, label in _ISO_LRU_IMPL_LABELS.items()}

DEFAULT_HERO = 'triton_auto'
# families carrying a block size + a selectable scan impl
LRU_FAMILIES = ('bdlru', 'hlru')
# `orig` is an O(T) Python loop kept as a correctness anchor, not something to
# claim speedups against: at cells where it is the only other impl present the
# ratio runs to ~340x and says nothing about the kernel. Shown in the line
# plots, excluded from "best baseline".
BASELINE_EXCLUDE = frozenset({'orig'})

FAMILY_PRETTY = {
    'lstm': 'LSTM (cuDNN)', 'pdssm': 'PDSSM', 'mamba2': 'Mamba2',
    'deltanet': 'DeltaNet', 'bdlru': 'BD-LRU', 'hlru': 'H-LRU',
}
FAMILY_STYLE = {
    'bdlru': dict(color='#c1121f', marker='*', lw=2.6, z=6),
    'hlru': dict(color='#7b2cbf', marker='h', lw=2.2, z=5),
    'lstm': dict(color='#6c757d', marker='x', lw=1.6, z=2),
    'pdssm': dict(color='#1f77b4', marker='s', lw=1.8, z=3),
    'mamba2': dict(color='#2a9d8f', marker='^', lw=1.9, z=3),
    'deltanet': dict(color='#e9a13b', marker='D', lw=1.8, z=3),
}
_FALLBACK = dict(color='#333333', marker='.', lw=1.5, z=2)
# Block sizes of one family are shaded along a single hue ramp, so a curve's
# family reads off its marker and its `m` off how dark it is. Iso-parameter
# curves sit nearly on top of each other, so `m` also picks a dash pattern --
# shade alone is not separable where they overlap.
FAMILY_CMAP = {'bdlru': 'Reds', 'hlru': 'Purples'}
BLOCK_DASHES = ['-', (0, (5, 1.4)), (0, (1, 1.2)), (0, (6, 1.4, 1, 1.4)),
                (0, (3, 1.2, 1, 1.2, 1, 1.2))]

# The suite runs DeltaProduct at several Householder ranks, and writes each as
# its own family tag. Rank is DeltaProduct's state-width knob, so treat it the
# way block size is treated elsewhere: DeltaNet's marker, darkening by rank.
for _i, _rank in enumerate((2, 4, 8)):
    FAMILY_PRETTY[f'deltaproduct{_rank}'] = f'DeltaProduct n={_rank}'
    FAMILY_STYLE[f'deltaproduct{_rank}'] = dict(
        color=plt.get_cmap('YlOrBr')(0.45 + 0.22 * _i), marker='D', lw=1.7, z=3,
        ls=BLOCK_DASHES[_i + 1])
FAMILY_PRETTY.setdefault('deltaproduct', 'DeltaProduct')
FAMILY_STYLE.setdefault('deltaproduct', dict(color='#b5651d', marker='D', lw=1.7, z=3))

_TAG_RE = re.compile(r'^(?P<family>[a-z0-9]+)(?:-wd(?P<block>\d+)-(?P<impl>.+))?$')


def _parse_tag(tag: str) -> tuple[str, int | None, str | None]:
    """``bdlru-wd4-triton_auto|batch_size=32`` -> ``('bdlru', 4, 'triton_auto')``."""
    head = str(tag).split('|')[0]
    m = _TAG_RE.match(head)
    if not m:
        return head, None, None
    block = m.group('block')
    label = m.group('impl')
    return (m.group('family'), (int(block) if block else None),
            _LABEL_TO_IMPL.get(label, label))


def load(csv_paths: list[str]) -> pd.DataFrame:
    df = pd.concat([pd.read_csv(c) for c in csv_paths], ignore_index=True)
    if 'status' in df.columns:
        df = df[df['status'] == 'ok'].copy()
    if df.empty:
        raise SystemExit('No successful rows to plot (check the CSV / status column).')
    parsed = df['run_tag'].apply(_parse_tag)
    df['family'] = [p[0] for p in parsed]
    df['block'] = [p[1] for p in parsed]
    df['impl'] = [p[2] for p in parsed]
    df['tps_m'] = df['tokens_per_s'] / 1e6
    if 'peak_mem_mb' in df.columns:
        df['peak_gb'] = pd.to_numeric(df['peak_mem_mb'], errors='coerce') / 1024
    for col in ('batch_size', 'seq_len'):
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df.dropna(subset=['batch_size', 'seq_len'])


def _median(df: pd.DataFrame, col: str = 'tps_m') -> pd.DataFrame:
    """Collapse repeats to one value per (seq_len,) point."""
    return df.dropna(subset=['seq_len', col]).groupby('seq_len', as_index=False)[col].median()


def hero_only(df: pd.DataFrame, hero: str = DEFAULT_HERO) -> pd.DataFrame:
    """Restrict the LRU families to one scan impl, leaving baselines untouched.

    Cross-family figures need a named kernel rather than a per-cell best-of:
    taking the max over impls silently varies which kernel each point came from,
    so a family's curve could bend for dispatch reasons and read as architecture.
    """
    return df[df['impl'].isna() | (df['impl'] == hero)]


def _impl_order(impls) -> list[str]:
    """Hero last so it draws on top; otherwise alphabetical for stability."""
    rest = sorted(i for i in impls if i != DEFAULT_HERO)
    return rest + ([DEFAULT_HERO] if DEFAULT_HERO in impls else [])


def _log2_ticks(ax, values):
    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xticks(sorted(set(values)))
    ax.tick_params(axis='x', labelrotation=45)


# --------------------------------------------------------------------------- #
# (1) metric vs T, panel per (batch, block)
# --------------------------------------------------------------------------- #
# column -> (axis label, figure-title noun, which direction is good, how to pick
# the best of several impls at one point)
METRICS = {
    'tps_m': ('Throughput (M tokens/s)', 'throughput', 'higher is better', 'max'),
    'step_ms': ('Wall-clock time / step (ms)', 'wall-clock time', 'lower is better', 'min'),
    'peak_gb': ('Peak memory (GB)', 'peak memory', 'lower is better', 'min'),
}


def build_scan_impl_figure(df: pd.DataFrame, family: str, subtitle: str,
                           metric: str = 'tps_m') -> plt.Figure | None:
    """Grid of (batch x block) panels, one line per scan impl.

    Every panel shares one y-axis so cells can be compared by eye across both
    batch and block size, not just within a row.
    """
    fam = df[df['family'] == family]
    if fam.empty:
        return None
    batches = sorted(fam['batch_size'].unique())
    blocks = sorted(int(b) for b in fam['block'].dropna().unique())
    if not blocks:
        return None
    ylabel, noun, direction, _ = METRICS[metric]

    plt.rcParams.update({'font.size': 9, 'axes.titlesize': 9,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.dpi': 120})
    fig, axes = plt.subplots(len(batches), len(blocks), squeeze=False,
                             figsize=(3.1 * len(blocks), 2.7 * len(batches)),
                             sharex=True, sharey=True)
    seqs = sorted(fam['seq_len'].unique())
    for r, B in enumerate(batches):
        for c, m in enumerate(blocks):
            ax = axes[r][c]
            cell = fam[(fam['batch_size'] == B) & (fam['block'] == m)]
            for impl in _impl_order(cell['impl'].dropna().unique()):
                pts = _median(cell[cell['impl'] == impl], metric)
                if pts.empty:
                    continue
                s = _style(impl)
                ax.plot(pts['seq_len'], pts[metric], marker=s['marker'], color=s['color'],
                        lw=s['lw'], ms=5, label=_pretty(impl), zorder=s['z'])
            ax.set_yscale('log')
            _log2_ticks(ax, seqs)
            ax.grid(True, which='both', alpha=0.22)
            if r == 0:
                ax.set_title(f'm = {m}', fontsize=10, fontweight='bold')
            if c == 0:
                ax.set_ylabel(f'B = {int(B)}', fontsize=10, fontweight='bold')
            if r == len(batches) - 1:
                ax.set_xlabel('sequence length T')
    fig.supylabel(f'{ylabel}  \u2014  {direction}', fontsize=10, x=0.004)
    handles, labels = axes[0][0].get_legend_handles_labels()
    for row in axes:  # first populated panel may not be [0][0] on a partial run
        for ax in row:
            h, l = ax.get_legend_handles_labels()
            if len(l) > len(labels):
                handles, labels = h, l
    if labels:
        fig.legend(handles, labels, loc='lower center', ncol=len(labels),
                   fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(
        f'{FAMILY_PRETTY.get(family, family)} scan implementations: {noun} \u2014 {subtitle}',
        fontsize=13, fontweight='bold')
    # left margin keeps the shared y-label clear of the per-row "B = ..." labels
    fig.tight_layout(rect=(0.028, 0.03, 1, 0.97))
    return fig


# --------------------------------------------------------------------------- #
# (2) hero speedup heatmap over the best baseline
# --------------------------------------------------------------------------- #
def build_speedup_figure(df: pd.DataFrame, family: str, hero: str,
                         subtitle: str) -> plt.Figure | None:
    fam = df[df['family'] == family]
    if fam.empty or hero not in set(fam['impl'].dropna()):
        return None
    batches = sorted(fam['batch_size'].unique())
    blocks = sorted(int(b) for b in fam['block'].dropna().unique())
    seqs = sorted(fam['seq_len'].unique())
    if not blocks or not seqs:
        return None

    plt.rcParams.update({'font.size': 9, 'axes.titlesize': 9, 'figure.dpi': 120})
    fig, axes = plt.subplots(1, len(batches), squeeze=False,
                             figsize=(3.4 * len(batches), 0.52 * len(blocks) + 2.6))
    grids = []
    for B in batches:
        g = np.full((len(blocks), len(seqs)), np.nan)
        for i, m in enumerate(blocks):
            cell = fam[(fam['batch_size'] == B) & (fam['block'] == m)]
            hero_pts = dict(zip(*_median(cell[cell['impl'] == hero]).values.T))
            base = cell[~cell['impl'].isin(BASELINE_EXCLUDE | {hero})]
            if base.empty:
                continue
            # best baseline at each T, so the ratio is never flattered by a weak one
            best = base.dropna(subset=['seq_len', 'tps_m']).groupby('seq_len')['tps_m'].max()
            for j, T in enumerate(seqs):
                if T in hero_pts and T in best.index and best[T]:
                    g[i, j] = hero_pts[T] / best[T]
        grids.append(g)

    finite = np.concatenate([g[np.isfinite(g)] for g in grids]) if grids else np.array([])
    if finite.size == 0:
        return None
    # Colour on log2(ratio): a handful of 20x cells would otherwise flatten the
    # rest of the map. 0 stays neutral, so red is still exactly "slower".
    lo, hi = np.log2(float(finite.min())), np.log2(float(finite.max()))
    norm = TwoSlopeNorm(vmin=min(lo, -0.05), vcenter=0.0, vmax=max(hi, 0.05))

    for k, (B, g) in enumerate(zip(batches, grids)):
        ax = axes[0][k]
        im = ax.imshow(np.log2(g), cmap='RdYlGn', norm=norm, aspect='auto', origin='lower')
        ax.set_xticks(range(len(seqs)))
        ax.set_xticklabels([f'{int(s)}' for s in seqs], rotation=45, fontsize=8)
        ax.set_yticks(range(len(blocks)))
        ax.set_yticklabels([str(b) for b in blocks], fontsize=8)
        ax.set_title(f'B = {int(B)}', fontsize=10, fontweight='bold')
        ax.set_xlabel('T')
        if k == 0:
            ax.set_ylabel('block size m')
        for i in range(len(blocks)):
            for j in range(len(seqs)):
                if np.isfinite(g[i, j]):
                    ax.text(j, i, f'{g[i, j]:.2f}', ha='center', va='center', fontsize=7,
                            color='black')
    cbar = fig.colorbar(im, ax=axes[0].tolist(), fraction=0.025, pad=0.02,
                        label=f'{_pretty(hero)} speedup over best baseline (\u00d7)')
    ticks = [t for t in (-1, 0, 1, 2, 3, 4, 5) if norm.vmin <= t <= norm.vmax]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f'{2 ** t:g}\u00d7' for t in ticks])
    fig.suptitle(
        f'{FAMILY_PRETTY.get(family, family)}: {_pretty(hero)} vs best other impl \u2014 {subtitle}',
        fontsize=13, fontweight='bold')
    return fig


# --------------------------------------------------------------------------- #
# (3) cross-family iso-parameter throughput
# --------------------------------------------------------------------------- #
def _family_rows(df: pd.DataFrame, at_block: int | None = None
                 ) -> list[tuple[str, list[tuple[tuple[str, int | None], str, dict]]]]:
    """Curves grouped into one row per family, as ``(row label, series)``.

    A series entry is ``((family, m), legend label, style)``. Each block size of
    an LRU family is its own curve -- ``m`` sets how much of the state is
    recurrent vs. parallel, so collapsing it to the per-family best hides the
    trade-off. The non-block baselines share a final row. Built over the whole
    frame, so every panel in a row draws the same curves in the same order.
    """
    rows, baselines = [], []
    for family in sorted(df['family'].unique()):
        fam = df[df['family'] == family]
        style = FAMILY_STYLE.get(family, _FALLBACK)
        if not fam['block'].notna().any():
            baselines.append(((family, None), FAMILY_PRETTY.get(family, family), style))
            continue
        blocks = sorted(int(b) for b in fam['block'].dropna().unique())
        if at_block is not None:
            blocks = [b for b in blocks if b == at_block]
        cmap = plt.get_cmap(FAMILY_CMAP[family]) if family in FAMILY_CMAP else None
        # keep the ramp clear of the near-white end so light curves stay legible
        shades = np.linspace(0.48, 1.0, len(blocks)) if len(blocks) > 1 else [0.85]
        series = []
        for i, m in enumerate(blocks):
            s = dict(style, ls=BLOCK_DASHES[i % len(BLOCK_DASHES)])
            if cmap is not None:
                s['color'] = cmap(shades[i])
            series.append(((family, m), f'm = {m}', s))
        if series:
            rows.append((FAMILY_PRETTY.get(family, family), series))
    if baselines:
        rows.append(('Baselines', baselines))
    return rows


def build_family_figure(df: pd.DataFrame, subtitle: str, at_block: int | None = None,
                        metric: str = 'tps_m', hero: str = DEFAULT_HERO
                        ) -> plt.Figure | None:
    """Batch across columns, family down rows, on one shared y-axis.

    One family per row rather than all of them overlaid: iso-parameter curves
    land within a factor of a few of each other, so a single panel holding every
    (family, block size) is unreadable. The shared axis is what keeps the rows
    comparable after the split.
    """
    df = hero_only(df, hero)
    batches = sorted(df['batch_size'].unique())
    rows = _family_rows(df, at_block)
    if not batches or not rows:
        return None
    ylabel, noun, direction, _ = METRICS[metric]

    plt.rcParams.update({'font.size': 9, 'axes.titlesize': 9,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.dpi': 120})
    legend_w = 1.25  # right margin reserved for the per-row legends
    fig_w = 3.15 * len(batches) + legend_w
    fig, axes = plt.subplots(len(rows), len(batches), squeeze=False,
                             sharex=True, sharey=True,
                             figsize=(fig_w, 2.45 * len(rows) + 0.8))
    seqs = sorted(df['seq_len'].unique())
    for r, (row_label, series) in enumerate(rows):
        drawn: set[tuple[str, int | None]] = set()
        for c, B in enumerate(batches):
            ax = axes[r][c]
            sub = df[df['batch_size'] == B]
            for (family, m), _, s in series:
                cell = sub[sub['family'] == family]
                if m is not None:
                    cell = cell[cell['block'] == m]
                pts = _median(cell, metric)
                if pts.empty:
                    continue
                drawn.add((family, m))
                ax.plot(pts['seq_len'], pts[metric], marker=s['marker'], color=s['color'],
                        lw=s['lw'], ls=s.get('ls', '-'), ms=5, zorder=s['z'])
            ax.set_yscale('log')
            _log2_ticks(ax, seqs)
            ax.grid(True, which='both', alpha=0.22)
            if r == 0:
                ax.set_title(f'B = {int(B)}', fontsize=10, fontweight='bold')
            if r == len(rows) - 1:
                ax.set_xlabel('sequence length T')
            if c == 0:
                ax.set_ylabel(row_label, fontsize=10, fontweight='bold')
        # proxy handles: the legend holds every curve of the row in a stable
        # order, not just those the fullest panel happened to have data for
        handles = [Line2D([], [], color=s['color'], marker=s['marker'], lw=s['lw'],
                          ls=s.get('ls', '-'), ms=5)
                   for key, _, s in series if key in drawn]
        labels = [label for key, label, _ in series if key in drawn]
        if labels:
            axes[r][-1].legend(handles, labels, loc='center left', fontsize=8,
                               frameon=False, bbox_to_anchor=(1.02, 0.5),
                               handlelength=2.6, borderaxespad=0)
    fig.supylabel(f'{ylabel}  \u2014  {direction}', fontsize=10, x=0.004)
    note = (f'every block size, {_pretty(hero)}' if at_block is None
            else f'block m = {at_block}, {_pretty(hero)}')
    fig.suptitle(f'Iso-parameter {noun} by family ({note}) \u2014 {subtitle}',
                 fontsize=13, fontweight='bold')
    # left margin clears the shared y-label; right margin holds the row legends
    fig.tight_layout(rect=(0.022, 0.01, 1 - legend_w / fig_w, 0.96))
    return fig


# --------------------------------------------------------------------------- #
# (4) what the expressivity knob costs
# --------------------------------------------------------------------------- #
# BD-LRU / H-LRU widen their recurrent state with block size m; DeltaProduct
# does it with the number of Householder factors n. Both knobs buy expressivity
# at a cost linear in the knob and are swept at a fixed parameter budget, so the
# *shape* of throughput vs knob is comparable even though m and n are not the
# same unit. DeltaNet is the n=1 member of the DeltaProduct family.
_DPROD_RE = re.compile(r'^deltaproduct(?P<rank>\d+)$')
KNOB_GROUP_PRETTY = {
    'bdlru': 'BD-LRU  (block size $m$)', 'hlru': 'H-LRU  (window size $m$)',
    'delta': 'DeltaNet / DeltaProduct  (Householder rank $n$)',
}
# Deliberately not the per-family colours used elsewhere: this figure contrasts
# our two families against the delta baseline, so the baseline takes the warm
# colour and ours the cool ones.
KNOB_GROUP_STYLE = {
    'bdlru': dict(color='#1f6fb4', marker='o', lw=2.4, z=6),
    'hlru': dict(color='#7b2cbf', marker='s', lw=2.4, z=5),
    'delta': dict(color='#c1121f', marker='D', lw=2.4, z=4),
}
KNOB_XLABEL = ('block size $m$  (BD-LRU)      window size $m$  (H-LRU)      '
               'Householder rank $n$  (DeltaProduct)')
# DeltaProduct's sweep tops out at rank 8, so past 8 there is no cross-family
# comparison left to draw -- the LRU m=16 tail is a different question.
KNOB_CAP = float(max(_ISO_DPROD_RANKS))


def with_knob(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the state-width knob and the group it varies within."""
    out = df.copy()
    knob, group = [], []
    for family, block in zip(out['family'], out['block']):
        if family in LRU_FAMILIES:
            knob.append(block)
            group.append(family)
        elif family == 'deltanet':
            knob.append(1.0)
            group.append('delta')
        elif (m := _DPROD_RE.match(str(family))):
            knob.append(float(m.group('rank')))
            group.append('delta')
        else:
            knob.append(np.nan)
            group.append(None)
    out['knob'], out['kgroup'] = knob, group
    return out.dropna(subset=['knob', 'kgroup'])


def _knob_curves(df: pd.DataFrame, batch: float, seq: float,
                 metric: str = 'tps_m') -> dict[str, pd.Series]:
    """``group -> metric indexed by knob`` at one shape."""
    cell = df[(df['batch_size'] == batch) & (df['seq_len'] == seq)]
    curves = {}
    for group, g in cell.groupby('kgroup'):
        s = g.dropna(subset=[metric]).groupby('knob')[metric].median().sort_index()
        if len(s) > 1:
            curves[group] = s
    return curves


def _pick_knob_batch(df: pd.DataFrame) -> float | None:
    """Batch where the knob sweep is most complete -- the fairest one to show.

    Scored on the *shared* knob range: a batch where one group reached m=16 but
    the others stopped at 1 would otherwise win while showing nothing.
    """
    best, best_score = None, -1
    for B, g in df.groupby('batch_size'):
        score = 0
        for _, cell in g.groupby('seq_len'):
            per_group = cell.groupby('kgroup')['knob'].nunique()
            if len(per_group) > 1:
                score += int(per_group.min()) * len(per_group)
        if score > best_score:
            best, best_score = B, score
    return None if best_score <= 0 else best


def build_expressivity_figure(df: pd.DataFrame, subtitle: str, hero: str = DEFAULT_HERO,
                              metric: str = 'tps_m', max_knob: float = KNOB_CAP
                              ) -> plt.Figure | None:
    """The metric against each family's width knob, over a batch x length grid.

    Only absolute curves are drawn. A row normalised to ``knob=1`` would carry
    no extra shape on a log axis -- dividing by a constant just slides a curve
    vertically -- so the grid spends its space on batch and length instead, on
    one shared y-axis so every panel is comparable.
    """
    if metric not in df.columns:  # older CSVs predate the column
        return None
    kdf = with_knob(hero_only(df, hero))
    if kdf.empty:
        return None
    kdf = kdf[kdf['knob'] <= max_knob]
    plottable = {(B, T) for B in kdf['batch_size'].unique()
                 for T in kdf['seq_len'].unique()
                 if len(_knob_curves(kdf, B, T, metric)) > 1}
    if not plottable:
        return None
    batches = sorted({B for B, _ in plottable})
    seqs = sorted({T for _, T in plottable})
    ylabel, noun, direction, _ = METRICS[metric]

    plt.rcParams.update({'font.size': 9, 'axes.titlesize': 9,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.dpi': 120})
    fig_h = 2.25 * len(batches) + 1.25
    fig, axes = plt.subplots(len(batches), len(seqs), squeeze=False,
                             sharex=True, sharey=True,
                             figsize=(2.45 * len(seqs) + 0.7, fig_h))
    knobs = sorted(kdf['knob'].unique())
    seen: list[str] = []
    for r, B in enumerate(batches):
        for c, T in enumerate(seqs):
            ax = axes[r][c]
            curves = _knob_curves(kdf, B, T, metric)
            for group in ('bdlru', 'hlru', 'delta'):
                if group not in curves:
                    continue
                s, st = curves[group], KNOB_GROUP_STYLE[group]
                seen.append(group) if group not in seen else None
                ax.plot(s.index, s.values, color=st['color'], marker=st['marker'],
                        lw=st['lw'], ms=5.5, zorder=st['z'])
            ax.set_yscale('log')
            ax.set_xscale('log', base=2)
            ax.set_xticks(knobs)
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.grid(True, which='major', alpha=0.22)
            ax.set_xlim(knobs[0] / 1.2, knobs[-1] * 1.15)
            if r == 0:
                ax.set_title(f'T = {int(T)}', fontsize=10, fontweight='bold')
            if c == 0:
                ax.set_ylabel(f'B = {int(B)}', fontsize=10, fontweight='bold')
    fig.supylabel(f'{ylabel}  \u2014  {direction}', fontsize=10, x=0.004)
    handles = [Line2D([], [], color=KNOB_GROUP_STYLE[g]['color'], lw=2.4, ms=5.5,
                      marker=KNOB_GROUP_STYLE[g]['marker']) for g in seen]
    fig.legend(handles, [KNOB_GROUP_PRETTY[g] for g in seen], loc='lower center',
               ncol=len(seen), fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, 0.42 / fig_h))
    fig.supxlabel(KNOB_XLABEL, fontsize=9, y=0.12 / fig_h)
    fig.suptitle(f'Cost of matrix rank / expressivity: {noun} at a fixed parameter '
                 f'budget ({_pretty(hero)}) \u2014 {subtitle}',
                 fontsize=12, fontweight='bold')
    # bottom band holds the legend and the shared x-label
    fig.tight_layout(rect=(0.022, 0.85 / fig_h, 1, 1 - 0.42 / fig_h))
    return fig


def summarize_expressivity(df: pd.DataFrame, hero: str = DEFAULT_HERO,
                           batch: float | None = None,
                           max_knob: float = KNOB_CAP) -> list[str]:
    """Retention at the widest knob every group shares, per sequence length."""
    kdf = with_knob(hero_only(df, hero))
    if kdf.empty:
        return []
    kdf = kdf[kdf['knob'] <= max_knob]
    batch = batch if batch is not None else _pick_knob_batch(kdf)
    if batch is None:
        return []
    lines = []
    for T in sorted(kdf['seq_len'].unique()):
        curves = _knob_curves(kdf, batch, T)
        if len(curves) < 2:
            continue
        # only compare where every group actually reached the same knob
        shared = set.intersection(*(set(s.index) for s in curves.values()))
        if len(shared) < 2:
            continue
        k = max(shared)
        parts = ' '.join(f'{g}={curves[g][k] / curves[g].iloc[0]:.2f}' for g in curves)
        lines.append(f'B={int(batch)} T={int(T)}: retained at knob={int(k)}  {parts}')
    return lines


def summarize(df: pd.DataFrame, family: str, hero: str) -> str:
    """One-line headline: where the hero wins most, and whether it ever loses."""
    fam = df[(df['family'] == family) & df['block'].notna()]
    if fam.empty or hero not in set(fam['impl'].dropna()):
        return ''
    best, worst, ctx, wctx = 0.0, float('inf'), '', ''
    for (B, m), cell in fam.groupby(['batch_size', 'block']):
        hero_pts = dict(zip(*_median(cell[cell['impl'] == hero]).values.T))
        base = cell[~cell['impl'].isin(BASELINE_EXCLUDE | {hero})].dropna(
            subset=['seq_len', 'tps_m'])
        if base.empty:
            continue
        bb = base.groupby('seq_len')['tps_m'].max()
        for T, hv in hero_pts.items():
            if T in bb.index and bb[T]:
                r = hv / bb[T]
                if r > best:
                    best, ctx = r, f'B={int(B)} m={int(m)} T={int(T)}'
                if r < worst:
                    worst, wctx = r, f'B={int(B)} m={int(m)} T={int(T)}'
    if not ctx:
        return ''
    return (f'{family}: {_pretty(hero)} best {best:.2f}x ({ctx}), '
            f'worst {worst:.2f}x ({wctx})')


def generate(csv_paths: list[str], name: str, *, hero: str = DEFAULT_HERO,
             family: str | None = None, at_block: int | None = None,
             figures_root: str = FIGURES_ROOT) -> list[str]:
    df = load(csv_paths)
    out_dir = os.path.join(figures_root, name)
    subtitle = name
    families = [family] if family else [f for f in LRU_FAMILIES if f in set(df['family'])]

    written: list[str] = []
    for fam in families:
        suffix = '' if len(families) == 1 and family else f'_{fam}'
        fig = build_scan_impl_figure(df, fam, subtitle, metric='tps_m')
        if fig is not None:
            written += _save(fig, out_dir, f'scan_impls{suffix}')
        fig = build_scan_impl_figure(df, fam, subtitle, metric='step_ms')
        if fig is not None:
            written += _save(fig, out_dir, f'scan_latency{suffix}')
        fig = build_speedup_figure(df, fam, hero, subtitle)
        if fig is not None:
            written += _save(fig, out_dir, f'scan_speedup{suffix}')
        line = summarize(df, fam, hero)
        if line:
            print('  ' + line)
    for metric, stem in (('tps_m', 'families'), ('step_ms', 'families_latency')):
        fig = build_family_figure(df, subtitle, at_block, metric=metric, hero=hero)
        if fig is not None:
            written += _save(fig, out_dir, stem)
    for metric, stem in (('tps_m', 'expressivity_scaling'),
                         ('step_ms', 'expressivity_latency'),
                         ('peak_gb', 'expressivity_memory')):
        fig = build_expressivity_figure(df, subtitle, hero=hero, metric=metric)
        if fig is not None:
            written += _save(fig, out_dir, stem)
    for line in summarize_expressivity(df, hero):
        print('  ' + line)
    return written


def get_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--csv', nargs='+', required=True, help='iso result CSV(s)')
    p.add_argument('--name', required=True, help='sub-folder under figures/')
    p.add_argument('--hero', default=DEFAULT_HERO, help=f'featured impl (default: {DEFAULT_HERO})')
    p.add_argument('--family', default=None, choices=list(LRU_FAMILIES),
                   help='restrict the per-impl figures to one family (default: all present)')
    p.add_argument('--at-block', type=int, default=None,
                   help='fix the block size in the family figure (default: best per length)')
    p.add_argument('--figures-root', default=FIGURES_ROOT)
    return p.parse_args()


def main():
    args = get_args()
    written = generate(args.csv, args.name, hero=args.hero, family=args.family,
                       at_block=args.at_block, figures_root=args.figures_root)
    if not written:
        print('Nothing written (no plottable rows).')
    for path in written:
        print(f'Wrote {path}')


if __name__ == '__main__':
    main()
