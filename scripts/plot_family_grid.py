#!/usr/bin/env python3
"""Latency + feasibility heatmap across families at a fixed batch size.

A single image that answers "how does each family's per-step latency scale with
sequence length, and where does it stop being feasible?" at one batch size.

  * rows    -- one per family. The LRU families (BD-LRU / H-LRU) are shown at a
               single scan impl (the hero, default ``triton_auto``) with one row
               per block size ``m``; the single-impl baselines (LSTM, PDSSM,
               Mamba2, DeltaNet, DeltaProduct n=2/4/8) get one row each.
  * columns -- sequence length T (the scaling axis).
  * cells   -- median wall-clock ms/step for ``ok`` cells, coloured on a log
               scale (green = fast, red = slow) with the number written in.
               Non-``ok`` cells are drawn categorically so feasibility reads at
               a glance:
                   OOM   -- out of memory
                   >=cap -- hit the per-step / compile wall-clock ceiling
                   err   -- runtime/compile error
                   .     -- not run

Unlike ``plot_iso`` (which drops every non-``ok`` row), this keeps them so the
OOM / capped frontier is part of the picture.

Example:
    uv run python -m scripts.plot_family_grid \
        --csv results/iso-d128-iso1m/results.csv --name iso-d128-iso1m --batch 4
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
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle, Patch

from scripts.plot_compare import FIGURES_ROOT, _save
from scripts.plot_iso import _parse_tag, FAMILY_PRETTY, DEFAULT_HERO, LRU_FAMILIES

# Order families down the page: our two LRU families (by block size) first, then
# the baselines in a stable order. Anything present but unlisted is appended.
LRU_ORDER = ('bdlru', 'hlru')
BASELINE_ORDER = ('lstm', 'pdssm', 'mamba2', 'deltanet',
                  'deltaproduct2', 'deltaproduct4', 'deltaproduct8')

# categorical cell colours for the non-`ok` states
STATE_COLOR = {'oom': '#2b2b2b', 'cap': '#f4a259', 'err': '#9aa0a6', 'miss': '#f2f2f2'}
STATE_TEXT = {'oom': 'OOM', 'cap': '\u2265cap', 'err': 'err', 'miss': '\u00b7'}


def _nominal_batch(run_tag: str) -> float:
    m = re.search(r'batch_size=(\d+)', str(run_tag))
    return float(m.group(1)) if m else np.nan


def _classify(status: str) -> str:
    s = str(status)
    if s == 'ok':
        return 'ok'
    if s.startswith('OOM'):
        return 'oom'
    if s.startswith('capped'):
        return 'cap'
    return 'err'


def load(csv_path: str) -> pd.DataFrame:
    """All rows (every status), tagged with family / block / impl / nominal batch."""
    df = pd.read_csv(csv_path)
    parsed = df['run_tag'].apply(_parse_tag)
    df['family'] = [p[0] for p in parsed]
    df['block'] = [p[1] for p in parsed]
    df['impl'] = [p[2] for p in parsed]
    df['nb'] = df['run_tag'].apply(_nominal_batch)
    df['seq_len'] = pd.to_numeric(df['seq_len'], errors='coerce')
    df['step_ms'] = pd.to_numeric(df['step_ms'], errors='coerce')
    df['state'] = df['status'].apply(_classify)
    return df


def _rows(df: pd.DataFrame, hero: str) -> list[tuple[str, pd.DataFrame]]:
    """Ordered ``(row label, sub-frame)`` list, one entry per family/block row."""
    present = set(df['family'])
    out: list[tuple[str, pd.DataFrame]] = []
    for fam in LRU_ORDER:
        if fam not in present:
            continue
        sub = df[(df['family'] == fam) & (df['impl'] == hero)]
        for m in sorted(int(b) for b in sub['block'].dropna().unique()):
            out.append((f'{FAMILY_PRETTY.get(fam, fam)}  m={m}',
                        sub[sub['block'] == m]))
    ordered_base = [b for b in BASELINE_ORDER if b in present]
    ordered_base += sorted(f for f in present
                           if f not in LRU_ORDER and f not in BASELINE_ORDER)
    for fam in ordered_base:
        out.append((FAMILY_PRETTY.get(fam, fam), df[df['family'] == fam]))
    return out


def _fmt(v: float) -> str:
    if v < 10:
        return f'{v:.1f}'
    if v < 10000:
        return f'{v:.0f}'
    return f'{v / 1000:.0f}k'


def _cell(sub: pd.DataFrame, T: float) -> tuple[str, float]:
    """One cell's ``(state, step_ms)`` at sequence length ``T``.

    Prefer an ``ok`` measurement; otherwise report the most informative failure
    (capped over OOM over error), so a family that OOMs at one repeat but is
    really capped is not mislabelled.
    """
    c = sub[sub['seq_len'] == T]
    if c.empty:
        return 'miss', np.nan
    ok = c[c['state'] == 'ok']
    if not ok.empty:
        return 'ok', float(ok['step_ms'].median())
    for st in ('cap', 'oom', 'err'):
        if (c['state'] == st).any():
            return st, np.nan
    return 'miss', np.nan


def build(df: pd.DataFrame, batch: float, subtitle: str,
          hero: str = DEFAULT_HERO, exclude: tuple[str, ...] = ()) -> plt.Figure | None:
    df = df[df['nb'] == batch]
    if exclude:
        df = df[~df['family'].isin(exclude)]
    if df.empty:
        return None
    rows = _rows(df, hero)
    seqs = sorted(df['seq_len'].dropna().unique())
    if not rows or not seqs:
        return None
    R, C = len(rows), len(seqs)

    # colour scale over the ok latencies only, so failures don't skew it
    ok_vals = df[df['state'] == 'ok']['step_ms'].dropna()
    vmin = max(float(ok_vals.min()), 1e-2) if not ok_vals.empty else 1.0
    vmax = float(ok_vals.max()) if not ok_vals.empty else 10.0
    norm = LogNorm(vmin=vmin, vmax=max(vmax, vmin * 1.01))
    cmap = plt.get_cmap('RdYlGn_r')

    plt.rcParams.update({'font.size': 9, 'figure.dpi': 130})
    fig, ax = plt.subplots(figsize=(1.05 * C + 3.4, 0.46 * R + 1.9))

    for i, (_, sub) in enumerate(rows):
        y = R - 1 - i                       # first row on top
        for j, T in enumerate(seqs):
            state, v = _cell(sub, T)
            if state == 'ok':
                face = cmap(norm(v))
                txt, tcol, hatch = _fmt(v), _text_color(face), None
            else:
                face = STATE_COLOR[state]
                txt = STATE_TEXT[state]
                tcol = 'white' if state in ('oom',) else '#222222'
                hatch = '///' if state == 'cap' else None
            ax.add_patch(Rectangle((j, y), 1, 1, facecolor=face, edgecolor='white',
                                   lw=1.2, hatch=hatch))
            ax.text(j + 0.5, y + 0.5, txt, ha='center', va='center',
                    fontsize=8, color=tcol,
                    fontweight='bold' if state == 'ok' else 'normal')

    # group separators between BD-LRU / H-LRU / baselines
    for label, yline in _group_boundaries(rows, R):
        ax.axhline(yline, color='#222222', lw=1.8)

    ax.set_xlim(0, C)
    ax.set_ylim(0, R)
    ax.set_xticks([j + 0.5 for j in range(C)])
    ax.set_xticklabels([f'{int(T)}' for T in seqs])
    ax.set_yticks([R - 1 - i + 0.5 for i in range(R)])
    ax.set_yticklabels([lbl for lbl, _ in rows])
    ax.set_xlabel('sequence length  T', fontsize=10)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.015)
    cbar.set_label('median latency (ms/step, log)  \u2014  green = fast', fontsize=9)

    legend = [Patch(facecolor=STATE_COLOR['oom'], label='OOM'),
              Patch(facecolor=STATE_COLOR['cap'], hatch='///', label='hit ceiling'),
              Patch(facecolor=STATE_COLOR['err'], label='error'),
              Patch(facecolor=STATE_COLOR['miss'], label='not run')]
    ax.legend(handles=legend, loc='upper left', bbox_to_anchor=(1.14, 1.0),
              fontsize=8, frameon=False, title='status', title_fontsize=8)

    fig.suptitle(f'Per-step latency & feasibility by family  (batch B = {int(batch)}, '
                 f'{hero}) \u2014 {subtitle}', fontsize=12, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def _text_color(rgba) -> str:
    """Black on light cells, white on dark, by perceived luminance."""
    r, g, b = rgba[:3]
    return '#111111' if (0.299 * r + 0.587 * g + 0.114 * b) > 0.55 else 'white'


def _group_boundaries(rows, R) -> list[tuple[str, float]]:
    """Y positions to rule between BD-LRU, H-LRU and the baseline block."""
    fams = [lbl.split()[0] for lbl, _ in rows]
    lines = []
    for i in range(1, len(rows)):
        prev, cur = fams[i - 1], fams[i]
        prev_lru = prev in ('BD-LRU', 'H-LRU')
        cur_lru = cur in ('BD-LRU', 'H-LRU')
        if prev != cur and (prev_lru or cur_lru):
            lines.append((cur, R - i))
    return lines


def generate(csv_path: str, name: str, *, batch: float = 4.0,
             hero: str = DEFAULT_HERO, exclude: tuple[str, ...] = (),
             stem: str | None = None, figures_root: str = FIGURES_ROOT) -> list[str]:
    df = load(csv_path)
    fig = build(df, batch, name, hero=hero, exclude=exclude)
    if fig is None:
        return []
    stem = stem or f'family_grid_b{int(batch)}'
    return _save(fig, os.path.join(figures_root, name), stem)


def get_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--csv', required=True)
    p.add_argument('--name', required=True, help='sub-folder under figures/')
    p.add_argument('--batch', type=float, default=4.0)
    p.add_argument('--hero', default=DEFAULT_HERO)
    p.add_argument('--exclude', nargs='*', default=[],
                   help='family names to drop (e.g. lstm mamba2)')
    p.add_argument('--stem', default=None, help='output filename stem override')
    p.add_argument('--figures-root', default=FIGURES_ROOT)
    return p.parse_args()


def main():
    args = get_args()
    written = generate(args.csv, args.name, batch=args.batch, hero=args.hero,
                       exclude=tuple(args.exclude), stem=args.stem,
                       figures_root=args.figures_root)
    if not written:
        print('Nothing written (no rows at that batch size).')
    for path in written:
        print(f'Wrote {path}')


if __name__ == '__main__':
    main()
