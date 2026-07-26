#!/usr/bin/env python3
"""Publication-quality scan-implementation comparisons for MAD speed benchmarks.

This is the mad-lab analogue of nisys-bench's ``plot_persistent_vs_parallel`` /
``plot_seq_vs_parallel``: instead of comparing framework backends it compares the
scan *implementations* wired into the BD-LRU / H-LRU layers
(``affine_scan_torch_impl``, ``hopscan_custom``, ``triton_sequential``,
``triton_persistent``, ``triton_parallel_blelloch``, ...), as measured by
``speed_benchmark.py``.

It reads one or more result CSVs and renders, into ``figures/<name>/``:

  * ``comparison.png`` / ``.pdf`` -- a 2x2 "hero vs baselines" figure at a single
    block size:
      (a) throughput vs sequence length            (log-log)
      (b) speedup of the hero impl over each baseline vs T (shaded faster/slower)
      (c) latency vs sequence length + slope-1 O(T) guide  (log-log)
      (d) average wall-clock time per impl          (mean over T, horizontal bars)

  * ``block_comparison.png`` / ``.pdf`` -- only when the CSVs span multiple block
    sizes (e.g. wd1 + wd4): throughput-by-block grouped bars at a fixed T, plus
    the hero's median speedup over each baseline vs block size.

Examples:
    uv run python -m scripts.plot_compare \
        --csv results/bdlru-wd4-all-impls/results.csv --name bdlru-wd4-all-impls

    # combined block-size comparison across two runs
    uv run python -m scripts.plot_compare --name bdlru-block-compare \
        --csv results/bdlru-wd1-all-impls/results.csv \
              results/bdlru-wd4-all-impls/results.csv
"""

from __future__ import annotations

import os
import re
import argparse
from statistics import median

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


FIGURES_ROOT = 'figures'

# canonical hero + the order implementations are drawn/listed in
DEFAULT_HERO = 'triton_persistent'
IMPL_ORDER = [
    'triton_fused_gates', 'triton_auto_v2_compile', 'triton_auto_v2',
    'triton_auto', 'triton_chunked',
    'triton_persistent', 'triton_sequential', 'triton_parallel_blelloch',
    'hopscan_custom', 'custom_hopscan_autotune', 'affine_scan_torch_impl',
    # legacy / misc names still found in older CSVs
    'hopscan_opt', 'affine_scan', 'orig',
    # cross-model comparison series (whole layers, not scan-impl overrides)
    'associative_scan', 'dneto', 'dproducto', 'mamba2',
]

PRETTY = {
    'triton_fused_gates': 'Triton fused gates',
    'triton_auto_v2': 'Triton auto v2',
    'triton_auto_v2_compile': 'Triton auto v2 (compiled)',
    'triton_auto': 'Triton auto',
    'triton_chunked': 'Triton chunked',
    'triton_persistent': 'Triton persistent',
    'triton_sequential': 'Triton sequential',
    'triton_parallel_blelloch': 'Triton Blelloch',
    'hopscan_custom': 'Hopscan (custom)',
    'custom_hopscan_autotune': 'Hopscan (compiled)',
    'affine_scan_torch_impl': 'Torch assoc. scan',
    'hopscan_opt': 'Hopscan (opt)',
    'affine_scan': 'Torch assoc. scan',
    'orig': 'Sequential (orig)',
    'associative_scan': 'PDSSM (assoc. scan)',
    'dneto': 'DeltaNet',
    'dproducto': 'DeltaProduct',
    'mamba2': 'Mamba2',
}

# hero gets the bold red "hero" style; baselines get distinct colors/markers.
STYLE = {
    'triton_fused_gates': dict(color='#03071e', marker='*', lw=2.8, z=7),
    'triton_auto_v2': dict(color='#f48c06', marker='<', lw=2.0, z=5),
    'triton_auto_v2_compile': dict(color='#dc2f02', marker='>', lw=2.6, z=6),
    'triton_auto': dict(color='#c1121f', marker='*', lw=2.8, z=6),
    'triton_chunked': dict(color='#7b2cbf', marker='h', lw=2.0, z=4),
    'triton_persistent': dict(color='#d1495b', marker='o', lw=2.6, z=5),
    'triton_sequential': dict(color='#e9a13b', marker='D', lw=1.9, z=4),
    'triton_parallel_blelloch': dict(color='#8c2d8f', marker='v', lw=1.8, z=3),
    'hopscan_custom': dict(color='#2a9d8f', marker='^', lw=1.9, z=3),
    'custom_hopscan_autotune': dict(color='#117a65', marker='P', lw=2.0, z=4),
    'affine_scan_torch_impl': dict(color='#1f77b4', marker='s', lw=1.8, z=3),
    'hopscan_opt': dict(color='#2a9d8f', marker='^', lw=1.9, z=3),
    'affine_scan': dict(color='#1f77b4', marker='s', lw=1.8, z=3),
    'orig': dict(color='#6c757d', marker='x', lw=1.6, z=2),
    'associative_scan': dict(color='#d1495b', marker='o', lw=2.2, z=4),
    'dneto': dict(color='#1f77b4', marker='s', lw=2.0, z=3),
    'dproducto': dict(color='#e9a13b', marker='D', lw=2.0, z=3),
    'mamba2': dict(color='#2a9d8f', marker='^', lw=2.0, z=3),
}
_FALLBACK_STYLE = dict(color='#333333', marker='.', lw=1.6, z=2)

_IMPL_RE = re.compile(r'implementation=([^\]]+)')
_BLOCK_RE = re.compile(r'wd(\d+)')


def _impl_of(row) -> str:
    """Implementation name from the `overrides` field, else the model shorthand."""
    ov = row.get('overrides')
    if isinstance(ov, str):
        m = _IMPL_RE.search(ov)
        if m:
            return m.group(1)
    return str(row.get('model', 'unknown'))


_WD_OVERRIDE_RE = re.compile(r'window_dim=(\d+)')


def _block_of(row) -> int | None:
    """Block size m: a ``window_dim`` column/override wins, else the model name.

    Precedence: an explicit ``window_dim`` column (block-size *sweep*), then a
    ``--layer-overrides window_dim=N`` in the overrides string, then the registry
    model shorthand (e.g. ``...wd4...``).
    """
    wd = row.get('window_dim')
    if wd is not None and pd.notna(wd):
        try:
            return int(wd)
        except (TypeError, ValueError):
            pass
    ov = row.get('overrides')
    if isinstance(ov, str):
        m = _WD_OVERRIDE_RE.search(ov)
        if m:
            return int(m.group(1))
    m = _BLOCK_RE.search(str(row.get('model', '')))
    return int(m.group(1)) if m else None


def _pretty(impl: str) -> str:
    return PRETTY.get(impl, impl)


def _style(impl: str) -> dict:
    return STYLE.get(impl, _FALLBACK_STYLE)


def load(csv_paths: list[str]) -> pd.DataFrame:
    df = pd.concat([pd.read_csv(c) for c in csv_paths], ignore_index=True)
    if 'status' in df.columns:
        df = df[df['status'] == 'ok'].copy()
    if df.empty:
        raise SystemExit('No successful rows to plot (check the CSV / status column).')
    df['impl'] = df.apply(_impl_of, axis=1)
    df['block'] = df.apply(_block_of, axis=1)
    df['tps_m'] = df['tokens_per_s'] / 1e6
    return df


def _impls_present(df: pd.DataFrame) -> list[str]:
    present = list(df['impl'].unique())
    ordered = [i for i in IMPL_ORDER if i in present]
    ordered += [i for i in present if i not in ordered]
    return ordered


def _series(df: pd.DataFrame, impl: str, x: str, y: str) -> tuple[list, list]:
    sub = df[df['impl'] == impl].dropna(subset=[x, y]).sort_values(x)
    sub = sub.groupby(x, as_index=False)[y].median()
    return list(sub[x]), list(sub[y])


# --------------------------------------------------------------------------- #
# 2x2 hero-vs-baselines scaling figure (single block size)
# --------------------------------------------------------------------------- #
def _panel_throughput(ax, df, impls, x):
    for impl in impls:
        xs, ys = _series(df, impl, x, 'tps_m')
        if xs:
            s = _style(impl)
            ax.plot(xs, ys, marker=s['marker'], color=s['color'], lw=s['lw'],
                    label=_pretty(impl), zorder=s['z'])
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlabel(x)
    ax.set_ylabel('Throughput (M tokens/s)')
    ax.set_title('(a) Throughput scaling', fontsize=10)
    ax.grid(True, which='both', alpha=0.25)
    ax.legend(fontsize=8, framealpha=0.9)


def _panel_speedup(ax, df, hero, baselines, x):
    hx, hy = _series(df, hero, x, 'tps_m')
    hero_map = dict(zip(hx, hy))
    drew = False
    for impl in baselines:
        bx, by = _series(df, impl, x, 'tps_m')
        xs, ys = [], []
        for xv, bv in zip(bx, by):
            if xv in hero_map and bv:
                xs.append(xv)
                ys.append(hero_map[xv] / bv)
        if xs:
            s = _style(impl)
            ax.plot(xs, ys, marker=s['marker'], color=s['color'], lw=s['lw'],
                    label=f'vs {_pretty(impl)}', zorder=s['z'])
            drew = True
    ax.axhline(1.0, color='black', lw=1.0, ls='--', zorder=1)
    if drew:
        top = ax.get_ylim()[1]
        ax.axhspan(1.0, top, color='#2a9d8f', alpha=0.06, zorder=0)
    ax.text(0.02, 0.96, f'{_pretty(hero)} faster \u2191', transform=ax.transAxes,
            fontsize=8, va='top', color='#2a9d8f')
    ax.text(0.02, 0.04, f'{_pretty(hero)} slower \u2193', transform=ax.transAxes,
            fontsize=8, va='bottom', color='#d1495b')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlabel(x)
    ax.set_ylabel('Speedup (\u00d7)')
    ax.set_title(f'(b) {_pretty(hero)} speedup over baselines', fontsize=10)
    ax.grid(True, which='both', alpha=0.25)
    if drew:
        ax.legend(fontsize=8, framealpha=0.9)


def _panel_latency(ax, df, impls, x):
    anchor = None
    for impl in impls:
        xs, ys = _series(df, impl, x, 'step_ms')
        if xs:
            s = _style(impl)
            ax.plot(xs, ys, marker=s['marker'], color=s['color'], lw=s['lw'],
                    label=_pretty(impl), zorder=s['z'])
            if anchor is None:
                anchor = (xs[0], ys[0])
    if anchor is not None:
        xs_all = sorted(df.dropna(subset=[x, 'step_ms'])[x].unique())
        x0, y0 = anchor
        guide = [y0 * (xv / x0) for xv in xs_all]
        ax.plot(xs_all, guide, color='gray', ls=(0, (2, 2)), lw=1.0,
                label='linear O(T) (slope 1)', zorder=1)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlabel(x)
    ax.set_ylabel('Latency / step (ms)')
    ax.set_title('(c) Latency vs length (slope 1 = linear time)', fontsize=10)
    ax.grid(True, which='both', alpha=0.25)
    ax.legend(fontsize=8, framealpha=0.9)


def _panel_avg_latency(ax, df, impls, x):
    labels, values, colors = [], [], []
    for impl in impls:
        _, ys = _series(df, impl, x, 'step_ms')
        if ys:
            labels.append(_pretty(impl))
            values.append(sum(ys) / len(ys))
            colors.append(_style(impl)['color'])
    y = range(len(labels))
    bars = ax.barh(list(y), values, color=colors, edgecolor='white', linewidth=0.4)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel('Average latency / step (ms)')
    ax.set_title('(d) Average wall-clock time (mean over lengths)', fontsize=10)
    ax.grid(True, axis='x', alpha=0.25)
    xmax = max(values) if values else 1.0
    ax.set_xlim(0, xmax * 1.2)
    for bar, v in zip(bars, values):
        ax.text(bar.get_width() + xmax * 0.01, bar.get_y() + bar.get_height() / 2,
                f'{v:.1f} ms', va='center', ha='left', fontsize=8)


def _headline(df, hero, baselines, x) -> str:
    hx, hy = _series(df, hero, x, 'tps_m')
    hero_map = dict(zip(hx, hy))
    best, ctx = 0.0, ''
    for impl in baselines:
        bx, by = _series(df, impl, x, 'tps_m')
        for xv, bv in zip(bx, by):
            if xv in hero_map and bv and hero_map[xv] / bv > best:
                best = hero_map[xv] / bv
                ctx = f'{_pretty(impl)} at {x}={xv}'
    return f'up to {best:.1f}\u00d7 faster than {ctx}' if best else ''


def build_scaling_figure(df, hero, x, subtitle) -> plt.Figure:
    plt.rcParams.update({
        'font.size': 10, 'axes.titlesize': 10,
        'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 120,
    })
    impls = _impls_present(df)
    if hero not in impls:
        hero = impls[0]
    baselines = [i for i in impls if i != hero]

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))
    _panel_throughput(axes[0, 0], df, impls, x)
    _panel_speedup(axes[0, 1], df, hero, baselines, x)
    _panel_latency(axes[1, 0], df, impls, x)
    _panel_avg_latency(axes[1, 1], df, impls, x)

    fig.suptitle(f'Scan implementation comparison \u2014 {subtitle}',
                 fontsize=13, fontweight='bold', y=0.985)
    head = _headline(df, hero, baselines, x)
    if head:
        fig.text(0.5, 0.945, head, ha='center', fontsize=10, color='#d1495b')
    fig.tight_layout(rect=(0, 0.02, 1, 0.925))
    return fig


# --------------------------------------------------------------------------- #
# block-size comparison figure (multiple block sizes)
# --------------------------------------------------------------------------- #
def _pick_seq_len(df, x, preferred):
    seqs = sorted(df.dropna(subset=[x])[x].unique())
    if preferred in seqs:
        return preferred
    return seqs[len(seqs) // 2] if seqs else None


def build_block_figure(df, hero, x, at_seq_len, subtitle) -> plt.Figure | None:
    blocks = sorted(b for b in df['block'].dropna().unique())
    if len(blocks) < 2:
        return None
    impls = _impls_present(df)
    if hero not in impls:
        hero = impls[0]
    baselines = [i for i in impls if i != hero]

    plt.rcParams.update({
        'font.size': 10, 'axes.titlesize': 10,
        'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 120,
    })
    fig, (ax_bar, ax_speed) = plt.subplots(1, 2, figsize=(13.5, 5.2))

    # (a) grouped bars: throughput per impl across block sizes at fixed T
    import numpy as np
    at_df = df[df[x] == at_seq_len]
    n = len(impls)
    width = 0.8 / max(n, 1)
    xb = np.arange(len(blocks))
    for i, impl in enumerate(impls):
        heights = []
        for b in blocks:
            sub = at_df[(at_df['impl'] == impl) & (at_df['block'] == b)]
            heights.append(sub['tps_m'].median() if not sub.empty else None)
        offs = xb + (i - (n - 1) / 2) * width
        xs = [o for o, h in zip(offs, heights) if h is not None]
        ys = [h for h in heights if h is not None]
        ax_bar.bar(xs, ys, width=width, color=_style(impl)['color'],
                   edgecolor='white', linewidth=0.4, label=_pretty(impl),
                   zorder=_style(impl)['z'])
    ax_bar.set_yscale('log')
    ax_bar.set_xticks(xb)
    ax_bar.set_xticklabels([f'{b}\u00d7{b}' for b in blocks])
    ax_bar.set_xlabel('Block size (m\u00d7m)')
    ax_bar.set_ylabel('Throughput (M tokens/s)')
    ax_bar.set_title(f'(a) Throughput by block size ({x}={at_seq_len})', fontsize=10)
    ax_bar.grid(True, axis='y', which='both', alpha=0.25)
    ax_bar.legend(fontsize=8, framealpha=0.9)

    # (b) hero median speedup over each baseline vs block size (median over T)
    for impl in baselines:
        xs, ys = [], []
        for b in blocks:
            bl = df[df['block'] == b]
            ratios = []
            for xv in sorted(bl[x].dropna().unique()):
                h = bl[(bl['impl'] == hero) & (bl[x] == xv)]['tps_m']
                o = bl[(bl['impl'] == impl) & (bl[x] == xv)]['tps_m']
                if not h.empty and not o.empty and o.median() > 0:
                    ratios.append(h.median() / o.median())
            if ratios:
                xs.append(b)
                ys.append(median(ratios))
        if xs:
            s = _style(impl)
            ax_speed.plot(xs, ys, marker=s['marker'], color=s['color'], lw=s['lw'],
                          label=f'vs {_pretty(impl)}', zorder=s['z'])
    ax_speed.axhline(1.0, color='black', lw=1.0, ls='--', zorder=1)
    ax_speed.set_xscale('log', base=2)
    ax_speed.set_yscale('log')
    ax_speed.xaxis.set_major_formatter(ScalarFormatter())
    ax_speed.yaxis.set_major_formatter(ScalarFormatter())
    ax_speed.set_xticks(blocks)
    ax_speed.set_xticklabels([str(b) for b in blocks])
    ax_speed.set_xlabel('Block size m')
    ax_speed.set_ylabel('Speedup (\u00d7)')
    ax_speed.set_title(f'(b) {_pretty(hero)} speedup vs block size (median over {x})', fontsize=10)
    ax_speed.grid(True, which='both', alpha=0.25)
    ax_speed.legend(fontsize=8, framealpha=0.9)

    fig.suptitle(f'Scan implementation comparison by block size \u2014 {subtitle}',
                 fontsize=13, fontweight='bold', y=0.98)
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    return fig


def _save(fig, out_dir, stem) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    png = os.path.join(out_dir, f'{stem}.png')
    pdf = os.path.join(out_dir, f'{stem}.pdf')
    fig.savefig(png, dpi=200, bbox_inches='tight')
    fig.savefig(pdf, bbox_inches='tight')
    plt.close(fig)
    return [png, pdf]


def generate(csv_paths: list[str], name: str, *, hero: str = DEFAULT_HERO,
             x: str = 'seq_len', at_seq_len: int = 32768,
             figures_root: str = FIGURES_ROOT, subtitle: str = None,
             min_impls: int = 2) -> list[str]:
    """Render the comparison (and block-comparison) figures for a benchmark.

    Returns the list of written paths. Skips silently (returns []) when fewer
    than ``min_impls`` implementations are present -- a single-implementation
    "comparison" carries no information the simple line plots don't already show.
    """
    df = load(csv_paths)
    if df['impl'].nunique() < min_impls:
        return []
    out_dir = os.path.join(figures_root, name)
    subtitle = subtitle or ' / '.join(sorted(df['layers'].dropna().unique())) or name

    written: list[str] = []
    written += _save(build_scaling_figure(df, hero, x, subtitle), out_dir, 'comparison')

    at = _pick_seq_len(df, x, at_seq_len)
    block_fig = build_block_figure(df, hero, x, at, subtitle) if at is not None else None
    if block_fig is not None:
        written += _save(block_fig, out_dir, 'block_comparison')
    return written


def get_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--csv', nargs='+', required=True, help='one or more result CSV files')
    p.add_argument('--name', type=str, required=True, help='benchmark name = sub-folder under figures/')
    p.add_argument('--hero', type=str, default=DEFAULT_HERO,
                   help=f'implementation to feature as the hero (default: {DEFAULT_HERO})')
    p.add_argument('--x', type=str, default='seq_len', help='swept x-axis column (default: seq_len)')
    p.add_argument('--at-seq-len', type=int, default=32768,
                   help='fixed length for the block-size bar panel (default: 32768)')
    p.add_argument('--figures-root', type=str, default=FIGURES_ROOT)
    p.add_argument('--subtitle', type=str, default=None,
                   help='override the figure subtitle (default: derived from the model)')
    return p.parse_args()


def main():
    args = get_args()
    written = generate(
        args.csv, args.name, hero=args.hero, x=args.x, at_seq_len=args.at_seq_len,
        figures_root=args.figures_root, subtitle=args.subtitle, min_impls=1,
    )
    if not written:
        print('Nothing written (no successful rows).')
    for path in written:
        print(f'Wrote {path}')
    if not any(p.endswith('block_comparison.png') for p in written):
        print('(single block size -> skipping block_comparison figure)')


if __name__ == '__main__':
    main()
