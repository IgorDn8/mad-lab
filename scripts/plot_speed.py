"""Plot speed-benchmark results produced by speed_benchmark.py.

Reads one or more result CSVs and plots a chosen metric against the swept
variable (default: tokens/s vs seq_len), with one line per model, so you can
see how architectures scale.

Figures are organized under a top-level ``figures/`` directory, with one
sub-folder per benchmark so results don't pile up in the repo root::

    figures/<benchmark>/<metric>.png

Pass ``--name`` to choose the benchmark folder (otherwise it is derived from
the first CSV's filename, e.g. ``speed_results_wd4.csv`` -> ``wd4``).

Example:
    uv run python -m scripts.plot_speed --csv results/bdlru-wd4-all-impls/results.csv \
        --name bdlru-wd4-all-impls --y tokens_per_s --logx --logy
"""

import os
import argparse

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


# all figures live under this top-level directory (one sub-folder per benchmark)
FIGURES_ROOT = 'figures'


def get_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--csv', nargs='+', required=True, help='one or more result CSV files')
    parser.add_argument('--name', type=str, default=None,
                        help='benchmark name = sub-folder under figures/ '
                             '(default: derived from the first CSV filename)')
    parser.add_argument('--x', type=str, default='seq_len', help='x-axis column')
    parser.add_argument('--y', type=str, default='tokens_per_s',
                        help='y-axis column (e.g. tokens_per_s, step_ms, peak_mem_mb)')
    parser.add_argument('--hue', type=str, default='model', help='column to color lines by')
    parser.add_argument('--logx', action='store_true', help='log-scale x-axis')
    parser.add_argument('--logy', action='store_true', help='log-scale y-axis')
    parser.add_argument('--figures-root', type=str, default=FIGURES_ROOT,
                        help=f'root directory for all figures (default: {FIGURES_ROOT})')
    parser.add_argument('--out', type=str, default=None,
                        help='figure filename within the benchmark folder '
                             '(default: <y>.png)')
    return parser.parse_args()


def derive_benchmark_name(csv_paths) -> str:
    """Derive a benchmark folder name from the first CSV filename.

    ``speed_results_wd4.csv`` -> ``wd4``; ``speed_results.csv`` -> ``speed``.
    """
    stem = os.path.splitext(os.path.basename(csv_paths[0]))[0]
    stem = stem.replace('speed_results', '').strip('_-')
    return stem or 'benchmark'


def figure_path(name: str, y: str, figures_root: str = FIGURES_ROOT, out: str = None) -> str:
    """Compose ``<figures-root>/<name>/<out>`` and ensure the folder exists."""
    out_dir = os.path.join(figures_root, name)
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, out or f'{y}.png')


def render(df: pd.DataFrame, name: str, y: str, *, x: str = 'seq_len', hue: str = 'model',
           logx: bool = False, logy: bool = False, figures_root: str = FIGURES_ROOT,
           out: str = None) -> str:
    """Plot metric ``y`` vs ``x`` into ``figures/<name>/`` and return the path.

    Returns None (without raising) if there are no successful rows for ``y`` --
    e.g. a metric that isn't present, or a run that entirely OOM'd.
    """
    if 'status' in df.columns:
        df = df[df['status'] == 'ok']
    if y not in df.columns or x not in df.columns:
        return None
    df = df.dropna(subset=[x, y])
    if df.empty:
        return None

    out_path = figure_path(name, y, figures_root=figures_root, out=out)
    sns.set_theme(style='whitegrid')
    plt.figure(figsize=(8, 5))
    ax = sns.lineplot(data=df, x=x, y=y, hue=hue, marker='o')
    if logx:
        ax.set_xscale('log', base=2)
    if logy:
        ax.set_yscale('log')
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f'{name}: {y} vs {x}')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def main():
    args = get_args()
    df = pd.concat([pd.read_csv(c) for c in args.csv], ignore_index=True)
    name = args.name or derive_benchmark_name(args.csv)
    out_path = render(
        df, name, args.y, x=args.x, hue=args.hue,
        logx=args.logx, logy=args.logy, figures_root=args.figures_root, out=args.out,
    )
    if out_path is None:
        raise SystemExit('No successful rows to plot (check the CSV / status column).')
    print(f'Saved plot to {out_path}')


if __name__ == '__main__':
    main()
