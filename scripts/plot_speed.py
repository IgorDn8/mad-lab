"""Plot speed-benchmark results produced by speed_benchmark.py.

Reads one or more result CSVs and plots a chosen metric against the swept
variable (default: tokens/s vs seq_len), with one line per model, so you can
see how architectures scale.

Example:
    uv run python -m scripts.plot_speed --csv speed_results.csv \
        --x seq_len --y tokens_per_s --logx --logy --out speed_scaling.png
"""

import argparse

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def get_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--csv', nargs='+', required=True, help='one or more result CSV files')
    parser.add_argument('--x', type=str, default='seq_len', help='x-axis column')
    parser.add_argument('--y', type=str, default='tokens_per_s',
                        help='y-axis column (e.g. tokens_per_s, step_ms, peak_mem_mb)')
    parser.add_argument('--hue', type=str, default='model', help='column to color lines by')
    parser.add_argument('--logx', action='store_true', help='log-scale x-axis')
    parser.add_argument('--logy', action='store_true', help='log-scale y-axis')
    parser.add_argument('--out', type=str, default='speed_scaling.png', help='output image path')
    return parser.parse_args()


def main():
    args = get_args()
    df = pd.concat([pd.read_csv(c) for c in args.csv], ignore_index=True)

    # keep only successful runs for the plotted metric
    if 'status' in df.columns:
        df = df[df['status'] == 'ok']
    df = df.dropna(subset=[args.x, args.y])
    if df.empty:
        raise SystemExit('No successful rows to plot (check the CSV / status column).')

    sns.set_theme(style='whitegrid')
    plt.figure(figsize=(8, 5))
    ax = sns.lineplot(data=df, x=args.x, y=args.y, hue=args.hue, marker='o')
    if args.logx:
        ax.set_xscale('log', base=2)
    if args.logy:
        ax.set_yscale('log')
    ax.set_xlabel(args.x)
    ax.set_ylabel(args.y)
    ax.set_title(f'{args.y} vs {args.x}')
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f'Saved plot to {args.out}')


if __name__ == '__main__':
    main()
