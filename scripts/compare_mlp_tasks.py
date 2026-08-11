#!/usr/bin/env python3
"""Compare ``<layer> mlp`` sweeps across memorization / noisy recall / fuzzy recall.

Reads ``results.csv`` (and optional ``logs/metrics.csv``) under each sweep's
``runs/<name>/`` tree, aggregates seeds, and writes a models × tasks table
with cells formatted as ``best(mean±3std)`` (best = max across seeds).
The winning model in each task column is marked.

By default the per-seed score is the best epoch test accuracy from
``metrics.csv`` (``test/Accuracy_epoch``, dropping the trailing validate
passes). Fall back to final ``results.csv`` ``test_acc`` when metrics are
missing. Use ``--metric final`` to force the final checkpoint only.

Example:
  uv run python -m scripts.compare_mlp_tasks

  uv run python -m scripts.compare_mlp_tasks \\
      --mem logs_mem_mlp_iso1m \\
      --noisy logs_nr32_mlp_iso1m \\
      --fuzzy logs_fr32_iso1m \\
      --out figures/mlp-task-compare/comparison.csv

Also writes a booktabs LaTeX table (``comparison.tex``) with values
rounded to 2 decimal places (override with ``--tex-decimals``).
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

# Same family ordering as scripts/plot_task_parity.py
FAMILY_COLORS = [
    ('lstm', '#6c757d'),
    ('bdlru', '#0f75bc'),
    ('hlru', '#d1495b'),
    ('deltanet', '#e9a13b'),
    ('deltaprod2', '#c47a1a'),
    ('deltaprod4', '#8a4b12'),
    ('deltaprod', '#b5651d'),
]

DEFAULT_TASKS = (
    ('memorization', 'logs_mem_mlp_iso1m'),
    ('noisy-recall', 'logs_nr32_mlp_iso1m'),
    ('fuzzy-recall', 'logs_fr32_iso1m'),
)


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


def best_epoch_test(metrics_path: str) -> float | None:
    if not os.path.isfile(metrics_path):
        return None
    m = pd.read_csv(metrics_path)
    if 'test/Accuracy_epoch' not in m:
        return None
    # train.py ends with two trainer.validate() passes into this column
    per_epoch = m['test/Accuracy_epoch'].dropna().iloc[:-2]
    if not len(per_epoch):
        return None
    return float(per_epoch.max())


def load_task_runs(sweep_dir: str, metric: str) -> pd.DataFrame:
    pattern = os.path.join(sweep_dir, 'runs', '*', '*', 'results.csv')
    rows = []
    for res in sorted(glob.glob(pattern)):
        name = res.split(os.sep)[-3]
        seed_m = re.search(r'_s-(\d+)_', res.split(os.sep)[-2])
        final = pd.read_csv(res).iloc[-1]
        final_acc = float(final['test_acc'])
        metrics = os.path.join(os.path.dirname(res), 'logs', 'metrics.csv')
        best = best_epoch_test(metrics)

        if metric == 'best':
            score = best if best is not None else final_acc
            score_src = 'best_epoch' if best is not None else 'final'
        else:
            score = final_acc
            score_src = 'final'

        rows.append(dict(
            model=name,
            seed=int(seed_m.group(1)) if seed_m else -1,
            score=score,
            final_acc=final_acc,
            best_acc=best,
            score_src=score_src,
            params=int(final['model_size']),
        ))

    if not rows:
        return pd.DataFrame(columns=['model', 'seed', 'score', 'params'])
    return pd.DataFrame(rows)


def aggregate(runs: pd.DataFrame) -> pd.DataFrame:
    if runs.empty:
        return pd.DataFrame(
            columns=['model', 'best', 'mean', 'std', 'n', 'params'])
    g = runs.groupby('model', sort=False)
    out = g.agg(
        best=('score', 'max'),
        mean=('score', 'mean'),
        std=('score', 'std'),
        n=('score', 'count'),
        params=('params', 'first'),
    ).reset_index()
    out['std'] = out['std'].fillna(0.0)
    out = out.sort_values('model', key=lambda s: s.map(_family_key)).reset_index(drop=True)
    return out


def fmt_cell(best: float, mean: float, std: float, n: int, is_winner: bool,
             show_n: bool, decimals: int = 4) -> str:
    """Format as ``best(mean±3std)``."""
    if n == 0 or np.isnan(mean):
        return '—'
    three_std = 3.0 * std
    body = (f'{best:.{decimals}f}'
            f'({mean:.{decimals}f}±{three_std:.{decimals}f})')
    if show_n and n != 3:
        body += f' (n={n})'
    if is_winner:
        body = f'**{body}**'
    return body


def fmt_cell_tex(best: float, mean: float, std: float, n: int, is_winner: bool,
                 show_n: bool, decimals: int = 2) -> str:
    """Format as ``best(mean±3std)`` for LaTeX."""
    if n == 0 or np.isnan(mean):
        return '---'
    three_std = 3.0 * std
    body = (f'{best:.{decimals}f}'
            f'({mean:.{decimals}f}\\pm{three_std:.{decimals}f})')
    if show_n and n != 3:
        body += f'\\,(n={n})'
    if is_winner:
        body = f'\\textbf{{{body}}}'
    return body


def _collect_models(task_aggs: dict[str, pd.DataFrame]) -> list[str]:
    models: list[str] = []
    seen = set()
    for agg in task_aggs.values():
        for m in agg['model']:
            if m not in seen:
                seen.add(m)
                models.append(m)
    return sorted(models, key=_family_key)


def _best_models(task_aggs: dict[str, pd.DataFrame]) -> dict[str, str]:
    """Winner per task = highest across-seed best score (ties → mean)."""
    best_model: dict[str, str] = {}
    for task, agg in task_aggs.items():
        if agg.empty:
            continue
        top = agg.sort_values(['best', 'mean'], ascending=False).iloc[0]
        best_model[task] = top['model']
    return best_model


def build_table(
    task_aggs: dict[str, pd.DataFrame],
    *,
    show_n: bool,
    md_decimals: int = 4,
    include_params: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (display_df with formatted strings, numeric_df for CSV)."""
    models = _collect_models(task_aggs)
    winner = _best_models(task_aggs)

    display_rows = []
    numeric_rows = []
    for model in models:
        drow = {'model': model}
        nrow = {'model': model}
        for task, agg in task_aggs.items():
            hit = agg[agg['model'] == model]
            if hit.empty:
                best = mean = std = np.nan
                n = 0
            else:
                r = hit.iloc[0]
                best, mean, std, n = (
                    float(r['best']), float(r['mean']), float(r['std']), int(r['n']))
            is_winner = winner.get(task) == model
            drow[task] = fmt_cell(
                best, mean, std, n, is_winner, show_n, md_decimals)
            nrow[f'{task}_best_seed'] = best
            nrow[f'{task}_mean'] = mean
            nrow[f'{task}_std'] = std
            nrow[f'{task}_3std'] = 3.0 * std if n else np.nan
            nrow[f'{task}_n'] = n
            nrow[f'{task}_winner'] = bool(is_winner)
        if include_params:
            # Prefer vs=32 sweeps (fuzzy/noisy); mem uses vs=4096 so embeddings differ.
            params_val = np.nan
            for task in ('fuzzy-recall', 'noisy-recall', 'memorization'):
                agg = task_aggs.get(task)
                if agg is None or agg.empty:
                    continue
                hit = agg[agg['model'] == model]
                if not hit.empty:
                    params_val = int(hit.iloc[0]['params'])
                    break
            drow['params'] = f'{params_val:,}' if params_val == params_val else '—'
            nrow['params'] = params_val
        display_rows.append(drow)
        numeric_rows.append(nrow)

    return pd.DataFrame(display_rows), pd.DataFrame(numeric_rows)


def to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    header = '| ' + ' | '.join(cols) + ' |'
    sep = '| ' + ' | '.join('---' for _ in cols) + ' |'
    lines = [header, sep]
    for _, row in df.iterrows():
        lines.append('| ' + ' | '.join(str(row[c]) for c in cols) + ' |')
    return '\n'.join(lines)


TASK_TEX_HEADERS = {
    'memorization': 'Mem.',
    'noisy-recall': 'Noisy recall',
    'fuzzy-recall': 'Fuzzy recall',
}


def to_latex(
    task_aggs: dict[str, pd.DataFrame],
    *,
    show_n: bool,
    decimals: int = 2,
    metric: str,
) -> str:
    """Booktabs table with cells ``best(mean±3std)`` rounded to ``decimals``."""
    models = _collect_models(task_aggs)
    winner = _best_models(task_aggs)
    tasks = list(task_aggs.keys())
    headers = ['Model'] + [TASK_TEX_HEADERS.get(t, t) for t in tasks] + ['Params']
    colspec = 'l' + 'c' * len(tasks) + 'r'

    lines = [
        '% Auto-generated by scripts/compare_mlp_tasks.py',
        f'% metric={metric}; format best(mean±3std); '
        f'rounded to {decimals} decimal places',
        r'\begin{table}[t]',
        r'\centering',
        r'\caption{Test accuracy as '
        r'\texttt{best(mean}$\pm$\texttt{3std)} across seeds for '
        r'\texttt{<layer> mlp} on memorization, noisy recall, and fuzzy recall. '
        r'Best model per column in bold. '
        r'Params from the vs$=$32 (fuzzy/noisy) runs.}',
        r'\label{tab:mlp-task-compare}',
        rf'\begin{{tabular}}{{{colspec}}}',
        r'\toprule',
        ' & '.join(headers) + r' \\',
        r'\midrule',
    ]

    for model in models:
        cells = [model.replace('_', r'\_')]
        for task, agg in task_aggs.items():
            hit = agg[agg['model'] == model]
            if hit.empty:
                best = mean = std = np.nan
                n = 0
            else:
                r = hit.iloc[0]
                best, mean, std, n = (
                    float(r['best']), float(r['mean']), float(r['std']), int(r['n']))
            cells.append(fmt_cell_tex(
                best, mean, std, n, winner.get(task) == model, show_n, decimals))
        params_val = np.nan
        for task in ('fuzzy-recall', 'noisy-recall', 'memorization'):
            agg = task_aggs.get(task)
            if agg is None or agg.empty:
                continue
            hit = agg[agg['model'] == model]
            if not hit.empty:
                params_val = int(hit.iloc[0]['params'])
                break
        if params_val == params_val:
            cells.append(f'{params_val / 1e6:.{decimals}f}M')
        else:
            cells.append('---')
        lines.append(' & '.join(cells) + r' \\')

    lines += [
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
        '',
    ]
    return '\n'.join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mem', default=DEFAULT_TASKS[0][1],
                   help='memorization <layer> mlp sweep dir')
    p.add_argument('--noisy', default=DEFAULT_TASKS[1][1],
                   help='noisy-recall <layer> mlp sweep dir')
    p.add_argument('--fuzzy', default=DEFAULT_TASKS[2][1],
                   help='fuzzy-recall <layer> mlp sweep dir')
    p.add_argument('--metric', choices=('best', 'final'), default='best',
                   help='per-seed score: best epoch test (default) or final test_acc')
    p.add_argument('--out', default='figures/mlp-task-compare/comparison.csv',
                   help='numeric CSV path (markdown/.tex written beside it)')
    p.add_argument('--tex-decimals', type=int, default=2,
                   help='decimal places in the LaTeX table (default: 2)')
    p.add_argument('--show-n', action='store_true',
                   help='annotate cells whose seed count is not 3')
    args = p.parse_args()

    tasks = (
        ('memorization', args.mem),
        ('noisy-recall', args.noisy),
        ('fuzzy-recall', args.fuzzy),
    )

    task_aggs: dict[str, pd.DataFrame] = {}
    for label, sweep in tasks:
        if not os.path.isdir(sweep):
            print(f'WARN: missing sweep dir {sweep!r} — column will be empty',
                  file=sys.stderr)
            task_aggs[label] = pd.DataFrame(
                columns=['model', 'best', 'mean', 'std', 'n', 'params'])
            continue
        runs = load_task_runs(sweep, args.metric)
        agg = aggregate(runs)
        task_aggs[label] = agg
        n_models = len(agg)
        n_runs = int(agg['n'].sum()) if n_models else 0
        print(f'{label:16s}  {sweep}  models={n_models} runs={n_runs}  '
              f'metric={args.metric}')

    # auto show_n if any task has incomplete seeds
    show_n = args.show_n or any(
        (not agg.empty) and (agg['n'] != agg['n'].max()).any()
        for agg in task_aggs.values()
    ) or any(
        (not agg.empty) and int(agg['n'].max()) < 3
        for agg in task_aggs.values()
    )

    display, numeric = build_table(task_aggs, show_n=show_n)
    display_round, _ = build_table(
        task_aggs, show_n=show_n, md_decimals=args.tex_decimals,
        include_params=False,
    )
    md = to_markdown(display)
    md_round = to_markdown(display_round)
    tex = to_latex(task_aggs, show_n=show_n, decimals=args.tex_decimals,
                   metric=args.metric)

    print()
    print(f'Test accuracy as best(mean±3std)  (metric={args.metric}; '
          f'winner per column in **bold**)')
    print(md)
    print()
    print(f'Rounded ({args.tex_decimals} d.p., no params):')
    print(md_round)
    print()
    print(f'LaTeX ({args.tex_decimals} d.p.):')
    print(tex)

    out_csv = args.out
    out_dir = os.path.dirname(out_csv) or '.'
    os.makedirs(out_dir, exist_ok=True)
    numeric.to_csv(out_csv, index=False)
    out_md = os.path.splitext(out_csv)[0] + '.md'
    out_tex = os.path.splitext(out_csv)[0] + '.tex'
    with open(out_md, 'w') as f:
        f.write(f'# `<layer> mlp` task comparison\n\n')
        f.write(f'Per-seed score: `{args.metric}` '
                f'({"best epoch test/Accuracy_epoch" if args.metric == "best" else "final results.csv test_acc"}).\n\n')
        f.write('Cell format: `best(mean±3std)` where `best` is the max across '
                'seeds and `3std` is three times the across-seed standard '
                'deviation.\n\n')
        f.write('## Full precision\n\n')
        f.write(md + '\n\n')
        f.write(f'## Rounded ({args.tex_decimals} d.p.)\n\n')
        f.write(md_round + '\n\n')
        f.write('Sources:\n')
        for label, sweep in tasks:
            f.write(f'- **{label}**: `{sweep}`\n')
    with open(out_tex, 'w') as f:
        f.write(tex)
    print(f'wrote {out_csv}')
    print(f'wrote {out_md}')
    print(f'wrote {out_tex}')


if __name__ == '__main__':
    main()
