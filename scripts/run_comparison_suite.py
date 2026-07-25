#!/usr/bin/env python3
"""Comprehensive BD-LRU scan-implementation comparison suite.

Runs ``speed_benchmark.py`` for every scan implementation across regimes that
stress different parts of the design space, then emits the publication-quality
comparison / crossover figures for each. H-LRU is intentionally out of scope for
now (needs refinement).

Implementations compared (BD-LRU selective layer):
    orig (sequential loop, reference baseline),
    affine_scan_torch_impl, hopscan_custom,
    triton_sequential, triton_persistent, triton_parallel_blelloch

Regimes (each -> results/<regime>/results.csv + figures/<regime>/*), mapped to
the questions the benchmark must answer:

  seqlen-m1   [note 1] Length scaling at a scalar block (m=1). Batch 32.
  seqlen-m4   [note 1] Length scaling at a matrix block (m=4). Batch 8.
  blocksize   [note 4 + 6] 2-D T x window_dim in {1,2,4,8,16,32,64}. Feeds the
              block_comparison AND the parallel-vs-sequential crossover map.
  batch       [note 3 + 6] 2-D T x batch in {1,8,32,128}, m=4.
  hidden      [note 5 + 6] 2-D T x hidden_dim in {128,256,512,1024}, m=4.
  inference   Forward-only length scaling at m=4.
  model-compare  Cross-model (PDSSM/DeltaNet/DeltaProduct/Mamba2) length scaling.

Iso-tier sweeps (whole registered iso layers over a full batch x length grid):
  iso-d128-iso1m     iso-param ~1M   @ d=128
  iso-d1024-iso100m  iso-param ~100M @ d=1024
  iso-d128-s1024     iso-state 1024  @ d=128
  iso-d1024-s4096    iso-state 4096  @ d=1024
  Grid: batch in {1,2,8,16,64,128} x T in {2^8,2^10,2^12,2^14,2^16,2^18}. Every
  family runs (LSTM, BD-LRU/H-LRU at m in {1,2,4,8,16}, PDSSM, Mamba2, DeltaNet,
  DeltaProduct at 2/4/8 Householders). BD-LRU/H-LRU run 3 scan impls: orig
  (Python-loop reference, capped at T <= 2^12), hopscan_custom (eager parallel),
  and triton_auto (occupancy-aware Triton scan). All families run in bf16. OOM
  cells are recorded (no auto batch-reduction) to map the feasibility frontier.

Resumable: every invocation is tagged (impl + grid value) and each produced row
records that ``run_tag`` and its ``seq_len``. On re-run we skip (run_tag, seq_len)
cells already present -- so adding a NEW implementation, block/hidden/batch value,
or even a new sequence length only runs the missing cells. Use --fresh to wipe a
regime and recompute from scratch. OOM/error cells count as done (not retried).

Correctness / precision (Q1/Q7) live in scripts/verify_scans.py, not here.

Usage:
    uv run python -m scripts.run_comparison_suite --fresh          # clean full run
    uv run python -m scripts.run_comparison_suite                  # resume / extend
    uv run python -m scripts.run_comparison_suite --regime blocksize --impls triton_persistent
    uv run python -m scripts.run_comparison_suite --dry-run
"""

from __future__ import annotations

import os
import sys
import json
import shutil
import argparse

BENCH_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BENCH_DIR not in sys.path:
    sys.path.insert(0, BENCH_DIR)


# reference baseline first, then the parallel torch scans, then the triton kernels
ALL_IMPLS = [
    'orig',
    'affine_scan_torch_impl',
    'hopscan_custom',
    'custom_hopscan_autotune',
    'triton_parallel_blelloch',
    'triton_sequential',
    'triton_persistent',
    'triton_auto',
]

# `orig` is an O(T) Python loop; cap its sequence length so a single point stays
# tractable (it exists only to anchor the sequential baseline, not to scale).
ORIG_MAX_SEQ = 2048

# reduced timing budget keeps the heavy (large-block / long-T) cells feasible;
# latency is still a median over `repeats` groups.
TIMING = dict(warmup=3, iters=10, repeats=3)

_POW2 = [2 ** e for e in range(8, 17)]          # 256 .. 65536
_SEQ_2D = [512, 2048, 8192, 32768]              # T grid for the cheaper 2-D regimes
_SEQ_2D_HEAVY = [512, 2048, 8192]               # shorter T grid for the block regime

# ---- iso benchmark grid (whole registered iso-tier layers) ---------------
# Full 2-D grid requested for the iso sweeps: batch x sequence length.
_ISO_BATCH = [1, 2, 8, 16, 64, 128]
_ISO_SEQ = [2 ** e for e in (8, 10, 12, 14, 16, 18)]        # 256 .. 262144
_ISO_NUM_SAMPLES = 128                                       # >= max(_ISO_BATCH)

# Precision for the iso sweeps. Applied via torch.autocast (not a model cast), so
# complex params (PDSSM) and kernel internals are preserved. bf16 for ALL families
# gives an apples-to-apples throughput comparison (fla baselines require bf16 anyway);
# set to '32' to run the fork recurrences in fp32 instead.
_ISO_PRECISION = 'bf16'
# BD-LRU/H-LRU scan implementations: layer `implementation` name -> run_tag label.
# The default set and the --iso-lru-impls override both draw from this map, so adding
# the compiled hopscan later is just `--iso-lru-impls hopscan_custom triton_auto
# custom_hopscan_autotune` (no code edit).
_ISO_LRU_IMPL_LABELS = {
    'orig': 'orig',                              # sequential Python-loop reference
    'hopscan_custom': 'hopscan',                 # eager parallel hopscan
    'triton_auto': 'triton_auto',                # occupancy-aware Triton scan
    'custom_hopscan_autotune': 'compile',        # torch.compile(max-autotune) hopscan
}
_ISO_LRU_IMPLS_DEFAULT = ['orig', 'hopscan_custom', 'triton_auto']
# per-impl sequence-length caps (impls whose cost is ~linear in T). `orig` is a slow
# Python loop -> cut it hardest so its long-T cells don't dominate wall-clock.
_ISO_SEQ_CAPS = {'orig': 2 ** 12}
_ISO_BLOCKS = (1, 2, 4, 8, 16)
_ISO_DPROD_RANKS = (2, 4, 8)


def _iso_entries(dim: int, suffix: str, lru_impls: list[str] | None = None) -> list[dict]:
    """Model entries for one iso tier (suffix = iso-param tag or iso-state `s{S}`).

    BD-LRU / H-LRU are emitted at every block m in {1,2,4,8,16} for each scan impl in
    ``lru_impls`` (default: orig, hopscan_custom, triton_auto). DeltaProduct is emitted
    at 2/4/8 Householders as independent models.
    """
    lru_impls = lru_impls or _ISO_LRU_IMPLS_DEFAULT
    e: list[dict] = []
    e.append(dict(tag='lstm', layers=[f'lstm-d{dim}-{suffix}'],
                  precision=_ISO_PRECISION, sweep=_ISO_SEQ))
    for fam in ('bdlru', 'hlru'):
        for m in _ISO_BLOCKS:
            base = f'{fam}-sel-wd{m}-d{dim}-{suffix}'
            for impl in lru_impls:
                label = _ISO_LRU_IMPL_LABELS[impl]
                cap = _ISO_SEQ_CAPS.get(impl)
                e.append(dict(
                    tag=f'{fam}-wd{m}-{label}', layers=[base],
                    overrides={'implementation': impl}, precision=_ISO_PRECISION,
                    sweep=[v for v in _ISO_SEQ if v <= cap] if cap else _ISO_SEQ,
                    # max-autotune recompiles per shape and can take minutes: run each
                    # seq_len under a wall-clock timeout so a runaway compile is killed.
                    compile_guard=(impl == 'custom_hopscan_autotune'),
                ))
    e.append(dict(tag='pdssm', layers=[f'pdssm-d{dim}-{suffix}'],
                  precision=_ISO_PRECISION, sweep=_ISO_SEQ))
    e.append(dict(tag='mamba2', layers=[f'mamba2-fla-d{dim}-{suffix}'],
                  precision=_ISO_PRECISION, sweep=_ISO_SEQ))
    e.append(dict(tag='deltanet', layers=[f'dnet-d{dim}-{suffix}'],
                  precision=_ISO_PRECISION, sweep=_ISO_SEQ))
    for r in _ISO_DPROD_RANKS:
        e.append(dict(tag=f'deltaproduct{r}', layers=[f'dproduct-hh{r}-d{dim}-{suffix}'],
                      precision=_ISO_PRECISION, sweep=_ISO_SEQ))
    return e

# A regime may carry a `grid` (second sweep axis run as a nested loop):
#   {'key': 'window_dim'|'hidden_dim'|'batch_size', 'values': [...]}
# The primary axis is always seq_len. `crossover` names the grid column for the
# parallel-vs-sequential phase map.
REGIMES: dict[str, dict] = {
    'seqlen-m1': dict(
        layers=['bdlru-sel-wd1-d128-h128'], sweep=_POW2, batch_size=32, at_seq_len=32768,
        desc='Length scaling, scalar block (m=1)',
    ),
    'seqlen-m4': dict(
        layers=['bdlru-sel-wd4-d128-h128'], sweep=_POW2, batch_size=8, at_seq_len=32768,
        desc='Length scaling, matrix block (m=4)',
    ),
    'blocksize': dict(
        layers=['bdlru-sel-wd1-d128-h128'], sweep=_SEQ_2D_HEAVY, batch_size=4,
        grid=dict(key='window_dim', values=[1, 2, 4, 8, 16, 32, 64]),
        crossover='window_dim', at_seq_len=8192,
        desc='Block-size x length (window_dim 1..64) -> block & crossover maps',
    ),
    'batch': dict(
        layers=['bdlru-sel-wd4-d128-h128'], sweep=_SEQ_2D,
        grid=dict(key='batch_size', values=[1, 8, 32, 128]),
        crossover='batch_size', at_seq_len=8192,
        desc='Batch x length (occupancy) -> crossover map',
    ),
    # batch x length throughput grids at other block sizes (m=4 == `batch` above).
    # Base layer is wd1; block size is set per-run via a window_dim override, exactly
    # as the `blocksize` regime does, so all share one registered layer.
    **{
        f'batch-m{m}': dict(
            layers=['bdlru-sel-wd1-d128-h128'], sweep=_SEQ_2D,
            grid=dict(key='batch_size', values=[1, 8, 32, 128]),
            extra_overrides={'window_dim': m},
            crossover='batch_size', at_seq_len=8192,
            desc=f'Batch x length throughput grid at block m={m}',
        )
        for m in (1, 2, 8, 16, 32)
    },
    'hidden': dict(
        layers=['bdlru-sel-wd4-d128-h128'], sweep=_SEQ_2D, batch_size=8,
        grid=dict(key='hidden_dim', values=[128, 256, 512, 1024]),
        crossover='hidden_dim', at_seq_len=8192,
        desc='Hidden-size x length -> crossover map',
    ),
    'inference': dict(
        layers=['bdlru-sel-wd4-d128-h128'], sweep=_POW2, batch_size=8, train=False,
        at_seq_len=32768, desc='Inference-only (forward) length scaling, m=4',
    ),
    # Cross-*model* comparison (not scan-impl overrides of one layer). Each entry is
    # a whole registered layer; the series label comes from the run_tag / model name.
    # NOTE: dnet-orig / dproduct-orig require the `fla` package (flash-linear-attention);
    # without it those two entries fail and are skipped (PDSSM still runs).
    'model-compare': dict(
        sweep=_POW2, batch_size=8, at_seq_len=8192,
        desc='Cross-model: PDSSM (assoc scan) vs DeltaNet vs DeltaProduct',
        models=[
            # PDSSM's dense complex NxN associative scan is memory-heavy -> cap length.
            dict(tag='pdssm_assoc', layers=['pdssm-d128-h128'],
                 overrides={'implementation': 'associative_scan'}, precision='32',
                 sweep=[256, 512, 1024, 2048, 4096, 8192]),
            # fla models run in their native bf16 (deltaproduct requires it; deltanet's
            # gated chunk backward also needs Triton+tilelang on Hopper -- see README).
            dict(tag='deltanet', layers=['dnet-orig'], precision='bf16'),
            dict(tag='deltaproduct', layers=['dproduct-orig'], precision='bf16'),
            # Mamba2 via fla's Triton backend (no mamba_ssm build); native bf16.
            dict(tag='mamba2', layers=['mamba2-fla-d128'], precision='bf16'),
        ],
    ),
    # ---- iso-tier sweeps: whole registered layers over a full B x T grid -----
    # Each is a model regime (whole layers, not scan-impl overrides of one layer)
    # with an added batch grid. OOM is recorded (no auto batch-reduction) so the
    # feasibility frontier is captured. BD-LRU/H-LRU run compiled-hopscan + Triton
    # sequential (sequential cut early in T). Precision is per-family (fp32 for the
    # fork recurrences, bf16 for fla baselines). autoplot is off -- these 27-series
    # grids are curated into per-family figures separately.
    'iso-d128-iso1m': dict(
        dim=128, suffix='iso1m', batch_grid=_ISO_BATCH, oom_retry=False,
        num_samples=_ISO_NUM_SAMPLES, autoplot=False, sweep=_ISO_SEQ,
        models=_iso_entries(128, 'iso1m'),
        desc='Iso-param ~1M @ d=128: full B x T grid, all families',
    ),
    'iso-d1024-iso100m': dict(
        dim=1024, suffix='iso100m', batch_grid=_ISO_BATCH, oom_retry=False,
        num_samples=_ISO_NUM_SAMPLES, autoplot=False, sweep=_ISO_SEQ,
        models=_iso_entries(1024, 'iso100m'),
        desc='Iso-param ~100M @ d=1024: full B x T grid, all families',
    ),
    'iso-d128-s1024': dict(
        dim=128, suffix='s1024', batch_grid=_ISO_BATCH, oom_retry=False,
        num_samples=_ISO_NUM_SAMPLES, autoplot=False, sweep=_ISO_SEQ,
        models=_iso_entries(128, 's1024'),
        desc='Iso-state 1024 @ d=128: full B x T grid, all families',
    ),
    'iso-d1024-s4096': dict(
        dim=1024, suffix='s4096', batch_grid=_ISO_BATCH, oom_retry=False,
        num_samples=_ISO_NUM_SAMPLES, autoplot=False, sweep=_ISO_SEQ,
        models=_iso_entries(1024, 's4096'),
        desc='Iso-state 4096 @ d=1024: full B x T grid, all families',
    ),
}

BASE = dict(task='selective-copying', vocab_size=32, precision='32')
RESULTS_ROOT = 'results'
FIGURES_ROOT = 'figures'


def _run_tag(regime: dict, impl: str, grid_val) -> str:
    grid = regime.get('grid')
    return impl if grid is None else f"{impl}|{grid['key']}={grid_val}"


def _sweep_for(regime: dict, impl: str) -> list[int]:
    sweep = list(regime['sweep'])
    if impl == 'orig':
        sweep = [v for v in sweep if v <= ORIG_MAX_SEQ]
    return sweep


def _missing_values(csv_path: str, run_tag: str, sweep: list[int]) -> list[int]:
    """Sweep values not yet recorded for this run_tag (resumability)."""
    if not os.path.exists(csv_path):
        return list(sweep)
    import pandas as pd
    df = pd.read_csv(csv_path)
    if 'run_tag' not in df.columns or 'seq_len' not in df.columns:
        return list(sweep)
    done = set(df[df['run_tag'] == run_tag]['seq_len'].dropna().astype(int))
    return [v for v in sweep if v not in done]


def _build_argv(regime: dict, impl: str, name: str, grid_val,
                sweep: list[int], run_tag: str, step_ceiling_ms=None) -> list[str]:
    grid = regime.get('grid')
    overrides = [f'implementation={impl}']
    for k, v in regime.get('extra_overrides', {}).items():
        overrides.append(f'{k}={v}')
    batch_size = regime.get('batch_size')
    if grid is not None:
        key = grid['key']
        if key in ('window_dim', 'hidden_dim'):
            overrides.append(f'{key}={grid_val}')
        elif key == 'batch_size':
            batch_size = grid_val

    argv = [
        'speed_benchmark.py',
        '--layers', *regime['layers'],
        '--task', BASE['task'], '--vocab-size', str(BASE['vocab_size']),
        '--precision', BASE['precision'],
        '--sweep-key', 'seq_len',
        '--sweep-values', *[str(v) for v in sweep],
        '--name', name, '--run-tag', run_tag, '--no-plot',
        '--warmup', str(TIMING['warmup']), '--iters', str(TIMING['iters']),
        '--repeats', str(TIMING['repeats']),
        '--layer-overrides', *overrides,
    ]
    if batch_size is not None:
        argv += ['--batch-size', str(batch_size)]
    if regime.get('train') is False:
        argv += ['--no-train']
    if step_ceiling_ms is not None:
        argv += ['--step-ceiling-ms', str(step_ceiling_ms)]
    return argv


def _build_model_argv(regime: dict, entry: dict, name: str, sweep: list[int],
                      batch_size, run_tag: str, step_ceiling_ms=None) -> list[str]:
    """Build a speed_benchmark argv for a whole-model entry of a model regime."""
    overrides = [f'{k}={v}' for k, v in entry.get('overrides', {}).items()]
    precision = entry.get('precision', BASE['precision'])
    argv = [
        'speed_benchmark.py',
        '--layers', *entry['layers'],
        '--dim', str(regime.get('dim', 128)),
        '--task', BASE['task'], '--vocab-size', str(BASE['vocab_size']),
        '--precision', precision,
        '--sweep-key', 'seq_len',
        '--sweep-values', *[str(v) for v in sweep],
        '--name', name, '--run-tag', run_tag, '--no-plot',
        '--warmup', str(TIMING['warmup']), '--iters', str(TIMING['iters']),
        '--repeats', str(TIMING['repeats']),
    ]
    if overrides:
        argv += ['--layer-overrides', *overrides]
    if batch_size is not None:
        argv += ['--batch-size', str(batch_size)]
    if regime.get('num_samples') is not None:
        argv += ['--num-samples', str(regime['num_samples'])]
    if regime.get('oom_retry') is False:
        argv += ['--no-oom-retry']
    if regime.get('train') is False:
        argv += ['--no-train']
    if step_ceiling_ms is not None:
        argv += ['--step-ceiling-ms', str(step_ceiling_ms)]
    return argv


def _run_speed_benchmark(argv: list[str], timeout: float | None = None) -> bool:
    """Run one speed_benchmark invocation in an ISOLATED subprocess.

    Each (entry|impl, batch) invocation sweeps its seq_len list in a fresh CUDA
    context. This matters because a CUDA *illegal memory access* (unlike an OOM,
    which we catch and record) permanently corrupts the process's CUDA context --
    running in-process would let one bad cell poison every subsequent cell. A
    subprocess confines any such fatal fault to a single invocation; the suite
    continues with a clean context. speed_benchmark still writes its own rows
    (ok/OOM/error) at the end of the sweep, so resume/skip logic is unaffected.

    If `timeout` is set, the subprocess is started in its own process group and,
    on expiry, the whole group (python + any torch.compile / autotune workers) is
    SIGKILLed. Returns True iff the cell timed out (caller records a capped row).
    """
    import subprocess
    import signal
    script = os.path.join(BENCH_DIR, 'speed_benchmark.py')
    proc = subprocess.Popen([sys.executable, script, *argv[1:]], cwd=BENCH_DIR,
                            start_new_session=True)
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        proc.wait()
        print(f"  [timeout] cell exceeded {timeout:.0f}s -> killed group "
              f"(compile budget); recording capped_compile")
        return True
    if proc.returncode != 0:
        # non-zero => the subprocess died before writing its CSV (e.g. an
        # uncatchable abort). Those cells stay "missing" and are retried on resume.
        print(f"  [!] speed_benchmark subprocess exited {proc.returncode} "
              f"(fault isolated; cells will be retried on resume)")
    return False


def _append_capped_row(csv_path: str, entry: dict, regime: dict, batch_size,
                       run_tag: str, seq_len: int, value_ms: float, status: str) -> None:
    """Append a single off-chart sentinel row for a cell we aborted (compile timeout).

    Mirrors speed_benchmark's schema and reuses the existing CSV header order so the
    appended row lines up. The sentinel latency (`value_ms`) is the step ceiling, so
    the cell reads as extremely slow (near-zero throughput) in every plot/aggregate.
    """
    import csv
    overrides = entry.get('overrides', {})
    override_suffix = ''.join(f'[{k}={v}]' for k, v in overrides.items())
    train = regime.get('train', True) is not False
    val = float(value_ms)
    row = {
        'batch_size': batch_size,
        'layers': ' '.join(entry['layers']),
        'mode': 'train' if train else 'inference',
        'model': '-'.join(entry['layers']) + override_suffix,
        'overrides': override_suffix,
        'params': '',
        'peak_mem_mb': '',
        'precision': entry.get('precision', BASE['precision']),
        'repeats': 0,
        'run_tag': run_tag,
        'samples_per_s': batch_size / (val / 1e3),
        'seq_len': seq_len,
        'status': status,
        'step_ms': val,
        'step_ms_iqr': 0.0,
        'step_ms_min': val,
        'step_ms_std': 0.0,
        'task': BASE['task'],
        'tokens_per_s': (batch_size * seq_len) / (val / 1e3),
        'train': train,
    }
    write_header = not os.path.exists(csv_path)
    if write_header:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        fieldnames = sorted(row)
    else:
        with open(csv_path, newline='') as f:
            fieldnames = next(csv.reader(f))
    with open(csv_path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore', restval='')
        if write_header:
            w.writeheader()
        w.writerow(row)


def _write_run_meta(name: str) -> None:
    """Record hardware / library versions for reproducibility (reviewers ask)."""
    import torch
    meta = {
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
        'timing': TIMING,
        'orig_max_seq': ORIG_MAX_SEQ,
    }
    try:
        import triton
        meta['triton'] = triton.__version__
    except Exception:  # noqa: BLE001
        meta['triton'] = None
    out_dir = os.path.join(RESULTS_ROOT, name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'run_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)


def _plot_regime(regime: dict, name: str) -> None:
    csv_path = os.path.join(RESULTS_ROOT, name, 'results.csv')
    if not os.path.exists(csv_path):
        print(f"  [!] no results for regime '{name}', skipping figures")
        return
    import pandas as pd
    from scripts.plot_speed import render
    from scripts.plot_compare import generate as generate_comparison
    from scripts.plot_crossover import generate as generate_crossover

    df = pd.read_csv(csv_path)
    is_grid = regime.get('grid') is not None

    for metric in ('tokens_per_s', 'step_ms', 'peak_mem_mb'):
        path = render(df, name, metric, x='seq_len', logx=True, logy=True, figures_root=FIGURES_ROOT)
        if path:
            print(f"  figure: {path}")

    at = regime.get('at_seq_len') or max(regime['sweep'])
    if is_grid:
        cy = regime.get('crossover', 'window_dim')
        for path in generate_crossover([csv_path], name, x='seq_len', y=cy, figures_root=FIGURES_ROOT):
            print(f"  figure: {path}")
        if regime['grid']['key'] == 'window_dim':
            for path in generate_comparison([csv_path], name, x='seq_len', at_seq_len=at,
                                            figures_root=FIGURES_ROOT, min_impls=2):
                if 'block_comparison' in path:
                    print(f"  figure: {path}")
        if regime['grid']['key'] == 'batch_size':
            from scripts.plot_grid import generate as generate_grid
            for path in generate_grid([csv_path], name, row_var='seq_len',
                                      col_var='batch_size', figures_root=FIGURES_ROOT):
                print(f"  figure: {path}")
    else:
        for path in generate_comparison([csv_path], name, x='seq_len', at_seq_len=at,
                                        figures_root=FIGURES_ROOT, min_impls=2):
            print(f"  figure: {path}")


def run_regime(name: str, regime: dict, impls: list[str], dry_run: bool, fresh: bool,
               step_ceiling_ms=None, per_cell_timeout=None) -> None:
    print(f"\n{'=' * 78}\n# regime '{name}': {regime['desc']}\n{'=' * 78}")
    csv_path = os.path.join(RESULTS_ROOT, name, 'results.csv')
    if fresh and not dry_run:
        for root in (RESULTS_ROOT, FIGURES_ROOT):
            shutil.rmtree(os.path.join(root, name), ignore_errors=True)

    # off-chart sentinel latency for compile-timeout rows (falls back if no ceiling)
    sentinel_ms = step_ceiling_ms if step_ceiling_ms is not None else 5000.0

    # model-comparison regimes iterate whole layers, not scan-impl overrides.
    # An optional `batch_grid` adds a second axis (run as a nested loop), with the
    # batch value baked into the run_tag for resumability and grid plotting.
    models = regime.get('models')
    if models is not None:
        batch_grid = regime.get('batch_grid')
        batch_vals = batch_grid if batch_grid is not None else [regime.get('batch_size')]
        for entry in models:
            for bv in batch_vals:
                tag = entry['tag'] if batch_grid is None else f"{entry['tag']}|batch_size={bv}"
                sweep = entry.get('sweep', regime['sweep'])
                missing = sweep if (fresh and not dry_run) else _missing_values(csv_path, tag, sweep)
                if not missing:
                    print(f"  skip (done): {tag}")
                    continue
                # compile-guarded entries (e.g. custom_hopscan_autotune) run one
                # seq_len per subprocess so a runaway per-shape compile can be killed
                # at `per_cell_timeout` without losing the cells already recorded.
                guarded = per_cell_timeout is not None and entry.get('compile_guard')
                try:
                    if guarded:
                        for sv in missing:
                            argv = _build_model_argv(regime, entry, name, [sv], bv, tag,
                                                     step_ceiling_ms)
                            print(f"\n--- {name} :: {tag}  (T={sv}) "
                                  f"[compile-guarded {per_cell_timeout:.0f}s] ---")
                            if dry_run:
                                print('  ' + ' '.join(argv))
                                continue
                            if _run_speed_benchmark(argv, timeout=per_cell_timeout):
                                _append_capped_row(csv_path, entry, regime, bv, tag, sv,
                                                   sentinel_ms,
                                                   f'capped_compile:>{int(per_cell_timeout)}s')
                    else:
                        argv = _build_model_argv(regime, entry, name, missing, bv, tag,
                                                 step_ceiling_ms)
                        print(f"\n--- {name} :: {tag}  (T={missing}) ---")
                        if dry_run:
                            print('  ' + ' '.join(argv))
                            continue
                        _run_speed_benchmark(argv)
                except Exception as exc:  # noqa: BLE001 - keep the suite going on any failure
                    print(f"  [!] {tag} failed: {type(exc).__name__}: {exc}")
        if not dry_run:
            _write_run_meta(name)
            if regime.get('autoplot', True):
                print(f"\n# plotting regime '{name}'")
                _plot_regime(regime, name)
            else:
                print(f"  (autoplot off for '{name}'; curate per-family figures separately)")
        return

    grid = regime.get('grid')
    grid_vals = grid['values'] if grid is not None else [None]
    for impl in impls:
        for gv in grid_vals:
            tag = _run_tag(regime, impl, gv)
            sweep = _sweep_for(regime, impl)
            missing = sweep if (fresh and not dry_run) else _missing_values(csv_path, tag, sweep)
            label = impl + (f" {grid['key']}={gv}" if grid is not None else "")
            if not missing:
                print(f"  skip (done): {label}")
                continue
            argv = _build_argv(regime, impl, name, gv, missing, tag, step_ceiling_ms)
            print(f"\n--- {name} :: {label}  (T={missing}) ---")
            if dry_run:
                print('  ' + ' '.join(argv))
                continue
            try:
                _run_speed_benchmark(argv)
            except Exception as exc:  # noqa: BLE001 - keep the suite going on any failure
                print(f"  [!] {label} failed: {type(exc).__name__}: {exc}")

    if not dry_run:
        _write_run_meta(name)
        print(f"\n# plotting regime '{name}'")
        _plot_regime(regime, name)


def get_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--regime', nargs='+', default=list(REGIMES),
                   choices=list(REGIMES), help='regimes to run (default: all)')
    p.add_argument('--impls', nargs='+', default=ALL_IMPLS,
                   help='implementations to compare (default: all six incl. orig)')
    p.add_argument('--fresh', action='store_true',
                   help='wipe each selected regime (results+figures) and recompute from scratch')
    p.add_argument('--dry-run', action='store_true', help='print the plan without running')
    p.add_argument('--iso-batch', nargs='+', type=int, default=None,
                   help='override the batch grid for iso regimes (e.g. --iso-batch 1 4 32 128)')
    p.add_argument('--iso-max-seq', type=int, default=None,
                   help='drop sweep sequence lengths above this for iso regimes '
                        '(e.g. --iso-max-seq 65536 drops 2^18)')
    p.add_argument('--iso-lru-impls', nargs='+', default=None,
                   choices=list(_ISO_LRU_IMPL_LABELS),
                   help='override which BD-LRU/H-LRU scan impls run in iso regimes '
                        '(default: orig hopscan_custom triton_auto). e.g. '
                        '--iso-lru-impls hopscan_custom triton_auto  (drops orig)')
    p.add_argument('--step-ceiling-ms', type=float, default=None,
                   help='per-step latency ceiling (ms): a cell whose step exceeds it is '
                        'aborted early and recorded as capped_step:>Nms (off-chart sentinel). '
                        'Stops non-competitive sequential loops (e.g. orig at long T).')
    p.add_argument('--per-cell-timeout-s', type=float, default=None,
                   help='wall-clock budget (s) per compile-guarded cell (e.g. custom_hopscan_'
                        'autotune). If exceeded the subprocess (and its compile workers) is '
                        'killed and the cell recorded as capped_compile:>Ns. Requires running '
                        'compile-guarded entries one seq_len per subprocess.')
    return p.parse_args()


def _apply_iso_overrides(regime: dict, iso_batch, iso_max_seq, iso_lru_impls=None) -> None:
    """Mutate an iso regime in place per the --iso-* overrides (no-op otherwise)."""
    # Rebuild the model entries with a chosen BD-LRU/H-LRU impl set first, so the
    # sequence-length filtering below applies to the rebuilt sweeps too.
    if iso_lru_impls is not None and regime.get('suffix') is not None:
        regime['models'] = _iso_entries(regime['dim'], regime['suffix'], iso_lru_impls)
    if iso_batch is not None and regime.get('batch_grid') is not None:
        regime['batch_grid'] = list(iso_batch)
    if iso_max_seq is not None:
        if 'sweep' in regime:
            regime['sweep'] = [v for v in regime['sweep'] if v <= iso_max_seq]
        for entry in regime.get('models') or []:
            if 'sweep' in entry:
                entry['sweep'] = [v for v in entry['sweep'] if v <= iso_max_seq]


def main():
    args = get_args()
    print(f"BD-LRU comparison suite: regimes={args.regime}\n  impls={args.impls}\n  fresh={args.fresh}")
    for name in args.regime:
        regime = REGIMES[name]
        _apply_iso_overrides(regime, args.iso_batch, args.iso_max_seq, args.iso_lru_impls)
        run_regime(name, regime, args.impls, args.dry_run, args.fresh,
                   step_ceiling_ms=args.step_ceiling_ms,
                   per_cell_timeout=args.per_cell_timeout_s)
    print('\nDone.')


if __name__ == '__main__':
    main()
