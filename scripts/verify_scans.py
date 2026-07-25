#!/usr/bin/env python3
"""Numerical correctness & precision reporter for BD-LRU scan implementations (Q1/Q7).

Unlike ``tests/test_scans.py`` (pass/fail asserts), this emits a *quantitative
error table* -- the artifact a reviewer expects -- reporting forward and backward
error of every scan implementation against a high-precision (**float64**)
sequential reference, at both fp32 and bf16.

Two levels are measured (BD-LRU only; H-LRU is deferred):

  1. Primitive level -- the differentiable affine scan on generic (A, b):
       affine_scan_torch_impl (torch.associative_scan),
       triton_sequential / triton_persistent / triton_parallel_blelloch.
     Ground truth: an fp64 sequential loop h_t = A_t h_{t-1} + b_t and its
     autograd (exact adjoint). We report max abs + relative error for the output
     y, and for grad_A / grad_b.

  2. Layer level -- BD-LRU_sel forward and input-gradient for every
     ``implementation`` option, against the ``orig`` sequential loop run in fp32
     (its native precision; identical init via a fixed seed). This is the
     end-to-end equivalence check.

For each row we sweep dtype in {float32, bfloat16} so the fp32 (kernels
accumulate in fp32) vs bf16 (storage-precision) trade-off is visible.

Results are written to ``results/verify/errors.csv`` and printed as a table.
Nothing is timed here; see speed_benchmark.py for throughput.

Usage:
    uv run python -m scripts.verify_scans
    uv run python -m scripts.verify_scans --dtypes float32 bfloat16 --out results/verify/errors.csv
"""

from __future__ import annotations

import os
import csv
import argparse
import typing as tp

import torch

from mad.model.layers.ops.scans.triton_scans import triton_affine_scan, _reference_scan
from mad.model.layers.bdlru_sel import BDLRU_sel


# primitive impl name -> callable(A, b) -> y  (all differentiable)
PRIMITIVES: dict[str, tp.Callable] = {
    'affine_scan_torch_impl': lambda A, b: _reference_scan(A, b),
    'triton_sequential': lambda A, b: triton_affine_scan(A, b, 'sequential'),
    'triton_persistent': lambda A, b: triton_affine_scan(A, b, 'persistent'),
    'triton_parallel_blelloch': lambda A, b: triton_affine_scan(A, b, 'blelloch'),
    'triton_chunked': lambda A, b: triton_affine_scan(A, b, 'chunked'),
    'triton_auto': lambda A, b: triton_affine_scan(A, b, 'auto'),
}

# BD-LRU layer implementation options to check against the `orig` fp64 loop.
LAYER_IMPLS = [
    'affine_scan_torch_impl', 'hopscan_custom', 'custom_hopscan_autotune',
    'triton_sequential', 'triton_persistent', 'triton_parallel_blelloch',
    'triton_chunked', 'triton_auto',
]

_DTYPES = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}

# primitive test shapes (BB, T, D) and layer shapes (window_dim, hidden_dim)
PRIMITIVE_SHAPES = [(128, 256, 1), (128, 256, 4), (64, 512, 8), (32, 512, 16)]
LAYER_SHAPES = [(1, 128), (4, 64), (8, 32), (16, 16)]


def _cuda_ready() -> bool:
    if not torch.cuda.is_available():
        print('CUDA unavailable -- this reporter needs a GPU + triton. Aborting.')
        return False
    try:
        import triton  # noqa: F401
    except Exception:
        print('triton unavailable -- aborting.')
        return False
    return True


def _err(got: torch.Tensor, ref64: torch.Tensor) -> tuple[float, float]:
    """(max abs error, max relative error) of `got` vs an fp64 reference."""
    got64 = got.detach().double()
    abs_err = (got64 - ref64).abs().max().item()
    denom = ref64.abs().max().item()
    rel_err = abs_err / denom if denom > 0 else float('nan')
    return abs_err, rel_err


def _sequential_fp64(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """fp64 ground-truth scan h_t = A_t h_{t-1} + b_t (differentiable)."""
    BB, T, D, _ = A.shape
    h = torch.zeros(BB, D, device=A.device, dtype=torch.float64)
    ys = []
    for t in range(T):
        h = torch.einsum('bij,bj->bi', A[:, t], h) + b[:, t]
        ys.append(h)
    return torch.stack(ys, dim=1)


# --------------------------------------------------------------------------- #
# primitive-level verification
# --------------------------------------------------------------------------- #
def verify_primitives(dtypes: list[str], seed: int) -> list[dict]:
    rows: list[dict] = []
    for (BB, T, D) in PRIMITIVE_SHAPES:
        torch.manual_seed(seed)
        # base tensors in fp64. Scale A by ~1/sqrt(D) so its operator norm stays
        # < 1 (the recurrence is *contractive*, like a trained gated BD-LRU);
        # otherwise the state diverges over T steps and abs errors are dominated
        # by huge magnitudes rather than kernel accuracy.
        A64 = torch.randn(BB, T, D, D, device='cuda', dtype=torch.float64) * (0.25 / D ** 0.5)
        b64 = torch.randn(BB, T, D, device='cuda', dtype=torch.float64)
        gy64 = torch.randn(BB, T, D, device='cuda', dtype=torch.float64)

        # fp64 ground truth (forward + exact adjoint)
        A_ref = A64.clone().requires_grad_(True)
        b_ref = b64.clone().requires_grad_(True)
        y_ref = _sequential_fp64(A_ref, b_ref)
        gA_ref, gb_ref = torch.autograd.grad(y_ref, (A_ref, b_ref), gy64)
        y_ref = y_ref.detach()

        for dname in dtypes:
            dt = _DTYPES[dname]
            for impl, fn in PRIMITIVES.items():
                row = dict(level='primitive', impl=impl, dtype=dname,
                           BB=BB, T=T, D=D, status='ok')
                try:
                    A = A64.to(dt).clone().requires_grad_(True)
                    b = b64.to(dt).clone().requires_grad_(True)
                    y = fn(A, b)
                    gA, gb = torch.autograd.grad(y, (A, b), gy64.to(dt))
                    row['fwd_abs'], row['fwd_rel'] = _err(y, y_ref)
                    row['gA_abs'], row['gA_rel'] = _err(gA, gA_ref)
                    row['gb_abs'], row['gb_rel'] = _err(gb, gb_ref)
                except Exception as exc:  # noqa: BLE001 - record failures, keep going
                    row['status'] = f'{type(exc).__name__}: {str(exc)[:80]}'
                rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# layer-level verification (BD-LRU): fwd + input-grad vs `orig` fp64 loop
# --------------------------------------------------------------------------- #
def _bdlru_fwd_grad(impl: str, x: torch.Tensor, w: torch.Tensor, seed: int,
                    window_dim: int, hidden_dim: int, dtype: torch.dtype):
    """Build BD-LRU at (impl, dtype) with fixed-seed init; return (y, input grad)."""
    torch.manual_seed(seed)
    layer = BDLRU_sel(dim=x.shape[-1], implementation=impl,
                      window_dim=window_dim, hidden_dim=hidden_dim).cuda().to(dtype)
    xin = x.to(dtype).clone().detach().requires_grad_(True)
    y = layer(xin)
    (y.float() * w).sum().backward()
    return y.detach(), xin.grad.detach()


def verify_bdlru_layer(dtypes: list[str], seed: int) -> list[dict]:
    rows: list[dict] = []
    B, T, dim = 2, 64, 128
    for (window_dim, hidden_dim) in LAYER_SHAPES:
        torch.manual_seed(seed + 1)
        x = torch.randn(B, T, dim, device='cuda')
        w = torch.randn(B, T, dim, device='cuda')  # fixed loss weighting (fp32)

        # Reference: the `orig` sequential loop in fp32 (its native precision;
        # the layer is not float64-clean, and the kernels compute in fp32 anyway,
        # so fp32-orig is the ground truth for this equivalence check).
        y_ref, g_ref = _bdlru_fwd_grad('orig', x, w, seed, window_dim, hidden_dim, torch.float32)
        y_ref64, g_ref64 = y_ref.double(), g_ref.double()

        for dname in dtypes:
            dt = _DTYPES[dname]
            for impl in LAYER_IMPLS:
                row = dict(level='bdlru', impl=impl, dtype=dname,
                           window_dim=window_dim, hidden_dim=hidden_dim, status='ok')
                try:
                    y, g = _bdlru_fwd_grad(impl, x, w, seed, window_dim, hidden_dim, dt)
                    row['fwd_abs'], row['fwd_rel'] = _err(y, y_ref64)
                    row['grad_abs'], row['grad_rel'] = _err(g, g_ref64)
                except Exception as exc:  # noqa: BLE001
                    row['status'] = f'{type(exc).__name__}: {str(exc)[:80]}'
                rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def _print_primitive(rows: list[dict]) -> None:
    print('\n=== Primitive scan error vs fp64 sequential reference ===')
    hdr = f"{'impl':26s} {'dtype':9s} {'BB,T,D':13s} {'fwd_abs':>10s} {'fwd_rel':>10s} {'gA_rel':>10s} {'gb_rel':>10s}"
    print(hdr)
    print('-' * len(hdr))
    for r in rows:
        shape = f"{r['BB']},{r['T']},{r['D']}"
        if r['status'] != 'ok':
            print(f"{r['impl']:26s} {r['dtype']:9s} {shape:13s}  -> {r['status']}")
            continue
        print(f"{r['impl']:26s} {r['dtype']:9s} {shape:13s} "
              f"{r['fwd_abs']:10.2e} {r['fwd_rel']:10.2e} {r['gA_rel']:10.2e} {r['gb_rel']:10.2e}")


def _print_layer(rows: list[dict]) -> None:
    print('\n=== BD-LRU layer error vs `orig` fp32 loop (fwd + input grad) ===')
    hdr = f"{'impl':26s} {'dtype':9s} {'m x N':10s} {'fwd_abs':>10s} {'fwd_rel':>10s} {'grad_abs':>10s} {'grad_rel':>10s}"
    print(hdr)
    print('-' * len(hdr))
    for r in rows:
        shape = f"{r['window_dim']}x{r['hidden_dim']}"
        if r['status'] != 'ok':
            print(f"{r['impl']:26s} {r['dtype']:9s} {shape:10s}  -> {r['status']}")
            continue
        print(f"{r['impl']:26s} {r['dtype']:9s} {shape:10s} "
              f"{r['fwd_abs']:10.2e} {r['fwd_rel']:10.2e} {r['grad_abs']:10.2e} {r['grad_rel']:10.2e}")


def _write_csv(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = sorted({k for r in rows for k in r})
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def get_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dtypes', nargs='+', default=['float32', 'bfloat16'],
                   choices=list(_DTYPES), help='precisions to test (default: float32 bfloat16)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', type=str, default='results/verify/errors.csv',
                   help='CSV output path (default: results/verify/errors.csv)')
    p.add_argument('--skip-primitives', action='store_true')
    p.add_argument('--skip-layer', action='store_true')
    return p.parse_args()


def main():
    args = get_args()
    if not _cuda_ready():
        raise SystemExit(1)

    rows: list[dict] = []
    if not args.skip_primitives:
        prim = verify_primitives(args.dtypes, args.seed)
        _print_primitive(prim)
        rows += prim
    if not args.skip_layer:
        layer = verify_bdlru_layer(args.dtypes, args.seed)
        _print_layer(layer)
        rows += layer

    _write_csv(args.out, rows)
    print(f'\nWrote {len(rows)} rows to {args.out}')


if __name__ == '__main__':
    main()
