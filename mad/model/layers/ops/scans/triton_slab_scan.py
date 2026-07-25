"""Log-depth affine scans for narrow blocks via ``tl.associative_scan``.

The layers store ``A`` as ``(BB, T, D, D)``, so *time* is the contiguous axis.
That makes a slab of consecutive timesteps a coalesced load, and lets a single
program resolve the whole slab in O(log SLAB) depth instead of walking it step
by step -- which is what the ``sequential`` / ``persistent`` kernels do, leaving
them latency-bound at these widths (~100-450 GB/s of ~3350 available on H100).

``tl.associative_scan`` hands ``combine_fn`` one *scalar* per input, so it cannot
reduce over a tile axis; a generic D implementation is therefore impossible and
each entry of ``A`` and ``b`` has to ride along as its own scan input. The
kernels below are consequently code-generated per width (D^2 + D inputs, and a
combine that is an explicit D x D matmul).

Measured against the best of sequential/persistent/chunked at T=8192:

    D=1   9-19x     (scalar case, in ``triton_auto_scan``)
    D=2   7-16x
    D=4   3.1-6.4x
    D=8   0.6-1.8x  -- not used

D=8 is the cutoff: the scan carries 72 live values, which spills at SLAB>=128
(104 spilled bytes, 3092 at SLAB=256), and the O(D^3 log SLAB) combine work
finally pushes past the H100 roofline (~28 flop/byte vs ~18 available), so it
loses outright above BB=1024.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# Widths with a generated kernel. Only powers of two are emitted, since those
# are the block sizes the layers actually use (``window_dim``); anything else
# falls through to the general chunked path.
SLAB_BLOCKS = frozenset({2, 4})

# (slab length, warps) per width. Warps drop at high row counts, where slabs
# from many rows already saturate the SMs and narrower programs schedule better.
_SLAB = {2: 512, 4: 128}
_WIDE_WARP_ROWS = {2: 512, 4: 256}


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def slab_params(block: int, rows: int, seq_len: int) -> tuple[int, int]:
    """Slab length (a power of 2, never wider than needed) and warp count."""
    slab = max(16, min(_SLAB[block], _next_pow2(seq_len)))
    return slab, (4 if rows <= _WIDE_WARP_ROWS[block] else 2)


@triton.jit
def _combine2(
    l0, l1, l2, l3, lb0, lb1, r0, r1, r2, r3, rb0, rb1,
):
    """Compose two 2x2 affine maps: ``(A_r @ A_l, A_r @ b_l + b_r)``."""
    return (
        r0 * l0 + r1 * l2,
        r0 * l1 + r1 * l3,
        r2 * l0 + r3 * l2,
        r2 * l1 + r3 * l3,
        r0 * lb0 + r1 * lb1 + rb0,
        r2 * lb0 + r3 * lb1 + rb1,
    )


@triton.jit
def _slab_scan2_kernel(A_ptr, b_ptr, y_ptr, T, SLAB: tl.constexpr):
    """One program per row; each pass resolves SLAB timesteps in log depth.

    The 4 entries of A and 2 of b ride along as separate scan inputs, since
    ``combine_fn`` sees one scalar per input (it cannot reduce over a tile axis).
    """
    row = tl.program_id(0)
    s = tl.arange(0, SLAB)
    h0 = 0.0; h1 = 0.0
    for t0 in range(0, T, SLAB):
        t = t0 + s
        msk = t < T
        ao = row.to(tl.int64) * T * 4 + t.to(tl.int64) * 4
        bo = row.to(tl.int64) * T * 2 + t.to(tl.int64) * 2
        # masked lanes load the identity map so the scan is exact past the end
        a0 = tl.load(A_ptr + ao + 0, mask=msk, other=1.0).to(tl.float32)
        a1 = tl.load(A_ptr + ao + 1, mask=msk, other=0.0).to(tl.float32)
        a2 = tl.load(A_ptr + ao + 2, mask=msk, other=0.0).to(tl.float32)
        a3 = tl.load(A_ptr + ao + 3, mask=msk, other=1.0).to(tl.float32)
        c0 = tl.load(b_ptr + bo + 0, mask=msk, other=0.0).to(tl.float32)
        c1 = tl.load(b_ptr + bo + 1, mask=msk, other=0.0).to(tl.float32)
        (
            p0, p1, p2, p3, y0, y1,
        ) = tl.associative_scan(
            (
                a0, a1, a2, a3, c0, c1,
            ), 0, _combine2,
        )
        # fold in the state carried from earlier slabs
        y0 = y0 + p0 * h0 + p1 * h1
        y1 = y1 + p2 * h0 + p3 * h1
        tl.store(y_ptr + bo + 0, y0.to(y_ptr.dtype.element_ty), mask=msk)
        tl.store(y_ptr + bo + 1, y1.to(y_ptr.dtype.element_ty), mask=msk)
        last = s == SLAB - 1
        h0 = tl.sum(tl.where(last, y0, 0.0), 0)
        h1 = tl.sum(tl.where(last, y1, 0.0), 0)


@triton.jit
def _combine4(
    l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, l15, lb0, lb1,
    lb2, lb3, r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
    rb0, rb1, rb2, rb3,
):
    """Compose two 4x4 affine maps: ``(A_r @ A_l, A_r @ b_l + b_r)``."""
    return (
        r0 * l0 + r1 * l4 + r2 * l8 + r3 * l12,
        r0 * l1 + r1 * l5 + r2 * l9 + r3 * l13,
        r0 * l2 + r1 * l6 + r2 * l10 + r3 * l14,
        r0 * l3 + r1 * l7 + r2 * l11 + r3 * l15,
        r4 * l0 + r5 * l4 + r6 * l8 + r7 * l12,
        r4 * l1 + r5 * l5 + r6 * l9 + r7 * l13,
        r4 * l2 + r5 * l6 + r6 * l10 + r7 * l14,
        r4 * l3 + r5 * l7 + r6 * l11 + r7 * l15,
        r8 * l0 + r9 * l4 + r10 * l8 + r11 * l12,
        r8 * l1 + r9 * l5 + r10 * l9 + r11 * l13,
        r8 * l2 + r9 * l6 + r10 * l10 + r11 * l14,
        r8 * l3 + r9 * l7 + r10 * l11 + r11 * l15,
        r12 * l0 + r13 * l4 + r14 * l8 + r15 * l12,
        r12 * l1 + r13 * l5 + r14 * l9 + r15 * l13,
        r12 * l2 + r13 * l6 + r14 * l10 + r15 * l14,
        r12 * l3 + r13 * l7 + r14 * l11 + r15 * l15,
        r0 * lb0 + r1 * lb1 + r2 * lb2 + r3 * lb3 + rb0,
        r4 * lb0 + r5 * lb1 + r6 * lb2 + r7 * lb3 + rb1,
        r8 * lb0 + r9 * lb1 + r10 * lb2 + r11 * lb3 + rb2,
        r12 * lb0 + r13 * lb1 + r14 * lb2 + r15 * lb3 + rb3,
    )


@triton.jit
def _slab_scan4_kernel(A_ptr, b_ptr, y_ptr, T, SLAB: tl.constexpr):
    """One program per row; each pass resolves SLAB timesteps in log depth.

    The 16 entries of A and 4 of b ride along as separate scan inputs, since
    ``combine_fn`` sees one scalar per input (it cannot reduce over a tile axis).
    """
    row = tl.program_id(0)
    s = tl.arange(0, SLAB)
    h0 = 0.0; h1 = 0.0; h2 = 0.0; h3 = 0.0
    for t0 in range(0, T, SLAB):
        t = t0 + s
        msk = t < T
        ao = row.to(tl.int64) * T * 16 + t.to(tl.int64) * 16
        bo = row.to(tl.int64) * T * 4 + t.to(tl.int64) * 4
        # masked lanes load the identity map so the scan is exact past the end
        a0 = tl.load(A_ptr + ao + 0, mask=msk, other=1.0).to(tl.float32)
        a1 = tl.load(A_ptr + ao + 1, mask=msk, other=0.0).to(tl.float32)
        a2 = tl.load(A_ptr + ao + 2, mask=msk, other=0.0).to(tl.float32)
        a3 = tl.load(A_ptr + ao + 3, mask=msk, other=0.0).to(tl.float32)
        a4 = tl.load(A_ptr + ao + 4, mask=msk, other=0.0).to(tl.float32)
        a5 = tl.load(A_ptr + ao + 5, mask=msk, other=1.0).to(tl.float32)
        a6 = tl.load(A_ptr + ao + 6, mask=msk, other=0.0).to(tl.float32)
        a7 = tl.load(A_ptr + ao + 7, mask=msk, other=0.0).to(tl.float32)
        a8 = tl.load(A_ptr + ao + 8, mask=msk, other=0.0).to(tl.float32)
        a9 = tl.load(A_ptr + ao + 9, mask=msk, other=0.0).to(tl.float32)
        a10 = tl.load(A_ptr + ao + 10, mask=msk, other=1.0).to(tl.float32)
        a11 = tl.load(A_ptr + ao + 11, mask=msk, other=0.0).to(tl.float32)
        a12 = tl.load(A_ptr + ao + 12, mask=msk, other=0.0).to(tl.float32)
        a13 = tl.load(A_ptr + ao + 13, mask=msk, other=0.0).to(tl.float32)
        a14 = tl.load(A_ptr + ao + 14, mask=msk, other=0.0).to(tl.float32)
        a15 = tl.load(A_ptr + ao + 15, mask=msk, other=1.0).to(tl.float32)
        c0 = tl.load(b_ptr + bo + 0, mask=msk, other=0.0).to(tl.float32)
        c1 = tl.load(b_ptr + bo + 1, mask=msk, other=0.0).to(tl.float32)
        c2 = tl.load(b_ptr + bo + 2, mask=msk, other=0.0).to(tl.float32)
        c3 = tl.load(b_ptr + bo + 3, mask=msk, other=0.0).to(tl.float32)
        (
            p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, y0,
            y1, y2, y3,
        ) = tl.associative_scan(
            (
                a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15,
                c0, c1, c2, c3,
            ), 0, _combine4,
        )
        # fold in the state carried from earlier slabs
        y0 = y0 + p0 * h0 + p1 * h1 + p2 * h2 + p3 * h3
        y1 = y1 + p4 * h0 + p5 * h1 + p6 * h2 + p7 * h3
        y2 = y2 + p8 * h0 + p9 * h1 + p10 * h2 + p11 * h3
        y3 = y3 + p12 * h0 + p13 * h1 + p14 * h2 + p15 * h3
        tl.store(y_ptr + bo + 0, y0.to(y_ptr.dtype.element_ty), mask=msk)
        tl.store(y_ptr + bo + 1, y1.to(y_ptr.dtype.element_ty), mask=msk)
        tl.store(y_ptr + bo + 2, y2.to(y_ptr.dtype.element_ty), mask=msk)
        tl.store(y_ptr + bo + 3, y3.to(y_ptr.dtype.element_ty), mask=msk)
        last = s == SLAB - 1
        h0 = tl.sum(tl.where(last, y0, 0.0), 0)
        h1 = tl.sum(tl.where(last, y1, 0.0), 0)
        h2 = tl.sum(tl.where(last, y2, 0.0), 0)
        h3 = tl.sum(tl.where(last, y3, 0.0), 0)

_KERNEL_FOR = {2: _slab_scan2_kernel, 4: _slab_scan4_kernel}
assert set(_KERNEL_FOR) == set(SLAB_BLOCKS)


def slab_affine_scan(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Affine scan for D in {2, 4}. A: (BB, T, D, D), b: (BB, T, D) -> (BB, T, D)."""
    BB, T, D, _ = A.shape
    if D not in _KERNEL_FOR:
        raise ValueError(f"slab_affine_scan supports D in {sorted(_KERNEL_FOR)}, got {D}")
    # flat offsets in the kernels assume the standard contiguous layout
    A = A.contiguous()
    b = b.contiguous()
    y = torch.empty_like(b)
    slab, warps = slab_params(D, BB, T)
    _KERNEL_FOR[D][(BB,)](A, b, y, T, SLAB=slab, num_warps=warps)
    return y
