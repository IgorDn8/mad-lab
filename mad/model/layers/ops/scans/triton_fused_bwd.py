"""Fused backward for the triton affine scans.

The baseline backward (``_TritonAffineScan.backward``) expresses the reverse
adjoint scan as a *forward* scan over flipped time, which forces it to
materialise the flipped/transposed transition tensor and then flip the result
back::

    AT_flip = flip(A^T)                            # full (BB,T,D,D) copy
    C = empty_like(A); C[:,1:] = AT_flip[:,:-1]     # second full copy
    grad_y_flip = flip(grad_y)                      # copy
    s = flip(kernel(C, grad_y_flip))                # copy
    y_prev = shift(y); grad_A = s (outer) y_prev    # copy + full write

Profiled on an H100, that bookkeeping dominates: at ``BB=7784, T=16384, D=1``
it is 2.30ms of a 2.82ms backward (82%), and at ``BB=104, T=16384, D=16`` it is
3.9ms of 6.4ms (61%) -- each of those tensors is 0.5-1.7GB.

None of it is necessary. Transposing ``A`` is a stride swap *inside* the kernel
and reversing time is an index mapping, so the kernels here walk the adjoint
recurrence directly against the original ``A`` and emit ``grad_b`` and
``grad_A`` in the same pass.

Indexing: with reversed time ``tau = T-1-t`` the adjoint
``s_t = grad_y_t + A_{t+1}^T s_{t+1}`` becomes a plain affine scan

    s'_tau = C_tau s'_{tau-1} + g'_tau,
    C_tau  = A[T-tau]^T   (tau >= 1; tau = 0 has no transition -> identity),
    g'_tau = grad_y[T-1-tau],

so the kernels read ``A`` at ``T-tau`` with the last two strides swapped (the
transpose) and write every output back at the forward index ``t = T-1-tau``.

Parallelism mirrors the forward path: the scalar (``D == 1``) case resolves a
slab of timesteps with ``tl.associative_scan`` in log depth and needs no time
chunking, while the matrix case has no such primitive and so buys parallelism
with a chunk -> relay -> rescan grid when rows are scarce.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .triton_auto_scan import (
    MATRIX_CHUNK,
    SCALAR_SLAB,
    _cdiv,
    _next_pow2,
    _relay_kernel,
)
from .triton_slab_scan import SLAB_BLOCKS, _combine2, _combine4


# Rows up to which the matrix adjoint chunks the time axis. Much higher than the
# forward's MATRIX_ROW_LIMIT (256): the reverse loop is latency-bound on its
# serial dependency chain, so shortening the chain keeps paying well past the
# point where rows alone fill the GPU. Measured at D=16, BB=104, T=16384:
# num_chunks 1 -> 0.30x vs baseline, 128 (chunk length 128) -> 1.58x.
BWD_ROW_LIMIT = 2048


def fused_backward_supported(block: int) -> bool:
    """Whether the fused reverse path beats the materialising baseline.

    The baseline pays for its flipped/transposed copies but then gets to run the
    *best* forward kernel on them, so the fused path only wins where it has a
    matching reverse kernel. Measured on an H100 in bf16 at the iso1m shapes:

      D == 1      3.9x  -- reverse slab + tl.associative_scan, log depth
      D in {2,4}  4.6x / 2.5x -- reverse slab kernels (``_slab_bwd*_kernel``)
      D == 8      0.90x -- the baseline's reverse scan dispatches to the
                  *persistent* kernel (PERSISTENT_MAX_BLOCK == 8), which beats
                  the per-step reverse loop even before its copies are counted.
                  Batching loads the way persistent does was tried and is worse
                  still (see ``_matrix_bwd_kernel``), so D=8 keeps the baseline.
      D >= 16     1.7x  -- baseline uses the same chunked/sequential matrix
                  kernel, so skipping the copies is a clean win
    """
    return block != 8


def _bwd_num_chunks(*, rows: int, seq_len: int) -> int:
    """Time chunks for the matrix adjoint scan.

    There is no log-depth kernel at any ``D > 1``, so the time axis is the only
    source of extra parallelism -- and it also shortens the serial dependency
    chain each program walks, which is what the reverse loop is bound by.
    """
    if rows > BWD_ROW_LIMIT:
        return 1
    return max(1, seq_len // MATRIX_CHUNK)


@triton.jit
def _affine_combine(a_l, b_l, a_r, b_r):
    """Compose two affine maps: apply the left one, then the right one."""
    return a_l * a_r, a_r * b_l + b_r


# --------------------------------------------------------------------------- #
# scalar path (D == 1): coalesced reversed slabs + tl.associative_scan
# --------------------------------------------------------------------------- #
@triton.jit
def _scalar_bwd_kernel(
    a_ptr, g_ptr, y_ptr, gb_ptr, ga_ptr,
    sa0, sa1,
    sg0, sg1,
    sy0, sy1,
    sgb0, sgb1,
    sga0, sga1,
    T,
    SLAB: tl.constexpr,
):
    """One program per row; walks reversed time and emits both gradients."""
    row = tl.program_id(0).to(tl.int64)
    off = tl.arange(0, SLAB)
    h = 0.0

    for tau0 in range(0, T, SLAB):
        tau = tau0 + off
        m = tau < T
        t_fwd = T - 1 - tau

        # C_tau = a[T-tau]; tau == 0 carries no transition, and masked lanes
        # past the end are identity-padded (a=1, b=0) so the scan stays exact
        # and lane SLAB-1 still holds the slab's running state.
        a = tl.load(a_ptr + row * sa0 + (T - tau).to(tl.int64) * sa1,
                    mask=m & (tau > 0), other=1.0).to(tl.float32)
        g = tl.load(g_ptr + row * sg0 + t_fwd.to(tl.int64) * sg1,
                    mask=m, other=0.0).to(tl.float32)

        P, S = tl.associative_scan((a, g), 0, _affine_combine)
        S = S + P * h

        tl.store(gb_ptr + row * sgb0 + t_fwd.to(tl.int64) * sgb1,
                 S.to(gb_ptr.dtype.element_ty), mask=m)
        # grad_A_t = s_t * y_{t-1}, with y_{-1} = 0
        y_prev = tl.load(y_ptr + row * sy0 + (t_fwd - 1).to(tl.int64) * sy1,
                         mask=m & (t_fwd > 0), other=0.0).to(tl.float32)
        tl.store(ga_ptr + row * sga0 + t_fwd.to(tl.int64) * sga1,
                 (S * y_prev).to(ga_ptr.dtype.element_ty), mask=m)

        h = tl.sum(tl.where(off == SLAB - 1, S, 0.0), axis=0)


# --------------------------------------------------------------------------- #
# slab path (D in SLAB_BLOCKS): log-depth reversed slabs + tl.associative_scan
#
# Mirrors triton_slab_scan's forward kernels -- same combine functions, same flat
# contiguous offsets, one program per row, no time chunking (rows are plentiful
# at these widths). The reversal is an index mapping and the transpose is a
# permutation of which flat offset each entry reads, so C is never materialised.
# --------------------------------------------------------------------------- #
@triton.jit
def _slab_bwd2_kernel(A_ptr, g_ptr, y_ptr, gb_ptr, ga_ptr, T, SLAB: tl.constexpr):
    """D=2 adjoint. C_tau = A[T-tau]^T, i.e. entries (0,1) and (1,0) swapped."""
    row = tl.program_id(0).to(tl.int64)
    s_off = tl.arange(0, SLAB)
    rA = row * T * 4
    rb = row * T * 2
    h0 = 0.0
    h1 = 0.0

    for tau0 in range(0, T, SLAB):
        tau = tau0 + s_off
        m = tau < T
        t_fwd = T - 1 - tau
        # tau == 0 carries no transition; masked lanes take the identity map so
        # the scan stays exact past the slab end
        mt = m & (tau > 0)
        ao = rA + (T - tau).to(tl.int64) * 4
        c0 = tl.load(A_ptr + ao + 0, mask=mt, other=1.0).to(tl.float32)
        c1 = tl.load(A_ptr + ao + 2, mask=mt, other=0.0).to(tl.float32)
        c2 = tl.load(A_ptr + ao + 1, mask=mt, other=0.0).to(tl.float32)
        c3 = tl.load(A_ptr + ao + 3, mask=mt, other=1.0).to(tl.float32)

        go = rb + t_fwd.to(tl.int64) * 2
        g0 = tl.load(g_ptr + go + 0, mask=m, other=0.0).to(tl.float32)
        g1 = tl.load(g_ptr + go + 1, mask=m, other=0.0).to(tl.float32)

        p0, p1, p2, p3, s0, s1 = tl.associative_scan(
            (c0, c1, c2, c3, g0, g1), 0, _combine2)
        s0 = s0 + p0 * h0 + p1 * h1
        s1 = s1 + p2 * h0 + p3 * h1

        tl.store(gb_ptr + go + 0, s0.to(gb_ptr.dtype.element_ty), mask=m)
        tl.store(gb_ptr + go + 1, s1.to(gb_ptr.dtype.element_ty), mask=m)

        # grad_A_t = s_t (outer) y_{t-1}, with y_{-1} = 0
        mp = m & (t_fwd > 0)
        yo = rb + (t_fwd - 1).to(tl.int64) * 2
        q0 = tl.load(y_ptr + yo + 0, mask=mp, other=0.0).to(tl.float32)
        q1 = tl.load(y_ptr + yo + 1, mask=mp, other=0.0).to(tl.float32)
        gao = rA + t_fwd.to(tl.int64) * 4
        tl.store(ga_ptr + gao + 0, (s0 * q0).to(ga_ptr.dtype.element_ty), mask=m)
        tl.store(ga_ptr + gao + 1, (s0 * q1).to(ga_ptr.dtype.element_ty), mask=m)
        tl.store(ga_ptr + gao + 2, (s1 * q0).to(ga_ptr.dtype.element_ty), mask=m)
        tl.store(ga_ptr + gao + 3, (s1 * q1).to(ga_ptr.dtype.element_ty), mask=m)

        last = s_off == SLAB - 1
        h0 = tl.sum(tl.where(last, s0, 0.0), 0)
        h1 = tl.sum(tl.where(last, s1, 0.0), 0)


@triton.jit
def _slab_bwd4_kernel(A_ptr, g_ptr, y_ptr, gb_ptr, ga_ptr, T, SLAB: tl.constexpr):
    """D=4 adjoint. C_tau = A[T-tau]^T, i.e. entry (i,j) reads flat offset j*4+i."""
    row = tl.program_id(0).to(tl.int64)
    s_off = tl.arange(0, SLAB)
    rA = row * T * 16
    rb = row * T * 4
    h0 = 0.0
    h1 = 0.0
    h2 = 0.0
    h3 = 0.0

    for tau0 in range(0, T, SLAB):
        tau = tau0 + s_off
        m = tau < T
        t_fwd = T - 1 - tau
        mt = m & (tau > 0)
        ao = rA + (T - tau).to(tl.int64) * 16
        c0 = tl.load(A_ptr + ao + 0, mask=mt, other=1.0).to(tl.float32)
        c1 = tl.load(A_ptr + ao + 4, mask=mt, other=0.0).to(tl.float32)
        c2 = tl.load(A_ptr + ao + 8, mask=mt, other=0.0).to(tl.float32)
        c3 = tl.load(A_ptr + ao + 12, mask=mt, other=0.0).to(tl.float32)
        c4 = tl.load(A_ptr + ao + 1, mask=mt, other=0.0).to(tl.float32)
        c5 = tl.load(A_ptr + ao + 5, mask=mt, other=1.0).to(tl.float32)
        c6 = tl.load(A_ptr + ao + 9, mask=mt, other=0.0).to(tl.float32)
        c7 = tl.load(A_ptr + ao + 13, mask=mt, other=0.0).to(tl.float32)
        c8 = tl.load(A_ptr + ao + 2, mask=mt, other=0.0).to(tl.float32)
        c9 = tl.load(A_ptr + ao + 6, mask=mt, other=0.0).to(tl.float32)
        c10 = tl.load(A_ptr + ao + 10, mask=mt, other=1.0).to(tl.float32)
        c11 = tl.load(A_ptr + ao + 14, mask=mt, other=0.0).to(tl.float32)
        c12 = tl.load(A_ptr + ao + 3, mask=mt, other=0.0).to(tl.float32)
        c13 = tl.load(A_ptr + ao + 7, mask=mt, other=0.0).to(tl.float32)
        c14 = tl.load(A_ptr + ao + 11, mask=mt, other=0.0).to(tl.float32)
        c15 = tl.load(A_ptr + ao + 15, mask=mt, other=1.0).to(tl.float32)

        go = rb + t_fwd.to(tl.int64) * 4
        g0 = tl.load(g_ptr + go + 0, mask=m, other=0.0).to(tl.float32)
        g1 = tl.load(g_ptr + go + 1, mask=m, other=0.0).to(tl.float32)
        g2 = tl.load(g_ptr + go + 2, mask=m, other=0.0).to(tl.float32)
        g3 = tl.load(g_ptr + go + 3, mask=m, other=0.0).to(tl.float32)

        (
            p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
            s0, s1, s2, s3,
        ) = tl.associative_scan(
            (
                c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15,
                g0, g1, g2, g3,
            ), 0, _combine4,
        )
        s0 = s0 + p0 * h0 + p1 * h1 + p2 * h2 + p3 * h3
        s1 = s1 + p4 * h0 + p5 * h1 + p6 * h2 + p7 * h3
        s2 = s2 + p8 * h0 + p9 * h1 + p10 * h2 + p11 * h3
        s3 = s3 + p12 * h0 + p13 * h1 + p14 * h2 + p15 * h3

        tl.store(gb_ptr + go + 0, s0.to(gb_ptr.dtype.element_ty), mask=m)
        tl.store(gb_ptr + go + 1, s1.to(gb_ptr.dtype.element_ty), mask=m)
        tl.store(gb_ptr + go + 2, s2.to(gb_ptr.dtype.element_ty), mask=m)
        tl.store(gb_ptr + go + 3, s3.to(gb_ptr.dtype.element_ty), mask=m)

        mp = m & (t_fwd > 0)
        yo = rb + (t_fwd - 1).to(tl.int64) * 4
        q0 = tl.load(y_ptr + yo + 0, mask=mp, other=0.0).to(tl.float32)
        q1 = tl.load(y_ptr + yo + 1, mask=mp, other=0.0).to(tl.float32)
        q2 = tl.load(y_ptr + yo + 2, mask=mp, other=0.0).to(tl.float32)
        q3 = tl.load(y_ptr + yo + 3, mask=mp, other=0.0).to(tl.float32)
        gao = rA + t_fwd.to(tl.int64) * 16
        et = ga_ptr.dtype.element_ty
        tl.store(ga_ptr + gao + 0, (s0 * q0).to(et), mask=m)
        tl.store(ga_ptr + gao + 1, (s0 * q1).to(et), mask=m)
        tl.store(ga_ptr + gao + 2, (s0 * q2).to(et), mask=m)
        tl.store(ga_ptr + gao + 3, (s0 * q3).to(et), mask=m)
        tl.store(ga_ptr + gao + 4, (s1 * q0).to(et), mask=m)
        tl.store(ga_ptr + gao + 5, (s1 * q1).to(et), mask=m)
        tl.store(ga_ptr + gao + 6, (s1 * q2).to(et), mask=m)
        tl.store(ga_ptr + gao + 7, (s1 * q3).to(et), mask=m)
        tl.store(ga_ptr + gao + 8, (s2 * q0).to(et), mask=m)
        tl.store(ga_ptr + gao + 9, (s2 * q1).to(et), mask=m)
        tl.store(ga_ptr + gao + 10, (s2 * q2).to(et), mask=m)
        tl.store(ga_ptr + gao + 11, (s2 * q3).to(et), mask=m)
        tl.store(ga_ptr + gao + 12, (s3 * q0).to(et), mask=m)
        tl.store(ga_ptr + gao + 13, (s3 * q1).to(et), mask=m)
        tl.store(ga_ptr + gao + 14, (s3 * q2).to(et), mask=m)
        tl.store(ga_ptr + gao + 15, (s3 * q3).to(et), mask=m)

        last = s_off == SLAB - 1
        h0 = tl.sum(tl.where(last, s0, 0.0), 0)
        h1 = tl.sum(tl.where(last, s1, 0.0), 0)
        h2 = tl.sum(tl.where(last, s2, 0.0), 0)
        h3 = tl.sum(tl.where(last, s3, 0.0), 0)


_SLAB_BWD_KERNEL = {2: _slab_bwd2_kernel, 4: _slab_bwd4_kernel}
assert set(_SLAB_BWD_KERNEL) == set(SLAB_BLOCKS)

# (slab length, warps) for the *adjoint*, tuned separately from the forward's
# ``slab_params``: emitting grad_A costs D^2 extra stores and live values per
# lane, so the reverse kernels want shorter slabs and narrower programs. On an
# H100 at the iso1m shapes, using the forward's parameters instead costs 1.3x
# (D=2, 512/2 warps) and 2.5x (D=4, 128/2 warps).
_SLAB_BWD_PARAMS = {2: (64, 1), 4: (32, 1)}


def _slab_fused_backward(A, y, grad_y):
    BB, T, D, _ = A.shape
    # flat offsets in the kernels assume the standard contiguous layout
    y = y.contiguous()
    grad_A = torch.empty_like(A)
    grad_b = torch.empty_like(grad_y)
    slab, warps = _SLAB_BWD_PARAMS[D]
    slab = max(16, min(slab, _next_pow2(T)))
    _SLAB_BWD_KERNEL[D][(BB,)](
        A, grad_y, y, grad_b, grad_A, T, SLAB=slab, num_warps=warps,
    )
    return grad_A, grad_b


# --------------------------------------------------------------------------- #
# matrix path (D > 1): per-step reversed loads, time-chunked grid
# --------------------------------------------------------------------------- #
@triton.jit
def _matrix_bwd_kernel(
    A_ptr, g_ptr, y_ptr, gb_ptr, ga_ptr,
    carry_A_ptr, carry_b_ptr, h_in_ptr,
    sa0, sa1, sa2, sa3,
    sg0, sg1, sg2,
    sy0, sy1, sy2,
    sgb0, sgb1, sgb2,
    sga0, sga1, sga2, sga3,
    sca0, sca1, sca2, sca3,
    scb0, scb1, scb2,
    T, CHUNK, D,
    BLOCK_D: tl.constexpr,
    SEEDED: tl.constexpr,
    EMIT_CARRY: tl.constexpr,
):
    """One program per (row, chunk) over reversed time.

    ``EMIT_CARRY``: write the chunk's (transition product, local end state)
    instead of gradients.  ``SEEDED``: start from the relayed incoming state.
    """
    row = tl.program_id(0).to(tl.int64)
    chunk = tl.program_id(1)
    tau_start = chunk * CHUNK
    tau_end = tl.minimum(tau_start + CHUNK, T)

    i_off = tl.arange(0, BLOCK_D)
    i_m = i_off < D
    m2 = i_m[:, None] & i_m[None, :]

    if SEEDED:
        s = tl.load(
            h_in_ptr + row * scb0 + chunk.to(tl.int64) * scb1
            + i_off.to(tl.int64) * scb2,
            mask=i_m, other=0.0,
        ).to(tl.float32)
    else:
        s = tl.zeros((BLOCK_D,), dtype=tl.float32)

    eye = (i_off[:, None] == i_off[None, :]).to(tl.float32)
    A_run = eye

    # Loads stay per-step. Batching CS_SUB timesteps into a register-resident
    # (CS_SUB, D, D) tile -- what the forward `persistent` kernel does -- was
    # tried and loses badly here, because extracting each step from the tile
    # costs a masked reduction over the stack: at D=16 it takes 1.71x down to
    # 0.62x (CS_SUB=2) and 0.14x (CS_SUB=4), and at D=8 0.90x down to 0.72x.
    for tau in range(tau_start, tau_end):
        t_fwd = T - 1 - tau
        # C_tau = A[T-tau]^T -- the transpose is just sa3/sa2 swapped below.
        has_transition = tau > 0
        C_t = tl.load(
            A_ptr + row * sa0 + (T - tau).to(tl.int64) * sa1
            + i_off[:, None].to(tl.int64) * sa3 + i_off[None, :].to(tl.int64) * sa2,
            mask=m2 & has_transition, other=0.0,
        ).to(tl.float32)
        C_t = tl.where(has_transition, C_t, eye)

        g = tl.load(
            g_ptr + row * sg0 + t_fwd.to(tl.int64) * sg1 + i_off.to(tl.int64) * sg2,
            mask=i_m, other=0.0,
        ).to(tl.float32)

        s = tl.sum(C_t * s[None, :], axis=1) + g

        if not EMIT_CARRY:
            tl.store(
                gb_ptr + row * sgb0 + t_fwd.to(tl.int64) * sgb1
                + i_off.to(tl.int64) * sgb2,
                s.to(gb_ptr.dtype.element_ty), mask=i_m,
            )
            y_prev = tl.load(
                y_ptr + row * sy0 + (t_fwd - 1).to(tl.int64) * sy1
                + i_off.to(tl.int64) * sy2,
                mask=i_m & (t_fwd > 0), other=0.0,
            ).to(tl.float32)
            tl.store(
                ga_ptr + row * sga0 + t_fwd.to(tl.int64) * sga1
                + i_off[:, None].to(tl.int64) * sga2
                + i_off[None, :].to(tl.int64) * sga3,
                (s[:, None] * y_prev[None, :]).to(ga_ptr.dtype.element_ty), mask=m2,
            )
        else:
            A_run = tl.sum(C_t[:, :, None] * A_run[None, :, :], axis=1)

    if EMIT_CARRY:
        tl.store(
            carry_A_ptr + row * sca0 + chunk.to(tl.int64) * sca1
            + i_off[:, None].to(tl.int64) * sca2 + i_off[None, :].to(tl.int64) * sca3,
            A_run, mask=m2,
        )
        tl.store(
            carry_b_ptr + row * scb0 + chunk.to(tl.int64) * scb1
            + i_off.to(tl.int64) * scb2,
            s, mask=i_m,
        )


def _scalar_fused_backward(A, y, grad_y):
    BB, T = A.shape[0], A.shape[1]
    a2, y2, g2 = A.reshape(BB, T), y.reshape(BB, T), grad_y.reshape(BB, T)
    grad_A = torch.empty_like(A)
    grad_b = torch.empty_like(grad_y)
    ga2, gb2 = grad_A.reshape(BB, T), grad_b.reshape(BB, T)

    _scalar_bwd_kernel[(BB,)](
        a2, g2, y2, gb2, ga2,
        a2.stride(0), a2.stride(1),
        g2.stride(0), g2.stride(1),
        y2.stride(0), y2.stride(1),
        gb2.stride(0), gb2.stride(1),
        ga2.stride(0), ga2.stride(1),
        T,
        SLAB=min(SCALAR_SLAB, _next_pow2(T)),
    )
    return grad_A, grad_b


def _matrix_fused_backward(A, y, grad_y, num_chunks: int):
    BB, T, D, _ = A.shape
    block_d = _next_pow2(D)
    grad_A = torch.empty_like(A)
    grad_b = torch.empty_like(grad_y)

    chunk = _cdiv(T, num_chunks)
    num_chunks = _cdiv(T, chunk)

    carry_A = carry_b = h_in = None
    if num_chunks > 1:
        carry_A = torch.empty(BB, num_chunks, D, D, device=A.device, dtype=torch.float32)
        carry_b = torch.empty(BB, num_chunks, D, device=A.device, dtype=torch.float32)
        h_in = torch.empty(BB, num_chunks, D, device=A.device, dtype=torch.float32)

    def _launch(seeded: bool, emit_carry: bool):
        ca = carry_A if carry_A is not None else A
        cb = carry_b if carry_b is not None else grad_b
        hi = h_in if h_in is not None else grad_b
        _matrix_bwd_kernel[(BB, num_chunks)](
            A, grad_y, y, grad_b, grad_A,
            ca, cb, hi,
            A.stride(0), A.stride(1), A.stride(2), A.stride(3),
            grad_y.stride(0), grad_y.stride(1), grad_y.stride(2),
            y.stride(0), y.stride(1), y.stride(2),
            grad_b.stride(0), grad_b.stride(1), grad_b.stride(2),
            grad_A.stride(0), grad_A.stride(1), grad_A.stride(2), grad_A.stride(3),
            ca.stride(0), ca.stride(1), ca.stride(2), ca.stride(3) if ca.dim() > 3 else 0,
            cb.stride(0), cb.stride(1), cb.stride(2),
            T, chunk, D,
            BLOCK_D=block_d, SEEDED=seeded, EMIT_CARRY=emit_carry,
        )

    if num_chunks > 1:
        _launch(seeded=False, emit_carry=True)
        # the carries are already in chunk order, so the forward path's relay
        # (exclusive prefix over chunks) applies unchanged
        _relay_kernel[(BB,)](
            carry_A, carry_b, h_in,
            carry_A.stride(0), carry_A.stride(1), carry_A.stride(2), carry_A.stride(3),
            carry_b.stride(0), carry_b.stride(1), carry_b.stride(2),
            num_chunks, D, BLOCK_D=block_d,
        )
    _launch(seeded=num_chunks > 1, emit_carry=False)
    return grad_A, grad_b


def fused_scan_backward(A: torch.Tensor, y: torch.Tensor, grad_y: torch.Tensor,
                        *, num_chunks: int | None = None):
    """Gradients of ``h_t = A_t h_{t-1} + b_t`` w.r.t. ``A`` and ``b``.

    A: (BB, T, D, D), y: (BB, T, D) (the forward output), grad_y: (BB, T, D).
    Returns ``(grad_A, grad_b)`` without materialising any transposed, flipped
    or shifted copy of ``A``.
    """
    BB, T, D, _ = A.shape
    grad_y = grad_y.contiguous()
    if D == 1:
        return _scalar_fused_backward(A, y, grad_y)
    if D in _SLAB_BWD_KERNEL:
        return _slab_fused_backward(A, y, grad_y)
    if num_chunks is None:
        num_chunks = _bwd_num_chunks(rows=BB, seq_len=T)
    num_chunks = max(1, min(num_chunks, T))
    return _matrix_fused_backward(A, y, grad_y, num_chunks)
