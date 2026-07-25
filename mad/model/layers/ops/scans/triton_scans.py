"""Triton affine-scan primitives (adapted from nisys-bench).

Solves the affine recurrence ``h_t = A_t @ h_{t-1} + b_t`` over a flattened
batch layout ``A: (BB, T, D, D)``, ``b: (BB, T, D)`` -> ``y: (BB, T, D)``.

Scan strategies exposed:

- ``sequential``      : one program per row, loops over time (O(T) depth).
- ``persistent``      : one program per row, state resident in registers, time
                        loaded in sub-chunks (single launch, O(T) depth).
- ``blelloch``        : work-efficient up-/down-sweep tree scan (O(T) work,
                        O(log T) depth). Vendored verbatim from nisys-bench's
                        ``triton_blelloch_scan.blelloch_affine_scan``.
- ``chunked``         : occupancy-aware scan from ``triton_auto_scan``. The
                        three kernels above expose only ``BB`` parallelism, so
                        they idle the GPU at small batch; this one resolves
                        contiguous time slabs with ``tl.associative_scan`` when
                        D == 1 and adds a time-chunk grid axis otherwise.
- ``auto``            : pick among the above from ``(BB, T, D)``.

Unlike nisys-bench (which fuses the ``W_t = einsum(L, x_t)`` generator into the
kernel), these operate on a *precomputed* transition tensor ``A`` — the form MAD
layers already produce from their gates — exactly like the ``affine_scan_torch``
option.

These forward kernels are not autograd-aware, so ``triton_affine_scan`` wraps
them in an autograd Function with a hand-written backward. The backward is the
exact adjoint of the affine scan and is itself an affine scan run in reverse, so
it is evaluated with the *same* triton kernel as the forward pass (keeping the
backward consistent with the chosen scan strategy).

Backward derivation (forward: ``h_t = A_t h_{t-1} + b_t``, ``y_t = h_t``,
``h_{-1}=0``). With ``s_t = dL/dh_t``:

    s_t      = grad_y_t + A_{t+1}^T s_{t+1}        (reverse affine scan, s_T = 0)
    grad_b_t = s_t
    grad_A_t = s_t (outer) y_{t-1}                 (y_{-1} = 0)

The ``s`` recurrence is a forward affine scan over time-flipped inputs with
transitions ``C_tau = A_{T-tau}^T`` (i.e. flipped, transposed and shifted by one
step), which we run through the same kernel and then flip back.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .affine_scan import scan_parallel as _torch_affine_scan_parallel
from .triton_auto_scan import chunked_affine_scan, select_scan_mode

# Triton 3.x tl.dot requires the K dimension to be >= 16 on recent GPUs.
TRITON_MIN_DOT_K = 16


def triton_dot_tile_size(dim: int, requested: int | None = None, max_tile: int = 64) -> int:
    """Pick a BLOCK tile size that satisfies tl.dot constraints and covers `dim`."""
    tile = requested if requested is not None else min(dim, max_tile)
    return max(tile, TRITON_MIN_DOT_K)


def _next_power_of_2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


@triton.jit
def _offset2(b, t, sb, st):
    """64-bit batch/time offset (int32 products overflow on large tensors)."""
    return b.to(tl.int64) * sb + t.to(tl.int64) * st


# --------------------------------------------------------------------------- #
# sequential: one program per row, loop over time.
# --------------------------------------------------------------------------- #
@triton.jit
def _sequential_scan_kernel(
    A_ptr, b_ptr, y_ptr,
    sa0, sa1, sa2, sa3,
    sb0, sb1, sb2,
    sy0, sy1, sy2,
    T, D,
    BLOCK_D: tl.constexpr,
):
    bb = tl.program_id(0)
    i_off = tl.arange(0, BLOCK_D)
    j_off = tl.arange(0, BLOCK_D)
    i_m = i_off < D
    j_m = j_off < D

    h = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for t in range(T):
        A_ij = tl.load(
            A_ptr + _offset2(bb, t, sa0, sa1)
            + i_off[:, None].to(tl.int64) * sa2
            + j_off[None, :].to(tl.int64) * sa3,
            mask=i_m[:, None] & j_m[None, :], other=0.0,
        ).to(tl.float32)
        b_i = tl.load(
            b_ptr + _offset2(bb, t, sb0, sb1) + i_off.to(tl.int64) * sb2,
            mask=i_m, other=0.0,
        ).to(tl.float32)
        h = tl.sum(A_ij * h[None, :], axis=1) + b_i
        tl.store(
            y_ptr + _offset2(bb, t, sy0, sy1) + i_off.to(tl.int64) * sy2,
            h.to(y_ptr.dtype.element_ty), mask=i_m,
        )


def sequential_affine_scan(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Sequential (single-launch, per-row loop) affine scan over dim 1."""
    BB, T, D, _ = A.shape
    block_d = _next_power_of_2(D)
    y = torch.empty(BB, T, D, device=b.device, dtype=b.dtype)
    _sequential_scan_kernel[(BB,)](
        A, b, y,
        A.stride(0), A.stride(1), A.stride(2), A.stride(3),
        b.stride(0), b.stride(1), b.stride(2),
        y.stride(0), y.stride(1), y.stride(2),
        T, D, BLOCK_D=block_d,
    )
    return y


# --------------------------------------------------------------------------- #
# persistent: state resident in registers, time streamed in sub-chunks.
# --------------------------------------------------------------------------- #
@triton.jit
def _persistent_scan_kernel(
    A_ptr, b_ptr, y_ptr,
    sa0, sa1, sa2, sa3,
    sb0, sb1, sb2,
    sy0, sy1, sy2,
    T, D,
    BLOCK_D: tl.constexpr,
    CS_SUB: tl.constexpr,
):
    bb = tl.program_id(0)
    i_off = tl.arange(0, BLOCK_D)
    j_off = tl.arange(0, BLOCK_D)
    s_off = tl.arange(0, CS_SUB)
    i_m = i_off < D
    j_m = j_off < D

    h = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for t0 in range(0, T, CS_SUB):
        t_idx = t0 + s_off
        t_m = t_idx < T
        # batch-load CS_SUB timesteps of A and b into registers
        A_stack = tl.load(
            A_ptr + bb.to(tl.int64) * sa0
            + t_idx[:, None, None].to(tl.int64) * sa1
            + i_off[None, :, None].to(tl.int64) * sa2
            + j_off[None, None, :].to(tl.int64) * sa3,
            mask=t_m[:, None, None] & i_m[None, :, None] & j_m[None, None, :], other=0.0,
        ).to(tl.float32)
        b_stack = tl.load(
            b_ptr + bb.to(tl.int64) * sb0
            + t_idx[:, None].to(tl.int64) * sb1
            + i_off[None, :].to(tl.int64) * sb2,
            mask=t_m[:, None] & i_m[None, :], other=0.0,
        ).to(tl.float32)

        for j in tl.static_range(CS_SUB):
            if t0 + j < T:
                A_j = tl.sum(tl.where(s_off[:, None, None] == j, A_stack, 0.0), axis=0)
                b_j = tl.sum(tl.where(s_off[:, None] == j, b_stack, 0.0), axis=0)
                h = tl.sum(A_j * h[None, :], axis=1) + b_j
                tl.store(
                    y_ptr + _offset2(bb, t0 + j, sy0, sy1) + i_off.to(tl.int64) * sy2,
                    h.to(y_ptr.dtype.element_ty), mask=i_m,
                )


def persistent_affine_scan(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Persistent (register-resident state, sub-chunked time) affine scan."""
    BB, T, D, _ = A.shape
    block_d = _next_power_of_2(D)
    # keep the register-resident A sub-chunk small; shrink for wider blocks.
    cs_sub = 8 if D <= 4 else (4 if D <= 8 else (2 if D <= 16 else 1))
    y = torch.empty(BB, T, D, device=b.device, dtype=b.dtype)
    _persistent_scan_kernel[(BB,)](
        A, b, y,
        A.stride(0), A.stride(1), A.stride(2), A.stride(3),
        b.stride(0), b.stride(1), b.stride(2),
        y.stride(0), y.stride(1), y.stride(2),
        T, D, BLOCK_D=block_d, CS_SUB=cs_sub,
    )
    return y


# --------------------------------------------------------------------------- #
# blelloch: work-efficient tree scan (vendored from nisys-bench).
# --------------------------------------------------------------------------- #
@triton.jit
def _blelloch_combine_kernel(
    A_ptr, b_ptr, A_scratch_ptr, b_scratch_ptr,
    sa0, sa1, sa2, sa3,
    sb0, sb1, sb2,
    ssa0, ssa1, ssa2, ssa3,
    ssb0, ssb1, ssb2,
    span, D,
    IS_UP: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    node = tl.program_id(1)
    pos_l = 2 * span * node + span - 1
    pos_r = 2 * span * node + 2 * span - 1

    if IS_UP:
        t_new = pos_r
        t_old = pos_l
    else:
        t_new = pos_l
        t_old = pos_r

    for i_start in range(0, D, BLOCK_D):
        i_off = i_start + tl.arange(0, BLOCK_D)
        i_m = i_off < D

        b_acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for k_start in range(0, D, BLOCK_D):
            k_off = k_start + tl.arange(0, BLOCK_D)
            k_m = k_off < D
            A_ik = tl.load(
                A_ptr + _offset2(batch_idx, t_new, sa0, sa1)
                + i_off[:, None].to(tl.int64) * sa2
                + k_off[None, :].to(tl.int64) * sa3,
                mask=i_m[:, None] & k_m[None, :], other=0.0,
            ).to(tl.float32)
            b_k = tl.load(
                b_ptr + _offset2(batch_idx, t_old, sb0, sb1) + k_off.to(tl.int64) * sb2,
                mask=k_m, other=0.0,
            ).to(tl.float32)
            b_acc += tl.sum(A_ik * b_k[None, :], axis=1)
        b_new_i = tl.load(
            b_ptr + _offset2(batch_idx, t_new, sb0, sb1) + i_off.to(tl.int64) * sb2,
            mask=i_m, other=0.0,
        ).to(tl.float32)
        tl.store(
            b_scratch_ptr + _offset2(batch_idx, node, ssb0, ssb1) + i_off.to(tl.int64) * ssb2,
            (b_acc + b_new_i).to(b_scratch_ptr.dtype.element_ty), mask=i_m,
        )

        for j_start in range(0, D, BLOCK_D):
            j_off = j_start + tl.arange(0, BLOCK_D)
            j_m = j_off < D
            acc = tl.zeros((BLOCK_D, BLOCK_D), dtype=tl.float32)
            for k_start in range(0, D, BLOCK_D):
                k_off = k_start + tl.arange(0, BLOCK_D)
                k_m = k_off < D
                A_ik = tl.load(
                    A_ptr + _offset2(batch_idx, t_new, sa0, sa1)
                    + i_off[:, None].to(tl.int64) * sa2
                    + k_off[None, :].to(tl.int64) * sa3,
                    mask=i_m[:, None] & k_m[None, :], other=0.0,
                ).to(tl.float32)
                A_kj = tl.load(
                    A_ptr + _offset2(batch_idx, t_old, sa0, sa1)
                    + k_off[:, None].to(tl.int64) * sa2
                    + j_off[None, :].to(tl.int64) * sa3,
                    mask=k_m[:, None] & j_m[None, :], other=0.0,
                ).to(tl.float32)
                acc += tl.dot(A_ik, A_kj)
            tl.store(
                A_scratch_ptr + _offset2(batch_idx, node, ssa0, ssa1)
                + i_off[:, None].to(tl.int64) * ssa2
                + j_off[None, :].to(tl.int64) * ssa3,
                acc.to(A_scratch_ptr.dtype.element_ty),
                mask=i_m[:, None] & j_m[None, :],
            )


@triton.jit
def _blelloch_writeback_kernel(
    A_ptr, b_ptr, A_scratch_ptr, b_scratch_ptr,
    sa0, sa1, sa2, sa3,
    sb0, sb1, sb2,
    ssa0, ssa1, ssa2, ssa3,
    ssb0, ssb1, ssb2,
    span, D,
    IS_UP: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    node = tl.program_id(1)
    pos_l = 2 * span * node + span - 1
    pos_r = 2 * span * node + 2 * span - 1

    for i_start in range(0, D, BLOCK_D):
        i_off = i_start + tl.arange(0, BLOCK_D)
        i_m = i_off < D
        for j_start in range(0, D, BLOCK_D):
            j_off = j_start + tl.arange(0, BLOCK_D)
            j_m = j_off < D
            m2 = i_m[:, None] & j_m[None, :]
            if not IS_UP:
                prefix = tl.load(
                    A_ptr + _offset2(batch_idx, pos_r, sa0, sa1)
                    + i_off[:, None].to(tl.int64) * sa2
                    + j_off[None, :].to(tl.int64) * sa3,
                    mask=m2, other=0.0,
                )
                tl.store(
                    A_ptr + _offset2(batch_idx, pos_l, sa0, sa1)
                    + i_off[:, None].to(tl.int64) * sa2
                    + j_off[None, :].to(tl.int64) * sa3,
                    prefix, mask=m2,
                )
            val = tl.load(
                A_scratch_ptr + _offset2(batch_idx, node, ssa0, ssa1)
                + i_off[:, None].to(tl.int64) * ssa2
                + j_off[None, :].to(tl.int64) * ssa3,
                mask=m2, other=0.0,
            )
            tl.store(
                A_ptr + _offset2(batch_idx, pos_r, sa0, sa1)
                + i_off[:, None].to(tl.int64) * sa2
                + j_off[None, :].to(tl.int64) * sa3,
                val, mask=m2,
            )

        if not IS_UP:
            b_prefix = tl.load(
                b_ptr + _offset2(batch_idx, pos_r, sb0, sb1) + i_off.to(tl.int64) * sb2,
                mask=i_m, other=0.0,
            )
            tl.store(
                b_ptr + _offset2(batch_idx, pos_l, sb0, sb1) + i_off.to(tl.int64) * sb2,
                b_prefix, mask=i_m,
            )
        b_val = tl.load(
            b_scratch_ptr + _offset2(batch_idx, node, ssb0, ssb1) + i_off.to(tl.int64) * ssb2,
            mask=i_m, other=0.0,
        )
        tl.store(
            b_ptr + _offset2(batch_idx, pos_r, sb0, sb1) + i_off.to(tl.int64) * sb2,
            b_val, mask=i_m,
        )


@triton.jit
def _blelloch_finalize_kernel(
    A_orig_ptr, b_orig_ptr, b_prefix_ptr, y_ptr,
    sa0, sa1, sa2, sa3,
    sb0, sb1, sb2,
    sp0, sp1, sp2,
    sy0, sy1, sy2,
    T, D,
    BLOCK_D: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    t = tl.program_id(1)
    if t >= T:
        return

    for i_start in range(0, D, BLOCK_D):
        i_off = i_start + tl.arange(0, BLOCK_D)
        i_m = i_off < D
        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for k_start in range(0, D, BLOCK_D):
            k_off = k_start + tl.arange(0, BLOCK_D)
            k_m = k_off < D
            A_ik = tl.load(
                A_orig_ptr + _offset2(batch_idx, t, sa0, sa1)
                + i_off[:, None].to(tl.int64) * sa2
                + k_off[None, :].to(tl.int64) * sa3,
                mask=i_m[:, None] & k_m[None, :], other=0.0,
            ).to(tl.float32)
            p_k = tl.load(
                b_prefix_ptr + _offset2(batch_idx, t, sp0, sp1) + k_off.to(tl.int64) * sp2,
                mask=k_m, other=0.0,
            ).to(tl.float32)
            acc += tl.sum(A_ik * p_k[None, :], axis=1)
        b_i = tl.load(
            b_orig_ptr + _offset2(batch_idx, t, sb0, sb1) + i_off.to(tl.int64) * sb2,
            mask=i_m, other=0.0,
        ).to(tl.float32)
        tl.store(
            y_ptr + _offset2(batch_idx, t, sy0, sy1) + i_off.to(tl.int64) * sy2,
            (acc + b_i).to(y_ptr.dtype.element_ty), mask=i_m,
        )


def blelloch_affine_scan(A: torch.Tensor, b: torch.Tensor, block_d: int = 64) -> torch.Tensor:
    """Inclusive affine scan over dim 1. A: (BB, T, D, D), b: (BB, T, D)."""
    BB, T, D, _ = A.shape
    block_d = triton_dot_tile_size(D, requested=min(block_d, D))
    T_pad = _next_power_of_2(T)

    if T_pad != T:
        eye = torch.eye(D, device=A.device, dtype=A.dtype)
        A_full = torch.empty(BB, T_pad, D, D, device=A.device, dtype=A.dtype)
        A_full[:, :T] = A
        A_full[:, T:] = eye
        b_full = torch.zeros(BB, T_pad, D, device=b.device, dtype=b.dtype)
        b_full[:, :T] = b
    else:
        A_full, b_full = A, b

    A_work = A_full.clone()
    b_work = b_full.clone()
    n_scratch = max(T_pad // 2, 1)
    A_scratch = torch.empty(BB, n_scratch, D, D, device=A.device, dtype=A.dtype)
    b_scratch = torch.empty(BB, n_scratch, D, device=b.device, dtype=b.dtype)

    def _level(span: int, is_up: bool) -> None:
        n_nodes = T_pad // (2 * span)
        args = (
            A_work, b_work, A_scratch, b_scratch,
            A_work.stride(0), A_work.stride(1), A_work.stride(2), A_work.stride(3),
            b_work.stride(0), b_work.stride(1), b_work.stride(2),
            A_scratch.stride(0), A_scratch.stride(1), A_scratch.stride(2), A_scratch.stride(3),
            b_scratch.stride(0), b_scratch.stride(1), b_scratch.stride(2),
            span, D,
        )
        _blelloch_combine_kernel[(BB, n_nodes)](*args, IS_UP=is_up, BLOCK_D=block_d)
        _blelloch_writeback_kernel[(BB, n_nodes)](*args, IS_UP=is_up, BLOCK_D=block_d)

    span = 1
    while span < T_pad:
        _level(span, is_up=True)
        span <<= 1

    A_work[:, T_pad - 1] = torch.eye(D, device=A.device, dtype=A.dtype)
    b_work[:, T_pad - 1] = 0

    span = T_pad // 2
    while span >= 1:
        _level(span, is_up=False)
        span >>= 1

    y = torch.empty(BB, T, D, device=b.device, dtype=b.dtype)
    _blelloch_finalize_kernel[(BB, T)](
        A_full, b_full, b_work, y,
        A_full.stride(0), A_full.stride(1), A_full.stride(2), A_full.stride(3),
        b_full.stride(0), b_full.stride(1), b_full.stride(2),
        b_work.stride(0), b_work.stride(1), b_work.stride(2),
        y.stride(0), y.stride(1), y.stride(2),
        T, D, BLOCK_D=block_d,
    )
    return y


# --------------------------------------------------------------------------- #
# autograd wrapper: triton forward, associative-scan (torch) backward.
# --------------------------------------------------------------------------- #
_KERNELS = {
    "sequential": sequential_affine_scan,
    "persistent": persistent_affine_scan,
    "blelloch": blelloch_affine_scan,
    "chunked": chunked_affine_scan,
}


def auto_affine_scan(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pick the fastest scan kernel for this shape (see ``select_scan_mode``)."""
    BB, T, D, _ = A.shape
    mode = select_scan_mode(rows=BB, seq_len=T, block=D)
    return _KERNELS[mode](A, b)


_KERNELS["auto"] = auto_affine_scan


def _reference_scan(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Differentiable reference over (BB, T, D, D), (BB, T, D) via torch scan.

    Used only by the test-suite as an autograd ground truth; the production
    backward below runs on the triton kernels instead.
    """
    y = _torch_affine_scan_parallel(A.unsqueeze(2), b.unsqueeze(2))  # (BB, T, 1, D)
    return y.squeeze(2)


class _TritonAffineScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, b, mode):
        A = A.contiguous()
        b = b.contiguous()
        with torch.no_grad():
            y = _KERNELS[mode](A, b)
        ctx.save_for_backward(A, y)
        ctx.mode = mode
        return y

    @staticmethod
    def backward(ctx, grad_y):
        A, y = ctx.saved_tensors
        mode = ctx.mode
        BB, T, D, _ = A.shape

        # Reverse adjoint scan: s_t = grad_y_t + A_{t+1}^T s_{t+1}.
        # Rewritten as a forward scan on flipped time with transitions
        # C_tau = A_{T-tau}^T  =>  C[0] = I, C[1:] = flip(A^T)[:-1].
        AT_flip = torch.flip(A.transpose(-1, -2), dims=[1])
        C = torch.empty_like(A)
        C[:, 0] = torch.eye(D, device=A.device, dtype=A.dtype)
        if T > 1:
            C[:, 1:] = AT_flip[:, :-1]
        grad_y_flip = torch.flip(grad_y, dims=[1]).contiguous()

        with torch.no_grad():
            s_flip = _KERNELS[mode](C.contiguous(), grad_y_flip)
        s = torch.flip(s_flip, dims=[1])  # dL/dh_t

        grad_b = s
        # grad_A_t = s_t (outer) y_{t-1}, with y_{-1} = 0.
        y_prev = torch.zeros_like(y)
        if T > 1:
            y_prev[:, 1:] = y[:, :-1]
        grad_A = s.unsqueeze(-1) * y_prev.unsqueeze(-2)
        return grad_A, grad_b, None


def triton_affine_scan(A: torch.Tensor, b: torch.Tensor, mode: str) -> torch.Tensor:
    """Differentiable affine scan; forward uses the triton `mode` kernel.

    A: (BB, T, D, D), b: (BB, T, D) -> y: (BB, T, D), solving
    h_t = A_t @ h_{t-1} + b_t (inclusive).
    """
    if mode not in _KERNELS:
        raise ValueError(f"unknown triton scan mode: {mode!r} (choose from {list(_KERNELS)})")
    return _TritonAffineScan.apply(A, b, mode)
