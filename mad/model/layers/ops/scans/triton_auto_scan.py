"""Occupancy-aware chunked affine scans, and the ``auto`` dispatcher.

The ``sequential`` / ``persistent`` / ``blelloch`` kernels in ``triton_scans``
all use a grid of one program per row, so the only parallelism they expose is
``BB = batch * hidden_dim``. At small batch that leaves the GPU mostly idle: on
an H100, ``m=1, BB=128, T=8192`` runs at ~18 GB/s of ~3350 GB/s available.

Since the layers store ``A`` as ``(BB, T, D, D)``, *time* is the contiguous axis,
and the fix depends on how wide the block is:

- ``D == 1``: the recurrence is ``h_t = a_t h_{t-1} + b_t``, so a program loads a
  coalesced slab of timesteps and resolves it with ``tl.associative_scan`` in log
  depth. No time chunking needed at any row count measured (down to 8 rows).
- ``D in {2, 4}``: same idea, but every entry of ``A`` and ``b`` must ride along
  as a separate scan input; see ``triton_slab_scan``, which also documents why
  D=8 is the cutoff.
- otherwise:  no log-depth option, so this keeps per-step loads (a row's D×D
  block is contiguous, so those are already coalesced) and buys parallelism with
  a time-chunk grid axis, structured as chunk -> relay -> rescan. It degenerates
  to a single pass once ``BB`` alone fills the GPU, keeping traffic optimal.

Chunk carries are combined by a sequential relay rather than a Blelloch tree:
the carry array is ``NC`` entries per row where ``NC`` is at most a few hundred,
and at that size the O(log NC) global round trips cost more than the serial
walk, whose loads do not depend on the carried state and so pipeline.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .triton_slab_scan import SLAB_BLOCKS, slab_affine_scan

# Timesteps a scalar-path program resolves per associative_scan.
SCALAR_SLAB = 1024
# Chunk length the matrix path targets when it does chunk.
MATRIX_CHUNK = 128
# Rows above which the matrix path already fills the GPU, so chunking (which
# costs a second read of A plus a D^3 transition product per step) stops paying.
MATRIX_ROW_LIMIT = 256
# Above this, the persistent kernel's register-resident sub-chunking loses to
# the plain per-step sequential kernel.
PERSISTENT_ROW_LIMIT = 2048
# Widest block the persistent kernel still beats sequential on.
PERSISTENT_MAX_BLOCK = 8


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _sm_count(device) -> int:
    try:
        return torch.cuda.get_device_properties(device).multi_processor_count
    except Exception:
        return 132


def has_log_depth_kernel(block: int) -> bool:
    """True when this width has an ``associative_scan`` kernel, so needs no chunking."""
    return block == 1 or block in SLAB_BLOCKS


def pick_num_chunks(*, rows: int, seq_len: int, block: int) -> int:
    """Number of time chunks to split the scan into.

    The scalar path never chunks: ``tl.associative_scan`` already resolves a
    1024-step slab in log depth, so a single pass is both the fastest and the
    cheapest option at every row count measured (down to 8 rows).

    The matrix path has no such primitive, so time chunking is its only source
    of extra parallelism - but it pays for it with a second read of ``A`` and a
    D^3 transition product per step, so it is used only while rows are scarce.
    """
    if has_log_depth_kernel(block):
        return 1
    if rows > MATRIX_ROW_LIMIT:
        return 1
    return max(1, seq_len // MATRIX_CHUNK)


def select_scan_mode(*, rows: int, seq_len: int, block: int) -> str:
    """Pick the scan kernel for this shape: chunked, persistent or sequential."""
    if has_log_depth_kernel(block):
        return "chunked"
    if rows <= MATRIX_ROW_LIMIT and seq_len >= 2 * MATRIX_CHUNK:
        return "chunked"
    if block <= PERSISTENT_MAX_BLOCK and rows <= PERSISTENT_ROW_LIMIT:
        return "persistent"
    return "sequential"


@triton.jit
def _affine_combine(a_l, b_l, a_r, b_r):
    """Compose two affine maps: apply the left one, then the right one."""
    return a_l * a_r, a_r * b_l + b_r


# --------------------------------------------------------------------------- #
# scalar path (D == 1): coalesced slabs + tl.associative_scan
# --------------------------------------------------------------------------- #
@triton.jit
def _scalar_scan_kernel(
    a_ptr, b_ptr, y_ptr, carry_a_ptr, carry_b_ptr, h_in_ptr,
    sa0, sa1,
    sb0, sb1,
    sy0, sy1,
    sc0, sc1,
    T, CHUNK,
    SLAB: tl.constexpr,
    SEEDED: tl.constexpr,
    EMIT_CARRY: tl.constexpr,
):
    """One program per (row, chunk).

    ``EMIT_CARRY``: write the chunk's (prod a, local end state) instead of y.
    ``SEEDED``: start from the relayed incoming state rather than zero.
    """
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    t_start = chunk * CHUNK
    t_end = tl.minimum(t_start + CHUNK, T)

    s_off = tl.arange(0, SLAB)

    if SEEDED:
        h = tl.load(h_in_ptr + row.to(tl.int64) * sc0 + chunk.to(tl.int64) * sc1).to(tl.float32)
    else:
        h = 0.0

    # Running chunk transition, only needed when emitting the carry.
    A_run = 1.0

    for t0 in range(t_start, t_end, SLAB):
        t_idx = t0 + s_off
        t_m = t_idx < t_end
        # Identity pad (a=1, b=0) keeps the scan exact past the chunk end.
        a = tl.load(
            a_ptr + row.to(tl.int64) * sa0 + t_idx.to(tl.int64) * sa1,
            mask=t_m, other=1.0,
        ).to(tl.float32)
        b = tl.load(
            b_ptr + row.to(tl.int64) * sb0 + t_idx.to(tl.int64) * sb1,
            mask=t_m, other=0.0,
        ).to(tl.float32)

        P, Y = tl.associative_scan((a, b), 0, _affine_combine)
        # Fold in the state carried from previous slabs / previous chunks.
        Y = Y + P * h

        if not EMIT_CARRY:
            tl.store(
                y_ptr + row.to(tl.int64) * sy0 + t_idx.to(tl.int64) * sy1,
                Y.to(y_ptr.dtype.element_ty), mask=t_m,
            )

        # Last lane holds the slab's inclusive result.
        h = tl.sum(tl.where(s_off == SLAB - 1, Y, 0.0), axis=0)
        A_run = A_run * tl.sum(tl.where(s_off == SLAB - 1, P, 0.0), axis=0)

    if EMIT_CARRY:
        off = row.to(tl.int64) * sc0 + chunk.to(tl.int64) * sc1
        tl.store(carry_a_ptr + off, A_run)
        tl.store(carry_b_ptr + off, h)


# --------------------------------------------------------------------------- #
# matrix path (D > 1): per-step loads, time-chunked grid
# --------------------------------------------------------------------------- #
@triton.jit
def _matrix_scan_kernel(
    A_ptr, b_ptr, y_ptr, carry_A_ptr, carry_b_ptr, h_in_ptr,
    sa0, sa1, sa2, sa3,
    sb0, sb1, sb2,
    sy0, sy1, sy2,
    sca0, sca1, sca2, sca3,
    scb0, scb1, scb2,
    T, CHUNK, D,
    BLOCK_D: tl.constexpr,
    SEEDED: tl.constexpr,
    EMIT_CARRY: tl.constexpr,
):
    """One program per (row, chunk); state and chunk transition in registers."""
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    t_start = chunk * CHUNK
    t_end = tl.minimum(t_start + CHUNK, T)

    i_off = tl.arange(0, BLOCK_D)
    i_m = i_off < D
    m2 = i_m[:, None] & i_m[None, :]

    if SEEDED:
        h = tl.load(
            h_in_ptr + row.to(tl.int64) * scb0 + chunk.to(tl.int64) * scb1
            + i_off.to(tl.int64) * scb2,
            mask=i_m, other=0.0,
        ).to(tl.float32)
    else:
        h = tl.zeros((BLOCK_D,), dtype=tl.float32)

    eye = (i_off[:, None] == i_off[None, :]).to(tl.float32)
    A_run = eye

    for t in range(t_start, t_end):
        A_t = tl.load(
            A_ptr + row.to(tl.int64) * sa0 + t.to(tl.int64) * sa1
            + i_off[:, None].to(tl.int64) * sa2 + i_off[None, :].to(tl.int64) * sa3,
            mask=m2, other=0.0,
        ).to(tl.float32)
        b_t = tl.load(
            b_ptr + row.to(tl.int64) * sb0 + t.to(tl.int64) * sb1
            + i_off.to(tl.int64) * sb2,
            mask=i_m, other=0.0,
        ).to(tl.float32)

        h = tl.sum(A_t * h[None, :], axis=1) + b_t
        if not EMIT_CARRY:
            tl.store(
                y_ptr + row.to(tl.int64) * sy0 + t.to(tl.int64) * sy1
                + i_off.to(tl.int64) * sy2,
                h.to(y_ptr.dtype.element_ty), mask=i_m,
            )
        else:
            A_run = tl.sum(A_t[:, :, None] * A_run[None, :, :], axis=1)

    if EMIT_CARRY:
        tl.store(
            carry_A_ptr + row.to(tl.int64) * sca0 + chunk.to(tl.int64) * sca1
            + i_off[:, None].to(tl.int64) * sca2 + i_off[None, :].to(tl.int64) * sca3,
            A_run, mask=m2,
        )
        tl.store(
            carry_b_ptr + row.to(tl.int64) * scb0 + chunk.to(tl.int64) * scb1
            + i_off.to(tl.int64) * scb2,
            h, mask=i_m,
        )


# --------------------------------------------------------------------------- #
# carry relay: exclusive prefix over chunks, one program per row
# --------------------------------------------------------------------------- #
@triton.jit
def _relay_kernel(
    carry_A_ptr, carry_b_ptr, h_in_ptr,
    sca0, sca1, sca2, sca3,
    scb0, scb1, scb2,
    NC, D,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    i_off = tl.arange(0, BLOCK_D)
    i_m = i_off < D
    m2 = i_m[:, None] & i_m[None, :]

    h = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for c in range(NC):
        A_c = tl.load(
            carry_A_ptr + row.to(tl.int64) * sca0 + c.to(tl.int64) * sca1
            + i_off[:, None].to(tl.int64) * sca2 + i_off[None, :].to(tl.int64) * sca3,
            mask=m2, other=0.0,
        ).to(tl.float32)
        b_c = tl.load(
            carry_b_ptr + row.to(tl.int64) * scb0 + c.to(tl.int64) * scb1
            + i_off.to(tl.int64) * scb2,
            mask=i_m, other=0.0,
        ).to(tl.float32)
        tl.store(
            h_in_ptr + row.to(tl.int64) * scb0 + c.to(tl.int64) * scb1
            + i_off.to(tl.int64) * scb2,
            h, mask=i_m,
        )
        h = tl.sum(A_c * h[None, :], axis=1) + b_c


def _scalar_chunked_scan(A: torch.Tensor, b: torch.Tensor, num_chunks: int) -> torch.Tensor:
    BB, T = A.shape[0], A.shape[1]
    a2 = A.reshape(BB, T)
    b2 = b.reshape(BB, T)
    y = torch.empty_like(b2)

    chunk = _cdiv(T, num_chunks)
    num_chunks = _cdiv(T, chunk)
    slab = min(SCALAR_SLAB, _next_pow2(chunk))

    carry_a = carry_b = h_in = None
    if num_chunks > 1:
        carry_a = torch.empty(BB, num_chunks, device=A.device, dtype=torch.float32)
        carry_b = torch.empty(BB, num_chunks, device=A.device, dtype=torch.float32)
        h_in = torch.empty(BB, num_chunks, device=A.device, dtype=torch.float32)

        _scalar_scan_kernel[(BB, num_chunks)](
            a2, b2, y, carry_a, carry_b, h_in,
            a2.stride(0), a2.stride(1),
            b2.stride(0), b2.stride(1),
            y.stride(0), y.stride(1),
            carry_a.stride(0), carry_a.stride(1),
            T, chunk,
            SLAB=slab, SEEDED=False, EMIT_CARRY=True,
        )
        _relay_kernel[(BB,)](
            carry_a.reshape(BB, num_chunks, 1, 1), carry_b, h_in,
            carry_a.stride(0), carry_a.stride(1), 0, 0,
            carry_b.stride(0), carry_b.stride(1), 0,
            num_chunks, 1, BLOCK_D=1,
        )

    seeded = num_chunks > 1
    _scalar_scan_kernel[(BB, num_chunks)](
        a2, b2, y,
        carry_a if seeded else a2,
        carry_b if seeded else a2,
        h_in if seeded else a2,
        a2.stride(0), a2.stride(1),
        b2.stride(0), b2.stride(1),
        y.stride(0), y.stride(1),
        h_in.stride(0) if seeded else 0,
        h_in.stride(1) if seeded else 0,
        T, chunk,
        SLAB=slab, SEEDED=seeded, EMIT_CARRY=False,
    )
    return y.reshape(b.shape)


def _matrix_chunked_scan(A: torch.Tensor, b: torch.Tensor, num_chunks: int) -> torch.Tensor:
    BB, T, D, _ = A.shape
    block_d = _next_pow2(D)
    y = torch.empty(BB, T, D, device=b.device, dtype=b.dtype)

    chunk = _cdiv(T, num_chunks)
    num_chunks = _cdiv(T, chunk)

    carry_A = carry_b = h_in = None
    if num_chunks > 1:
        carry_A = torch.empty(BB, num_chunks, D, D, device=A.device, dtype=torch.float32)
        carry_b = torch.empty(BB, num_chunks, D, device=A.device, dtype=torch.float32)
        h_in = torch.empty(BB, num_chunks, D, device=A.device, dtype=torch.float32)

    def _launch(seeded: bool, emit_carry: bool):
        ca = carry_A if carry_A is not None else A
        cb = carry_b if carry_b is not None else b
        hi = h_in if h_in is not None else b
        _matrix_scan_kernel[(BB, num_chunks)](
            A, b, y, ca, cb, hi,
            A.stride(0), A.stride(1), A.stride(2), A.stride(3),
            b.stride(0), b.stride(1), b.stride(2),
            y.stride(0), y.stride(1), y.stride(2),
            ca.stride(0), ca.stride(1), ca.stride(2), ca.stride(3) if ca.dim() > 3 else 0,
            cb.stride(0), cb.stride(1), cb.stride(2),
            T, chunk, D,
            BLOCK_D=block_d, SEEDED=seeded, EMIT_CARRY=emit_carry,
        )

    if num_chunks > 1:
        _launch(seeded=False, emit_carry=True)
        _relay_kernel[(BB,)](
            carry_A, carry_b, h_in,
            carry_A.stride(0), carry_A.stride(1), carry_A.stride(2), carry_A.stride(3),
            carry_b.stride(0), carry_b.stride(1), carry_b.stride(2),
            num_chunks, D, BLOCK_D=block_d,
        )
    _launch(seeded=num_chunks > 1, emit_carry=False)
    return y


def chunked_affine_scan(
    A: torch.Tensor,
    b: torch.Tensor,
    *,
    num_chunks: int | None = None,
) -> torch.Tensor:
    """Occupancy-aware affine scan. A: (BB, T, D, D), b: (BB, T, D) -> (BB, T, D)."""
    BB, T, D, _ = A.shape
    if num_chunks is None:
        num_chunks = pick_num_chunks(rows=BB, seq_len=T, block=D)
    num_chunks = max(1, min(num_chunks, T))
    if D == 1:
        return _scalar_chunked_scan(A, b, num_chunks)
    if D in SLAB_BLOCKS and num_chunks == 1:
        return slab_affine_scan(A, b)
    return _matrix_chunked_scan(A, b, num_chunks)
