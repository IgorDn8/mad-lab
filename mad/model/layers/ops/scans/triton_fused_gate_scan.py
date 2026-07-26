"""Affine scan with the BD-LRU gating fused in, so ``A`` is never materialised.

Every other scan in this package takes ``A`` and ``b`` as tensors, which means
the layer has to build them first::

    A_t = softmax(gates, -1)                 # B T N m m+1   (materialised)
    a0v = A_t[..., -1] * v                   # B T N m
    A_t = A_t[..., :-1]                      # B T N m m
    A_bb = A_t.permute(0,2,1,3,4).reshape(BB,T,m,m)   # materialised copy
    b_bb = a0v.permute(0,2,1,3).reshape(BB,T,m)       # materialised copy

Even with inductor fusing that chain (``triton_auto_v2_compile``), the two
largest tensors still round-trip through HBM, and the backward writes a matching
``grad_A``. Profiling the full layer step showed this, not the scan, is the
bottleneck: the scan kernel was ~4% of CUDA time while ~70% went to elementwise
launches.

So this module takes ``gates`` and ``v`` directly and computes the gates inside
the kernel, in registers. ``A`` and ``grad_A`` never exist, and because the
output is produced as ``(B, T, N, m)`` the layer's final reshape is a free view
rather than a third copy.

Layout: ``gates`` is ``(B, T, N, m, m+1)``, so for a fixed timestep consecutive
``n`` are adjacent in memory while consecutive *timesteps* are ``N*m*(m+1)``
apart. The tiling therefore runs over ``n`` -- a program owns ``BN`` consecutive
blocks and walks time -- which is the opposite of the ``A``-based kernels, where
the permute had already made time contiguous. Time is split into chunks
(local scan -> carry relay -> reseeded rescan) for parallelism, since tiling
``n`` alone leaves only ``B * N/BN`` programs.

Currently ``m == 1`` only, where the softmax over the two gate entries collapses
to ``a = sigmoid(z0 - z1)`` and the state is one scalar per lane. Wider blocks
need a ``(BN, m, m)`` register tile for the transition (and a second one in the
backward for ``A_{t+1}``), so they keep the ``A``-based path; see
``fused_gate_supported``.

Backward. With ``p = softmax(z)``, ``A_t = p[..., :m]`` and ``b_t = p[..., m]*v``,
the adjoint of the softmax collapses neatly. Writing ``s_t = dL/dh_t`` from the
reverse scan ``s_t = grad_y_t + A_{t+1}^T s_{t+1}``:

    dL/dz[i,k] = s[i] * p[i,k] * ((k < m ? y_{t-1}[k] : v[i]) - y_t[i])
    dL/dv[i]   = s[i] * p[i,m]

because the ``sum(p * dL/dp)`` term of the softmax adjoint is exactly
``s[i] * y_t[i]``. So the backward needs only ``s``, the recomputed gates, and
``y`` -- no ``grad_A``, and no second reduction pass.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .triton_auto_scan import _cdiv, _relay_kernel, _sm_count

# Blocks per program. The per-step load is BN * (m+1) contiguous floats, so this
# sets both the coalescing width and the register footprint of the state. 128 is
# best or within a few percent everywhere measured, except at very large batch
# (B=128), where 256 is ~12% better.
BN_TILE = 128
# Shortest time chunk worth running; see _pick_num_chunks.
MIN_CHUNK = 128
# Programs per SM above which the n-tiled grid alone already fills the GPU.
# Measured crossover on an H100 is between 3.9 and 5.8 programs/SM, and it does
# not move with T (checked at T=4096 and T=16384).
SATURATION_PROGRAMS_PER_SM = 5


def fused_gate_supported(block: int) -> bool:
    """Whether this module has a kernel for this block width."""
    return block == 1


def _pick_num_chunks(*, batch: int, hidden: int, seq_len: int) -> int:
    """Chunk count for the time split.

    The choice is close to binary. Chunking costs a second pass over the data
    (local scan, relay, reseeded rescan) whatever the chunk count, while the
    benefit -- cutting the sequential depth each program walks -- keeps growing
    as chunks shrink. So either don't chunk at all, or go straight to the
    shortest useful chunk. On an H100 at B=4, T=16384, N=1946 the total
    fwd+bwd is 20.3 ms unchunked, 4.6 ms at chunk=2048 and 2.2 ms at chunk=128;
    once the n-tiles saturate the GPU on their own, chunking instead costs up to
    22% (B=64) for no gain, at any T.
    """
    programs = batch * _cdiv(hidden, BN_TILE)
    saturated = SATURATION_PROGRAMS_PER_SM * _sm_count(torch.cuda.current_device())
    if programs >= saturated:
        return 1
    return max(1, seq_len // MIN_CHUNK)


@triton.jit
def _fwd_kernel(
    z_ptr, v_ptr, y_ptr, carry_a_ptr, carry_b_ptr, h_in_ptr,
    sz0, sz1, sz2, sz4,
    sv0, sv1, sv2,
    sy0, sy1, sy2,
    sc0, sc1,
    T, N, CHUNK,
    BN: tl.constexpr,
    SEEDED: tl.constexpr,
    EMIT_CARRY: tl.constexpr,
):
    """One program per (batch * n-tile, chunk); state is one scalar per lane."""
    pid = tl.program_id(0)
    chunk = tl.program_id(1)
    n_tiles = tl.cdiv(N, BN)
    b = (pid // n_tiles).to(tl.int64)
    n_off = (pid % n_tiles) * BN + tl.arange(0, BN)
    n_m = n_off < N
    n64 = n_off.to(tl.int64)

    t_start = chunk * CHUNK
    t_end = tl.minimum(t_start + CHUNK, T)
    # carry rows follow the (batch*hidden) convention the relay expects
    crow = (b * N + n64) * sc0 + chunk.to(tl.int64) * sc1

    if SEEDED:
        h = tl.load(h_in_ptr + crow, mask=n_m, other=0.0).to(tl.float32)
    else:
        h = tl.zeros((BN,), dtype=tl.float32)
    a_run = tl.full((BN,), 1.0, tl.float32)

    zb = z_ptr + b * sz0 + n64 * sz2
    vb = v_ptr + b * sv0 + n64 * sv2
    yb = y_ptr + b * sy0 + n64 * sy2

    for t in range(t_start, t_end):
        t64 = t.to(tl.int64)
        z0 = tl.load(zb + t64 * sz1, mask=n_m, other=0.0).to(tl.float32)
        z1 = tl.load(zb + t64 * sz1 + sz4, mask=n_m, other=0.0).to(tl.float32)
        vt = tl.load(vb + t64 * sv1, mask=n_m, other=0.0).to(tl.float32)
        # softmax over the two gate entries is a sigmoid of their difference
        a = tl.sigmoid(z0 - z1)
        h = a * h + (1.0 - a) * vt
        if EMIT_CARRY:
            a_run = a_run * a
        else:
            tl.store(yb + t64 * sy1, h.to(y_ptr.dtype.element_ty), mask=n_m)

    if EMIT_CARRY:
        tl.store(carry_a_ptr + crow, a_run, mask=n_m)
        tl.store(carry_b_ptr + crow, h, mask=n_m)


@triton.jit
def _bwd_kernel(
    z_ptr, v_ptr, y_ptr, g_ptr, gz_ptr, gv_ptr,
    carry_a_ptr, carry_b_ptr, h_in_ptr,
    sz0, sz1, sz2, sz4,
    sv0, sv1, sv2,
    sy0, sy1, sy2,
    sg0, sg1, sg2,
    sgz0, sgz1, sgz2, sgz4,
    sgv0, sgv1, sgv2,
    sc0, sc1,
    T, N, CHUNK, NC,
    BN: tl.constexpr,
    SEEDED: tl.constexpr,
    EMIT_CARRY: tl.constexpr,
):
    """Reverse adjoint with the gates recomputed, emitting grad_gates/grad_v.

    Carries are stored at ``NC-1-chunk`` so that the forward (ascending) relay
    composes the chunks in the order the reverse scan needs, right to left.
    """
    pid = tl.program_id(0)
    chunk = tl.program_id(1)
    n_tiles = tl.cdiv(N, BN)
    b = (pid // n_tiles).to(tl.int64)
    n_off = (pid % n_tiles) * BN + tl.arange(0, BN)
    n_m = n_off < N
    n64 = n_off.to(tl.int64)

    t_start = chunk * CHUNK
    t_end = tl.minimum(t_start + CHUNK, T)
    crow = (b * N + n64) * sc0 + (NC - 1 - chunk).to(tl.int64) * sc1

    if SEEDED:
        s = tl.load(h_in_ptr + crow, mask=n_m, other=0.0).to(tl.float32)
    else:
        s = tl.zeros((BN,), dtype=tl.float32)
    a_run = tl.full((BN,), 1.0, tl.float32)

    zb = z_ptr + b * sz0 + n64 * sz2
    vb = v_ptr + b * sv0 + n64 * sv2
    yb = y_ptr + b * sy0 + n64 * sy2
    gb = g_ptr + b * sg0 + n64 * sg2
    gzb = gz_ptr + b * sgz0 + n64 * sgz2
    gvb = gv_ptr + b * sgv0 + n64 * sgv2

    # transition into the last step of this chunk is a_{t_end}; there is none at
    # the very end of the sequence
    e64 = t_end.to(tl.int64)
    z0n = tl.load(zb + e64 * sz1, mask=n_m & (t_end < T), other=0.0).to(tl.float32)
    z1n = tl.load(zb + e64 * sz1 + sz4, mask=n_m & (t_end < T), other=0.0).to(tl.float32)
    a_next = tl.where(t_end < T, tl.sigmoid(z0n - z1n), 0.0)

    for t in range(t_end - 1, t_start - 1, -1):
        t64 = t.to(tl.int64)
        z0 = tl.load(zb + t64 * sz1, mask=n_m, other=0.0).to(tl.float32)
        z1 = tl.load(zb + t64 * sz1 + sz4, mask=n_m, other=0.0).to(tl.float32)
        a = tl.sigmoid(z0 - z1)

        gy = tl.load(gb + t64 * sg1, mask=n_m, other=0.0).to(tl.float32)
        s = gy + a_next * s

        if EMIT_CARRY:
            a_run = a_run * a_next
        else:
            vt = tl.load(vb + t64 * sv1, mask=n_m, other=0.0).to(tl.float32)
            yt = tl.load(yb + t64 * sy1, mask=n_m, other=0.0).to(tl.float32)
            y_prev = tl.load(yb + (t64 - 1) * sy1,
                             mask=n_m & (t > 0), other=0.0).to(tl.float32)
            # softmax adjoint, using sum(p * dL/dp) == s * y_t
            tl.store(gzb + t64 * sgz1,
                     (s * a * (y_prev - yt)).to(gz_ptr.dtype.element_ty), mask=n_m)
            tl.store(gzb + t64 * sgz1 + sgz4,
                     (s * (1.0 - a) * (vt - yt)).to(gz_ptr.dtype.element_ty), mask=n_m)
            tl.store(gvb + t64 * sgv1,
                     (s * (1.0 - a)).to(gv_ptr.dtype.element_ty), mask=n_m)

        a_next = a

    if EMIT_CARRY:
        tl.store(carry_a_ptr + crow, a_run, mask=n_m)
        tl.store(carry_b_ptr + crow, s, mask=n_m)


def _grid(B: int, N: int, num_chunks: int):
    return (B * _cdiv(N, BN_TILE), num_chunks)


def _relay(carry_a, carry_b, h_in, rows, num_chunks):
    _relay_kernel[(rows,)](
        carry_a.reshape(rows, num_chunks, 1, 1), carry_b, h_in,
        carry_a.stride(0), carry_a.stride(1), 0, 0,
        carry_b.stride(0), carry_b.stride(1), 0,
        num_chunks, 1, BLOCK_D=1,
    )


def _fused_gate_forward(gates, v, num_chunks):
    B, T, N = gates.shape[0], gates.shape[1], gates.shape[2]
    y = torch.empty(B, T, N, 1, device=v.device, dtype=v.dtype)
    chunk = _cdiv(T, num_chunks)
    num_chunks = _cdiv(T, chunk)
    rows = B * N

    carry_a = carry_b = h_in = None
    if num_chunks > 1:
        carry_a = torch.empty(rows, num_chunks, device=v.device, dtype=torch.float32)
        carry_b = torch.empty(rows, num_chunks, device=v.device, dtype=torch.float32)
        h_in = torch.empty(rows, num_chunks, device=v.device, dtype=torch.float32)

    def launch(seeded, emit_carry):
        ca = carry_a if carry_a is not None else y
        cb = carry_b if carry_b is not None else y
        hi = h_in if h_in is not None else y
        _fwd_kernel[_grid(B, N, num_chunks)](
            gates, v, y, ca, cb, hi,
            gates.stride(0), gates.stride(1), gates.stride(2), gates.stride(4),
            v.stride(0), v.stride(1), v.stride(2),
            y.stride(0), y.stride(1), y.stride(2),
            h_in.stride(0) if h_in is not None else 0,
            h_in.stride(1) if h_in is not None else 0,
            T, N, chunk,
            BN=BN_TILE, SEEDED=seeded, EMIT_CARRY=emit_carry,
        )

    if num_chunks > 1:
        launch(seeded=False, emit_carry=True)
        _relay(carry_a, carry_b, h_in, rows, num_chunks)
    launch(seeded=num_chunks > 1, emit_carry=False)
    return y


def _fused_gate_backward(gates, v, y, grad_y, num_chunks):
    B, T, N = gates.shape[0], gates.shape[1], gates.shape[2]
    grad_z = torch.empty_like(gates)
    grad_v = torch.empty_like(v)
    chunk = _cdiv(T, num_chunks)
    num_chunks = _cdiv(T, chunk)
    rows = B * N

    carry_a = carry_b = h_in = None
    if num_chunks > 1:
        carry_a = torch.empty(rows, num_chunks, device=v.device, dtype=torch.float32)
        carry_b = torch.empty(rows, num_chunks, device=v.device, dtype=torch.float32)
        h_in = torch.empty(rows, num_chunks, device=v.device, dtype=torch.float32)

    def launch(seeded, emit_carry):
        ca = carry_a if carry_a is not None else grad_v
        cb = carry_b if carry_b is not None else grad_v
        hi = h_in if h_in is not None else grad_v
        _bwd_kernel[_grid(B, N, num_chunks)](
            gates, v, y, grad_y, grad_z, grad_v,
            ca, cb, hi,
            gates.stride(0), gates.stride(1), gates.stride(2), gates.stride(4),
            v.stride(0), v.stride(1), v.stride(2),
            y.stride(0), y.stride(1), y.stride(2),
            grad_y.stride(0), grad_y.stride(1), grad_y.stride(2),
            grad_z.stride(0), grad_z.stride(1), grad_z.stride(2), grad_z.stride(4),
            grad_v.stride(0), grad_v.stride(1), grad_v.stride(2),
            h_in.stride(0) if h_in is not None else 0,
            h_in.stride(1) if h_in is not None else 0,
            T, N, chunk, num_chunks,
            BN=BN_TILE, SEEDED=seeded, EMIT_CARRY=emit_carry,
        )

    if num_chunks > 1:
        launch(seeded=False, emit_carry=True)
        _relay(carry_a, carry_b, h_in, rows, num_chunks)
    launch(seeded=num_chunks > 1, emit_carry=False)
    return grad_z, grad_v


class _FusedGateScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gates, v, num_chunks):
        gates = gates.contiguous()
        v = v.contiguous()
        with torch.no_grad():
            y = _fused_gate_forward(gates, v, num_chunks)
        ctx.save_for_backward(gates, v, y)
        ctx.num_chunks = num_chunks
        return y

    @staticmethod
    def backward(ctx, grad_y):
        gates, v, y = ctx.saved_tensors
        with torch.no_grad():
            grad_z, grad_v = _fused_gate_backward(
                gates, v, y, grad_y.contiguous(), ctx.num_chunks)
        return grad_z, grad_v, None


def fused_gate_scan(gates: torch.Tensor, v: torch.Tensor,
                    *, num_chunks: int | None = None) -> torch.Tensor:
    """Softmax-gated affine scan straight from ``gates``/``v``, no ``A`` tensor.

    gates: (B, T, N, m, m+1), v: (B, T, N, m) -> y: (B, T, N, m), solving
    ``h_t = A_t h_{t-1} + b_t`` with ``A_t = softmax(gates)[..., :m]`` and
    ``b_t = softmax(gates)[..., m] * v``. Requires ``m == 1``.
    """
    m = gates.shape[3]
    if not fused_gate_supported(m):
        raise ValueError(f"fused_gate_scan currently supports m == 1, got {m}")
    if num_chunks is None:
        num_chunks = _pick_num_chunks(
            batch=gates.shape[0], hidden=gates.shape[2], seq_len=gates.shape[1])
    return _FusedGateScan.apply(gates, v, num_chunks)
