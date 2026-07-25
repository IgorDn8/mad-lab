"""Affine parallel-scan primitive (adapted from nisys-bench torch_affine_scan).

Solves the affine (linear) recurrence

    h_t = W_t @ h_{t-1} + x_t

in a channelized layout so the same code covers dense (C=1) and block-diagonal
(C=num_blocks) models:

    x: (B, T, C, D)          per-step inputs
    W: (B, T, C, D, D)       per-step transition matrices
    y: (B, T, C, D)          outputs (hidden states)

Unlike ``hopscan_custom`` (which wraps a hand-written autograd Function),
this mirrors ``nisys-bench/models/torch_affine_scan.py``: it scans the *time*
dimension (dim=1) with ``torch.associative_scan`` in ``generic`` combine mode and
lets autograd differentiate straight through the higher-order op. Scanning a
non-trailing axis is what makes this work where a trailing-axis parallel scan
path would trip the vmap batching in current PyTorch.
"""

from __future__ import annotations

import torch
from torch._higher_order_ops.associative_scan import associative_scan


def scan_sequential(W: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Sequential reference: loop over time applying W_t @ h + x_t."""
    B, T, C, D = x.shape
    h = torch.zeros(B, C, D, device=x.device, dtype=x.dtype)
    outputs = torch.empty_like(x)
    for t in range(T):
        h = torch.einsum("bcij,bcj->bci", W[:, t], h) + x[:, t]
        outputs[:, t] = h
    return outputs


def _affine_combine(acc: dict, curr: dict) -> dict:
    """Compose two affine maps along the scan (curr is applied after acc)."""
    c = torch.einsum("bcij,bcjk->bcik", curr["c"], acc["c"])
    x = curr["x"] + torch.einsum("bcij,bcj->bci", curr["c"], acc["x"])
    return dict(x=x, c=c)


def scan_parallel(W: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Parallel associative scan over the time dimension (dim=1)."""
    # associative_scan (generic) needs materialized, contiguous operands.
    x = x.contiguous()
    W = W.contiguous()
    return associative_scan(
        _affine_combine,
        dict(x=x, c=W),
        dim=1,
        combine_mode="generic",
    )["x"]
