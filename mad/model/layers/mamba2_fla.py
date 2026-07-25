"""Mamba2 sequence-mixer backed by flash-linear-attention (`fla`).

Unlike ``mamba_v2.py`` (which wraps the ``mamba_ssm`` package and its CUDA
extensions), this layer uses ``fla.layers.Mamba2``, whose scan is implemented in
Triton -- so it needs no ``mamba_ssm`` / ``causal_conv1d`` build and runs on the
Triton already required by the other fla baselines. Maps ``(B, T, dim)`` ->
``(B, T, dim)``. Runs in bf16 (the fla native precision).
"""

import torch
import torch.nn as nn

try:
    from fla.layers import Mamba2 as FLAMamba2
except ImportError:
    print("fla is not installed! ")
    FLAMamba2 = None


class Mamba2fla(nn.Module):
    """Mamba2 layer using flash-linear-attention's Triton kernels."""

    def __init__(
        self,
        dim: int,
        head_dim: int = 64,
        state_size: int = 128,
        expand: int = 2,
        n_groups: int = 1,
        conv_kernel: int = 4,
        chunk_size: int = 256,
        backend: str = "triton",
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.layer = FLAMamba2(
            hidden_size=dim,
            head_dim=head_dim,
            state_size=state_size,
            expand=expand,
            n_groups=n_groups,
            conv_kernel=conv_kernel,
            chunk_size=chunk_size,
            backend=backend,
        )

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        out = self.layer(hidden_states)
        return out[0] if isinstance(out, (tuple, list)) else out
