"""Permutation-Dictionary State-Space Model (PDSSM) layer.

Adapted into a single mad-lab sequence-mixer layer from the ``PD_Block`` of the
`expressive-sparse-state-space-model` project. The block runs a complex-valued
recurrence

    h_t = M_t @ h_{t-1} + b_t ,     M_t = D_t * P_t

where ``D_t`` is an input-dependent complex diagonal (magnitude in (0,1) via
sigmoid, phase in (0,2*pi)) and ``P_t`` is a (straight-through) one-hot column
permutation selected from a learned matrix dictionary. The complex state is read
out (real+imag) and combined with a per-channel skip connection. The layer maps
``(B, T, dim) -> (B, T, dim)`` so it drops into the standard MAD backbone.

Two numerically-equivalent implementations of the time recurrence are provided,
selected by the ``implementation`` config field:

  * ``sequential``       -- the original explicit for-loop over time (reference).
  * ``associative_scan`` -- PyTorch's parallel prefix scan
    (``torch._higher_order_ops.associative_scan``) over the affine maps
    ``(M_t, b_t)``; the one-hot initial state is folded into ``b_0``.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum
from torch._higher_order_ops.associative_scan import associative_scan


def _affine_combine(acc: dict, cur: dict) -> dict:
    """Associative composition of two affine maps h -> A h + b (cur applied after acc)."""
    A = torch.einsum('bij,bjk->bik', cur['A'], acc['A'])
    b = cur['b'] + torch.einsum('bij,bj->bi', cur['A'], acc['b'])
    return {'A': A, 'b': b}


class PDSSM(nn.Module):
    """Permutation-Dictionary SSM block with sequential / associative-scan recurrence."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int = 128,
        dictionary_size: int = 8,
        hidden_D_multiple: int = 2,
        dropout_rate: float = 0.01,
        implementation: str = "sequential",
        **kwargs,
    ):
        super().__init__()

        self.dim = dim
        self.embed_size = dim
        self.hidden_size = hidden_dim
        self.dict_size = dictionary_size
        self.implementation = implementation

        # post-LN residual block structure (as in the original PD_Block)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(p=dropout_rate)

        # magnitudes in (0,1) via sigmoid; phases in (0,1) then scaled to (0,2*pi)
        self.D_magnitude_generator = nn.Sequential(
            nn.Linear(dim, hidden_D_multiple * hidden_dim), nn.GELU(),
            nn.Linear(hidden_D_multiple * hidden_dim, hidden_dim), nn.Sigmoid(),
        )
        self.D_phase_generator = nn.Sequential(
            nn.Linear(dim, hidden_D_multiple * hidden_dim), nn.GELU(),
            nn.Linear(hidden_D_multiple * hidden_dim, hidden_dim), nn.Sigmoid(),
        )

        # selector over the learned matrix dictionary
        self.S = nn.Linear(dim, dictionary_size, bias=False)
        self.A_dict = nn.Parameter(
            torch.randn(hidden_dim, hidden_dim, dictionary_size) / np.sqrt(hidden_dim)
        )

        # complex input projection (Glorot with halved variance)
        self.B_re = nn.Parameter(torch.randn(hidden_dim, dim) / np.sqrt(2 * hidden_dim))
        self.B_im = nn.Parameter(torch.randn(hidden_dim, dim) / np.sqrt(2 * hidden_dim))

        # per-channel skip and complex-state readout (real+imag -> dim)
        self.D = nn.Parameter(torch.randn(dim) / np.sqrt(dim))
        self.readout = nn.Linear(2 * hidden_dim, dim)

    def _scan_sequential(self, M: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Reference recurrence: explicit loop from the one-hot initial state."""
        B, L, N, _ = M.shape
        h = torch.zeros(B, N, dtype=b.dtype, device=b.device)
        h[:, 0] = 1.0  # one-hot initial state e_0 = [1, 0, ..., 0]
        outs = []
        for i in range(L):
            h = torch.einsum('bmn,bn->bm', M[:, i], h) + b[:, i]
            outs.append(h)
        return torch.stack(outs, dim=1)  # B L N

    def _scan_associative(self, M: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Parallel prefix scan of h_t = M_t h_{t-1} + b_t (h_{-1}=0).

        The non-zero one-hot initial state e_0 is folded into the first input:
        h_0 = M_0 e_0 + b_0, and since e_0 = [1,0,...], M_0 e_0 is M_0's first column.
        """
        b = b.clone()
        b[:, 0] = b[:, 0] + M[:, 0, :, 0]
        out = associative_scan(_affine_combine, {'A': M, 'b': b}, dim=1, combine_mode='generic')
        return out['b']  # B L N

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """x: (B, T, dim) -> y: (B, T, dim)."""
        B, L, E = x.shape
        N = self.hidden_size

        # dictionary selection -> per-step dense transition structure  (B L N N)
        selection_weights = F.softmax(self.S(x), dim=-1)                       # B L K
        M = einsum(self.A_dict, selection_weights, 'n1 n2 k, b l k -> b l n1 n2')

        # straight-through one-hot column permutation, gradient follows column softmax
        y_soft = F.softmax(M, dim=-1)
        y_hard = F.one_hot(torch.argmax(y_soft, dim=-1), num_classes=y_soft.shape[-1]).to(y_soft.dtype)
        P = (y_hard - y_soft).detach() + y_soft                                # B L N N

        # complex diagonal D_t = magnitude * exp(i * phase)
        magnitudes = self.D_magnitude_generator(x)                             # B L N (real)
        phases = 2 * math.pi * self.D_phase_generator(x)                       # B L N (real)
        D_diag = torch.complex(magnitudes, torch.zeros_like(magnitudes)) * torch.exp(
            torch.complex(torch.zeros_like(phases), phases)
        )                                                                      # B L N (complex)

        transition_matrices = D_diag.unsqueeze(-1) * P                         # B L N N (complex)

        # complex input transform b_t = B_mat @ x_t
        B_mat = torch.complex(self.B_re, self.B_im)                            # N E
        x_c = torch.complex(x, torch.zeros_like(x))                            # B L E
        b = torch.einsum('ne,ble->bln', B_mat, x_c)                            # B L N (complex)

        if self.implementation == "sequential":
            hidden_states = self._scan_sequential(transition_matrices, b)
        elif self.implementation == "associative_scan":
            hidden_states = self._scan_associative(transition_matrices, b)
        else:
            raise ValueError(
                f"PDSSM implementation {self.implementation!r} not supported "
                f"(choose 'sequential' or 'associative_scan')"
            )

        # readout of the complex state (real|imag) + per-channel skip, then post-LN
        linear_readout_re_im = torch.cat((hidden_states.real, hidden_states.imag), dim=-1)  # B L 2N
        output = self.readout(self.drop(linear_readout_re_im)) + self.D * x
        return self.norm(output)


if __name__ == '__main__':
    x = torch.randn(2, 64, 128, requires_grad=True)
    if torch.cuda.is_available():
        x = x.cuda()
    for impl in ('sequential', 'associative_scan'):
        m = PDSSM(dim=128, hidden_dim=128, implementation=impl)
        if torch.cuda.is_available():
            m = m.cuda()
        y = m(x)
        y.sum().backward()
        print(impl, tuple(y.shape), 'grad ok' if x.grad is not None else 'no grad')
        x.grad = None
