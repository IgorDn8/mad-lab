"""Selective higher-order LRU layer.

This variant uses input-dependent gates to select higher-order recurrent state
mixing at each sequence position. It corresponds to the formerly named
``HLRU_o1`` implementation.
"""

import math
import torch
import torch.nn.functional as F
import torch.nn as nn

try:
    from mad.model.layers.ops.scans.hopscan import hopscan_opt
except ImportError:
    hopscan_opt = None

try:
    from mad.model.layers.ops.scans.triton_scans import triton_affine_scan
except ImportError:
    triton_affine_scan = None

# implementation string -> triton scan kernel mode
_TRITON_SCAN_MODES = {
    "triton_sequential": "sequential",
    "triton_persistent": "persistent",
    "triton_parallel_blelloch": "blelloch",
    "triton_chunked": "chunked",
    # picks chunked/persistent/sequential from (batch*hidden_dim, T, window_dim)
    "triton_auto": "auto",
    # same forward as triton_auto, but the backward runs the fused reverse
    # kernels instead of materialising flipped/transposed copies of A
    "triton_auto_v2": "auto_v2",
}

# As above, but the softmax/gating/layout chain around the scan is handed to
# inductor instead of running as eager elementwise kernels. Separate from
# _TRITON_SCAN_MODES so the un-compiled variants stay selectable for A/B.
_TRITON_COMPILED_MODES = {
    "triton_auto_compile": "auto",
    "triton_auto_v2_compile": "auto_v2",
    # H-LRU's transition is a companion matrix, not a per-row softmax, so the
    # fused-gate kernel (which computes BD-LRU's gating inside the scan) does not
    # apply here; the name stays selectable and falls back to the compiled path.
    "triton_fused_gates": "auto_v2",
}


def _companion_A_and_b(A_temp: torch.Tensor, gates: torch.Tensor, v: torch.Tensor):
    """Build the H-LRU companion transition in the *orig* layout.

    ``orig`` stores coeffs in the first column with 1's on the superdiagonal and
    applies the right-multiply recurrence ``h <- h @ A + b``. Parallel scans
    (hopscan / triton) implement the left-multiply form ``h <- A @ h + b``, so
    callers must pass ``A.transpose(-1, -2)`` into those kernels.
    """
    m = A_temp.shape[0]
    a0v = F.pad(gates[..., -1:] * v.unsqueeze(-1), (0, m - 1))  # B T N m
    A = A_temp + F.pad(gates[..., :-1].unsqueeze(-1), (0, m - 1))  # B T N m m
    return A, a0v


def _hopscan_custom_recurrence(
    gates: torch.Tensor, v: torch.Tensor, A_temp: torch.Tensor
) -> torch.Tensor:
    """Softmax-gated H-LRU recurrence via hopscan (matches ``orig``).

    Returns the pre-projection state (B, T, N, m). Wrapped by ``torch.compile``
    below for the ``custom_hopscan_autotune`` variant.
    """
    A_t = torch.softmax(gates, -1)  # B T N m+1
    A_t, a0v = _companion_A_and_b(A_temp, A_t, v)
    # left-multiply scan needs the transpose of orig's companion layout
    return hopscan_opt(a0v, A_t.transpose(-1, -2))  # B T N m


_hopscan_custom_recurrence_autotune = (
    torch.compile(_hopscan_custom_recurrence, mode="max-autotune", dynamic=False)
    if hopscan_opt is not None
    else None
)


def _triton_gate_inputs(
    gates: torch.Tensor, v: torch.Tensor, A_temp: torch.Tensor, window_dim: int
):
    """Softmax gating -> the (batch*hidden, T, m, m) layout the triton scans take.

    The scan kernel is only a few percent of the step; most of it goes to eager
    elementwise launches, above all the ``permute(...).reshape(...)`` on the two
    largest tensors, which each materialise a copy (plus a mirror in the
    backward). Isolating the chain here lets inductor fuse it.
    """
    B, T, N = gates.shape[0], gates.shape[1], gates.shape[2]
    A_t = torch.softmax(gates, -1)
    A_t, a0v = _companion_A_and_b(A_temp, A_t, v)
    # triton affine scan is also left-multiply; transpose to match orig
    A_t = A_t.transpose(-1, -2)
    BB = B * N
    A_bb = A_t.permute(0, 2, 1, 3, 4).reshape(BB, T, window_dim, window_dim)
    b_bb = a0v.permute(0, 2, 1, 3).reshape(BB, T, window_dim)
    return A_bb, b_bb


def _triton_scan_output(y: torch.Tensor, B: int, T: int, N: int, window_dim: int):
    """(BB, T, m) scan output -> (B, T, N*m)."""
    y = y.reshape(B, N, T, window_dim).permute(0, 2, 1, 3)
    return y.reshape(B, T, N * window_dim)


_triton_gate_inputs_autotune = torch.compile(
    _triton_gate_inputs, mode="max-autotune", dynamic=False)
_triton_scan_output_autotune = torch.compile(
    _triton_scan_output, mode="max-autotune", dynamic=False)


# @torch.compile(mode="max-autotune", dynamic=False)
class HLRU_sel(nn.Module):
    """H-LRU selective variant with token-dependent recurrent transitions."""

    def __init__(
        self,
        dim: int,
        eps: float = 1e-12,
        implementation: str="parallel",
        window_dim: int = 16,
        max_length: int = 256,
        hidden_dim: int = 64,
        **kwargs
    ):
        super().__init__()
        
        self.dim = dim
        self.hidden_dim = hidden_dim
        # self.key_dim = int(self.dim * expand_k)
        # self.value_dim = int(self.dim * expand_v)
        # self.head_qk_dim = self.key_dim // self.num_heads
        # self.head_v_dim = self.value_dim // self.num_heads
        self.window_dim = window_dim
        self.eps = eps
        self.implementation = implementation
        self.max_length= max_length

        # initialize projections and feature map
        self.proj_gates = nn.Linear(self.dim, self.hidden_dim*(self.window_dim+1), bias=True)
        self.proj_v = nn.Linear(self.dim, self.hidden_dim, bias=False)
        self.proj_out = torch.nn.Linear(self.hidden_dim*self.window_dim, self.dim, bias=False)

        # self.proj_nl = nn.Linear(self.dim, self.dim, bias=False)
        self.hidden_dynamics = []
        #self.register_parameter('A_w', torch.nn.Parameter(torch.rand(self.dim,self.window_dim)))

        self.register_buffer("A_temp", torch.diag(torch.ones(self.window_dim-1), 1))
        self.register_buffer("R_mask", F.pad(torch.ones((self.window_dim,1)),(self.window_dim-1,0)).transpose(0,1))

    
    def forward(self, 
        hidden_states: torch.Tensor,
        *args, **kwargs
    ):
        """
        x (torch.Tensor): tensor of shape (b, t, c)
        y (torch.Tensor): tensor of shape (b, t, c)
        """
        # q is gate, k is non-linear proj, v is linear proj
        B, T, _ = hidden_states.size()
        v = self.proj_v(hidden_states) # B T C

        gates = self.proj_gates(hidden_states) # B T C
        gates = gates.reshape(B,T,self.hidden_dim,self.window_dim+1)

        if self.implementation == "orig":
            #b t c
            hidden_x = torch.zeros(B, self.hidden_dim, self.window_dim).to(hidden_states.device)    # B C WD        
            y = []
            A_qk = torch.softmax(gates,-1) # B T C WD+1
            nl = F.pad(A_qk[:,:,:,-1:]*v[:,:,:].unsqueeze(-1),(0,self.window_dim-1)) # B T C WD
            A_qk = self.A_temp + F.pad(A_qk[:,:,:,:-1].unsqueeze(-1),(0,self.window_dim-1)) # B T C WD WD

            for i in range(T):
                
                hidden_x = torch.einsum('bci, bcij -> bcj',hidden_x,A_qk[:,i,:,:,:])+ nl[:,i,:,:]
                y.append(hidden_x) 

            y=torch.stack(y, dim=1)
            y=y.reshape(B,T,self.hidden_dim*self.window_dim)
            
        elif self.implementation == "hopscan_custom":

            A_t = torch.softmax(gates, -1)  # B T N m+1
            A_t, a0v = _companion_A_and_b(self.A_temp, A_t, v)
            # hopscan does h <- A @ h + b; transpose matches orig's h <- h @ A + b
            y = hopscan_opt(a0v, A_t.transpose(-1, -2))  # B T N m
            y = y.reshape(B, T, self.hidden_dim * self.window_dim)

        elif self.implementation == "custom_hopscan_autotune":

            # identical recurrence to `hopscan_custom`, with gating + hopscan
            # wrapped in torch.compile(mode="max-autotune", dynamic=False).
            if _hopscan_custom_recurrence_autotune is None:
                raise ImportError(
                    "hopscan_opt unavailable; cannot build custom_hopscan_autotune"
                )
            y = _hopscan_custom_recurrence_autotune(gates, v, self.A_temp)  # B T N m
            y = y.reshape(B, T, self.hidden_dim * self.window_dim)

        elif self.implementation in _TRITON_SCAN_MODES:

            A_t = torch.softmax(gates, -1)  # B T N m+1
            A_t, a0v = _companion_A_and_b(self.A_temp, A_t, v)
            # triton affine scan is also left-multiply; transpose to match orig
            A_t = A_t.transpose(-1, -2)

            BB = B * self.hidden_dim
            A_bb = A_t.permute(0, 2, 1, 3, 4).reshape(BB, T, self.window_dim, self.window_dim)
            b_bb = a0v.permute(0, 2, 1, 3).reshape(BB, T, self.window_dim)

            y = triton_affine_scan(A_bb, b_bb, _TRITON_SCAN_MODES[self.implementation])  # BB T m

            y = y.reshape(B, self.hidden_dim, T, self.window_dim).permute(0, 2, 1, 3)
            y = y.reshape(B, T, self.hidden_dim * self.window_dim)

        elif self.implementation in _TRITON_COMPILED_MODES:

            # identical math to the branch above; the gating and layout chains
            # are compiled, and the scan stays an opaque op between them.
            A_bb, b_bb = _triton_gate_inputs_autotune(
                gates, v, self.A_temp, self.window_dim)

            y = triton_affine_scan(A_bb, b_bb,
                                   _TRITON_COMPILED_MODES[self.implementation])  # BB T m

            y = _triton_scan_output_autotune(
                y, B, T, self.hidden_dim, self.window_dim)

        elif self.implementation == "orig_save_dynamics":
            #b t c
            hidden_x = torch.zeros(B, self.hidden_dim, self.window_dim).to(hidden_states.device)    # B C WD        
            y = []
            A_qk = torch.softmax(gates,-1) # B T C WD+1
            nl = F.pad(A_qk[:,:,:,-1:]*v[:,:,:].unsqueeze(-1),(0,self.window_dim-1)) # B T C WD
            A_qk = self.A_temp + F.pad(A_qk[:,:,:,:-1].unsqueeze(-1),(0,self.window_dim-1)) # B T C WD WD

            self.hidden_dynamics = []
            self.hidden_dynamics.append(A_qk)

            for i in range(T):
                
                hidden_x = torch.einsum('bci, bcij -> bcj',hidden_x,A_qk[:,i,:,:,:])+ nl[:,i,:,:]
                y.append(hidden_x) 

                if hidden_x.requires_grad:
                    hidden_x.retain_grad()
                self.hidden_dynamics.append(hidden_x)

            y=torch.stack(y, dim=1)
            y=y.reshape(B,T,self.hidden_dim*self.window_dim)

        elif self.implementation == "sigmoid_l1":
            #b t c
            hidden_x = torch.zeros(B, self.hidden_dim, self.window_dim).to(hidden_states.device)    # B C WD        
            y = []
            A_qk = torch.sigmoid(gates) # B T C WD+1
            A_qk = A_qk/torch.sum(A_qk,-1,keepdim=True)
            nl = F.pad(A_qk[:,:,:,-1:]*v[:,:,:].unsqueeze(-1),(0,self.window_dim-1)) # B T C WD
            A_qk = self.A_temp + F.pad(A_qk[:,:,:,:-1].unsqueeze(-1),(0,self.window_dim-1)) # B T C WD WD

            for i in range(T):
                
                hidden_x = torch.einsum('bci, bcij -> bcj',hidden_x,A_qk[:,i,:,:,:])+ nl[:,i,:,:]
                y.append(hidden_x) 

            y=torch.stack(y, dim=1)
            #print(y.shape)
            y=y.reshape(B,T,self.hidden_dim*self.window_dim)

        elif self.implementation == "sigmoid_l1_save_dynamics":
            #b t c
            hidden_x = torch.zeros(B, self.hidden_dim, self.window_dim).to(hidden_states.device)    # B C WD        
            y = []
            A_qk = torch.sigmoid(gates) # B T C WD+1
            A_qk = A_qk/torch.sum(A_qk,-1,keepdim=True)
            nl = F.pad(A_qk[:,:,:,-1:]*v[:,:,:].unsqueeze(-1),(0,self.window_dim-1)) # B T C WD
            A_qk = self.A_temp + F.pad(A_qk[:,:,:,:-1].unsqueeze(-1),(0,self.window_dim-1)) # B T C WD WD

            self.hidden_dynamics = []
            self.hidden_dynamics.append(A_qk)

            for i in range(T):
                
                hidden_x = torch.einsum('bci, bcij -> bcj',hidden_x,A_qk[:,i,:,:,:])+ nl[:,i,:,:]
                y.append(hidden_x) 

                if hidden_x.requires_grad:
                    hidden_x.retain_grad()
                self.hidden_dynamics.append(hidden_x)

            y=torch.stack(y, dim=1)
            #print(y.shape)
            y=y.reshape(B,T,self.hidden_dim*self.window_dim)

        elif self.implementation == "sigmoid_l1_hopscan_custom":
            A_qk = torch.sigmoid(gates)  # B T C WD+1
            A_qk = A_qk / torch.sum(A_qk, -1, keepdim=True)
            A_qk, nl = _companion_A_and_b(self.A_temp, A_qk, v)
            y = hopscan_opt(nl, A_qk.transpose(-1, -2))  # B T C WD
            y = y.reshape(B, T, self.hidden_dim * self.window_dim)

        elif self.implementation == "relu_l1":
            #b t c
            hidden_x = torch.zeros(B, self.hidden_dim, self.window_dim).to(hidden_states.device)    # B C WD        
            y = []
            A_qk = torch.relu(gates) # B T C WD+1
            A_qk = A_qk/torch.sum(A_qk+0.001,-1,keepdim=True)
            nl = F.pad(A_qk[:,:,:,-1:]*v[:,:,:].unsqueeze(-1),(0,self.window_dim-1)) # B T C WD
            A_qk = self.A_temp + F.pad(A_qk[:,:,:,:-1].unsqueeze(-1),(0,self.window_dim-1)) # B T C WD WD

            for i in range(T):
                
                hidden_x = torch.einsum('bci, bcij -> bcj',hidden_x,A_qk[:,i,:,:,:])+ nl[:,i,:,:]
                y.append(hidden_x) 

            y=torch.stack(y, dim=1)
            #print(y.shape)
            y=y.reshape(B,T,self.hidden_dim*self.window_dim)

        elif self.implementation == "relu_l1_save_dynamics":
            #b t c
            hidden_x = torch.zeros(B, self.hidden_dim, self.window_dim).to(hidden_states.device)    # B C WD        
            y = []
            A_qk = torch.relu(gates) # B T C WD+1
            A_qk = A_qk/torch.sum(A_qk+0.001,-1,keepdim=True)
            nl = F.pad(A_qk[:,:,:,-1:]*v[:,:,:].unsqueeze(-1),(0,self.window_dim-1)) # B T C WD
            A_qk = self.A_temp + F.pad(A_qk[:,:,:,:-1].unsqueeze(-1),(0,self.window_dim-1)) # B T C WD WD

            self.hidden_dynamics = []
            self.hidden_dynamics.append(A_qk)

            for i in range(T):
                
                hidden_x = torch.einsum('bci, bcij -> bcj',hidden_x,A_qk[:,i,:,:,:])+ nl[:,i,:,:]
                y.append(hidden_x) 

                if hidden_x.requires_grad:
                    hidden_x.retain_grad()
                self.hidden_dynamics.append(hidden_x)

            y=torch.stack(y, dim=1)
            #print(y.shape)
            y=y.reshape(B,T,self.hidden_dim*self.window_dim)

        elif self.implementation == "relu_l1_hopscan_custom":
            A_qk = torch.relu(gates)  # B T C WD+1
            A_qk = A_qk / torch.sum(A_qk + 0.001, -1, keepdim=True)
            A_qk, nl = _companion_A_and_b(self.A_temp, A_qk, v)
            y = hopscan_opt(nl, A_qk.transpose(-1, -2))  # B T C WD
            y = y.reshape(B, T, self.hidden_dim * self.window_dim)

        #  
        elif self.implementation == "nonorm":
            #b t c
            hidden_x = torch.zeros(B, self.hidden_dim, self.window_dim).to(hidden_states.device)    # B C WD        
            y = []
            A_qk = gates# B T C WD+1
            # A_qk = A_qk/torch.sum(A_qk,-1,keepdim=True)
            nl = F.pad(A_qk[:,:,:,-1:]*v[:,:,:].unsqueeze(-1),(0,self.window_dim-1)) # B T C WD
            A_qk = self.A_temp + F.pad(A_qk[:,:,:,:-1].unsqueeze(-1),(0,self.window_dim-1)) # B T C WD WD

            for i in range(T):
                
                hidden_x = torch.einsum('bci, bcij -> bcj',hidden_x,A_qk[:,i,:,:,:])+ nl[:,i,:,:]
                y.append(hidden_x) 

            y=torch.stack(y, dim=1)
            #print(y.shape)
            y=y.reshape(B,T,self.hidden_dim*self.window_dim)

        elif self.implementation == "nonorm_save_dynamics":
            #b t c
            hidden_x = torch.zeros(B, self.hidden_dim, self.window_dim).to(hidden_states.device)    # B C WD        
            y = []
            A_qk = gates# B T C WD+1
            # A_qk = A_qk/torch.sum(A_qk,-1,keepdim=True)
            nl = F.pad(A_qk[:,:,:,-1:]*v[:,:,:].unsqueeze(-1),(0,self.window_dim-1)) # B T C WD
            A_qk = self.A_temp + F.pad(A_qk[:,:,:,:-1].unsqueeze(-1),(0,self.window_dim-1)) # B T C WD WD

            self.hidden_dynamics = []
            self.hidden_dynamics.append(A_qk)

            for i in range(T):
                
                hidden_x = torch.einsum('bci, bcij -> bcj',hidden_x,A_qk[:,i,:,:,:])+ nl[:,i,:,:]
                y.append(hidden_x) 

                if hidden_x.requires_grad:
                    hidden_x.retain_grad()
                self.hidden_dynamics.append(hidden_x)

            y=torch.stack(y, dim=1)
            #print(y.shape)
            y=y.reshape(B,T,self.hidden_dim*self.window_dim)
        
        elif self.implementation == "nonorm_hopscan_custom":
            A_qk = gates  # B T C WD+1
            A_qk, nl = _companion_A_and_b(self.A_temp, A_qk, v)
            y = hopscan_opt(nl, A_qk.transpose(-1, -2))  # B T C WD
            y = y.reshape(B, T, self.hidden_dim * self.window_dim)

        #  tanh
        elif self.implementation == "tanh_l1":
            #b t c
            hidden_x = torch.zeros(B, self.hidden_dim, self.window_dim).to(hidden_states.device)    # B C WD        
            y = []
            A_qk = torch.tanh(gates) # B T C WD+1
            A_qk = A_qk/torch.sum(torch.abs(A_qk),-1,keepdim=True)
            nl = F.pad(A_qk[:,:,:,-1:]*v[:,:,:].unsqueeze(-1),(0,self.window_dim-1)) # B T C WD
            A_qk = self.A_temp + F.pad(A_qk[:,:,:,:-1].unsqueeze(-1),(0,self.window_dim-1)) # B T C WD WD

            for i in range(T):
                
                hidden_x = torch.einsum('bci, bcij -> bcj',hidden_x,A_qk[:,i,:,:,:])+ nl[:,i,:,:]
                y.append(hidden_x) 

            y=torch.stack(y, dim=1)
            #print(y.shape)
            y=y.reshape(B,T,self.hidden_dim*self.window_dim)

        elif self.implementation == "tanh_l1_save_dynamics":
            #b t c
            hidden_x = torch.zeros(B, self.hidden_dim, self.window_dim).to(hidden_states.device)    # B C WD        
            y = []
            A_qk = torch.tanh(gates) # B T C WD+1
            A_qk = A_qk/torch.sum(torch.abs(A_qk),-1,keepdim=True)
            nl = F.pad(A_qk[:,:,:,-1:]*v[:,:,:].unsqueeze(-1),(0,self.window_dim-1)) # B T C WD
            A_qk = self.A_temp + F.pad(A_qk[:,:,:,:-1].unsqueeze(-1),(0,self.window_dim-1)) # B T C WD WD

            self.hidden_dynamics = []
            self.hidden_dynamics.append(A_qk)

            for i in range(T):
                
                hidden_x = torch.einsum('bci, bcij -> bcj',hidden_x,A_qk[:,i,:,:,:])+ nl[:,i,:,:]
                y.append(hidden_x) 

                if hidden_x.requires_grad:
                    hidden_x.retain_grad()
                self.hidden_dynamics.append(hidden_x)

            y=torch.stack(y, dim=1)
            #print(y.shape)
            y=y.reshape(B,T,self.hidden_dim*self.window_dim)

        elif self.implementation == "tanh_l1_hopscan_custom":
            A_qk = torch.tanh(gates)  # B T C WD+1
            A_qk = A_qk / torch.sum(torch.abs(A_qk), -1, keepdim=True)
            A_qk, nl = _companion_A_and_b(self.A_temp, A_qk, v)
            y = hopscan_opt(nl, A_qk.transpose(-1, -2))  # B T C WD
            y = y.reshape(B, T, self.hidden_dim * self.window_dim)

        # linl1
        elif self.implementation == "lin_l1":
            #b t c
            hidden_x = torch.zeros(B, self.hidden_dim, self.window_dim).to(hidden_states.device)    # B C WD        
            y = []
            A_qk = gates # B T C WD+1
            A_qk = A_qk/torch.sum(torch.abs(A_qk),-1,keepdim=True)
            nl = F.pad(A_qk[:,:,:,-1:]*v[:,:,:].unsqueeze(-1),(0,self.window_dim-1)) # B T C WD
            A_qk = self.A_temp + F.pad(A_qk[:,:,:,:-1].unsqueeze(-1),(0,self.window_dim-1)) # B T C WD WD

            for i in range(T):
                
                hidden_x = torch.einsum('bci, bcij -> bcj',hidden_x,A_qk[:,i,:,:,:])+ nl[:,i,:,:]
                y.append(hidden_x) 

            y=torch.stack(y, dim=1)
            #print(y.shape)
            y=y.reshape(B,T,self.hidden_dim*self.window_dim)

        elif self.implementation == "lin_l1_save_dynamics":
            #b t c
            hidden_x = torch.zeros(B, self.hidden_dim, self.window_dim).to(hidden_states.device)    # B C WD        
            y = []
            A_qk = gates # B T C WD+1
            A_qk = A_qk/torch.sum(torch.abs(A_qk),-1,keepdim=True)
            nl = F.pad(A_qk[:,:,:,-1:]*v[:,:,:].unsqueeze(-1),(0,self.window_dim-1)) # B T C WD
            A_qk = self.A_temp + F.pad(A_qk[:,:,:,:-1].unsqueeze(-1),(0,self.window_dim-1)) # B T C WD WD

            self.hidden_dynamics = []
            self.hidden_dynamics.append(A_qk)

            for i in range(T):
                
                hidden_x = torch.einsum('bci, bcij -> bcj',hidden_x,A_qk[:,i,:,:,:])+ nl[:,i,:,:]
                y.append(hidden_x) 

                if hidden_x.requires_grad:
                    hidden_x.retain_grad()
                self.hidden_dynamics.append(hidden_x)

            y=torch.stack(y, dim=1)
            #print(y.shape)
            y=y.reshape(B,T,self.hidden_dim*self.window_dim)

        elif self.implementation == "lin_l1_hopscan_custom":
            A_qk = gates  # B T C WD+1
            A_qk = A_qk / torch.sum(torch.abs(A_qk), -1, keepdim=True)
            A_qk, nl = _companion_A_and_b(self.A_temp, A_qk, v)
            y = hopscan_opt(nl, A_qk.transpose(-1, -2))  # B T C WD
            y = y.reshape(B, T, self.hidden_dim * self.window_dim)

        else: 
            raise ValueError(f"Parallel implementation {self.implementation} not supported")

        return self.proj_out(y)
