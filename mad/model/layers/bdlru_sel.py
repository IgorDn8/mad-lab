"""Selective block-diagonal LRU layer.

This variant uses input-dependent gates to select the block-diagonal recurrent
state mixing at each sequence position. It corresponds to the formerly named
``BDLRU_o1`` implementation.
"""

import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange
import sys

try:
    from mad.model.layers.ops.scans.hopscan import hopscan_opt
except ImportError:
    hopscan_opt = None

try:
    from mad.model.layers.ops.scans.affine_scan import scan_parallel as affine_scan_parallel
except ImportError:
    affine_scan_parallel = None

try:
    from mad.model.layers.ops.scans.triton_scans import triton_affine_scan
except ImportError:
    triton_affine_scan = None

try:
    from mad.model.layers.ops.scans.triton_fused_gate_scan import (
        fused_gate_scan, fused_gate_supported,
    )
except ImportError:
    fused_gate_scan = None

    def fused_gate_supported(block):  # noqa: D103
        return False

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
}


def _hopscan_custom_recurrence(gates: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """BD-LRU ``hopscan_custom`` recurrence: softmax gating + custom hopscan scan.

    Returns the pre-projection hidden state (B, T, N, m). Wrapped by
    ``torch.compile`` below to form the ``custom_hopscan_autotune`` variant.
    """
    A_t = torch.softmax(gates, -1)          # B T N m m+1
    a0v = A_t[..., -1] * v                   # B T N m
    A_t = A_t[..., :-1]                      # B T N m m
    return hopscan_opt(a0v, A_t)             # B T N m


# max-autotune build of the recurrence above (torch.compile is lazy, so importing
# this module never triggers a compile). None when hopscan_opt is unavailable.
_hopscan_custom_recurrence_autotune = (
    torch.compile(_hopscan_custom_recurrence, mode="max-autotune", dynamic=False)
    if hopscan_opt is not None else None
)


def _triton_gate_inputs(gates: torch.Tensor, v: torch.Tensor, window_dim: int):
    """Softmax gating -> the (batch*hidden, T, m, m) layout the triton scans take.

    Profiling the plain triton path showed the scan kernel is only ~4% of the
    step while ~70% goes to eager elementwise launches: the softmax, the gated
    input, and above all the ``permute(...).reshape(...)`` on the two largest
    tensors, each of which materialises a fresh copy (plus its mirror in the
    backward). Isolating that chain here lets inductor fuse it.
    """
    B, T, N = gates.shape[0], gates.shape[1], gates.shape[2]
    A_t = torch.softmax(gates, -1)                      # B T N m m+1
    a0v = A_t[..., -1] * v                              # B T N m
    A_t = A_t[..., :-1]                                 # B T N m m
    BB = B * N
    A_bb = A_t.permute(0, 2, 1, 3, 4).reshape(BB, T, window_dim, window_dim)
    b_bb = a0v.permute(0, 2, 1, 3).reshape(BB, T, window_dim)
    return A_bb, b_bb


def _triton_scan_output(y: torch.Tensor, B: int, T: int, N: int, window_dim: int):
    """(BB, T, m) scan output -> (B, T, N*m) for the out projection."""
    y = y.reshape(B, N, T, window_dim).permute(0, 2, 1, 3)
    return y.reshape(B, T, N * window_dim)


_triton_gate_inputs_autotune = torch.compile(
    _triton_gate_inputs, mode="max-autotune", dynamic=False)
_triton_scan_output_autotune = torch.compile(
    _triton_scan_output, mode="max-autotune", dynamic=False)

# @torch.compile(mode="max-autotune", dynamic=False)
class BDLRU_sel(nn.Module):
    """BD-LRU selective variant with token-dependent recurrent transitions."""

    def __init__(
        self,
        dim: int,
        window_dim: int = 1,
        eps: float = 1e-12,
        implementation: str="parallel",
        hidden_dim: int = 5,
        **kwargs
    ):
        super().__init__()
        
        self.dim = dim
        self.window_dim = window_dim
        self.eps = eps
        self.implementation = implementation
        self.hidden_dim = hidden_dim

        # initialize projections and feature map
        self.proj_gates = nn.Linear(self.dim, self.hidden_dim*self.window_dim*(self.window_dim+1), bias=True)
        self.proj_v = nn.Linear(self.dim, self.hidden_dim*self.window_dim, bias=False)
        self.proj_out = nn.Linear(self.hidden_dim*self.window_dim, self.dim, bias=False)

        self.hidden_dynamics = []

    
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
         
        v = self.proj_v(hidden_states) # B T N*H
        v = v.reshape(B,T,self.hidden_dim,self.window_dim)

        gates = self.proj_gates(hidden_states) # B T N*H*(H+1)
        gates = gates.reshape(B,T,self.hidden_dim,self.window_dim,self.window_dim+1)

        if self.implementation == "orig":
            #b h t c
            hidden_x = torch.zeros(B, self.hidden_dim, self.window_dim).to(hidden_states.device)      # B H       
            y = []
            #print(hidden_x.shape)
            A_qk = torch.softmax(gates,-1) # B T N H H+1
            nl = A_qk[:,:,:,:,-1]*v[:,:,:,:] # B T N H 
            A_qk = A_qk[:,:,:,:,:-1] # B T N H H

            for i in range(T):
                
                # check order
                hidden_x = torch.einsum('bci, bcji -> bcj',hidden_x,A_qk[:,i,:,:,:])+ nl[:,i,:,:]
                y.append(hidden_x) 
                # y.append(hidden_x) 
            #print(A_window)
            y=torch.stack(y, dim=1).reshape(B, T, self.hidden_dim*self.window_dim) # B T N*H
            y=self.proj_out(y)
            #import sys
            #sys.exit(0)
        elif self.implementation == "orig_save_dynamics":
            #b h t c
            hidden_x = torch.zeros(B, self.hidden_dim, self.window_dim).to(hidden_states.device)      # B H       
            y = []
            #print(hidden_x.shape)
            A_qk = torch.softmax(gates,-1) # B T N H H+1
            nl = A_qk[:,:,:,:,-1]*v[:,:,:,:] # B T N H 
            A_qk = A_qk[:,:,:,:,:-1] # B T N H H

            self.hidden_dynamics = []
            self.hidden_dynamics.append(A_qk)
            for i in range(T):
                
                # check order
                hidden_x = torch.einsum('bci, bcji -> bcj',hidden_x,A_qk[:,i,:,:,:])+ nl[:,i,:,:]
                y.append(hidden_x) 
                # y.append(hidden_x) 

                # hidden_x.requires_grad = True
                if hidden_x.requires_grad:
                    hidden_x.retain_grad()
                self.hidden_dynamics.append(hidden_x)
            #print(A_window)
            y=torch.stack(y, dim=1).reshape(B, T, self.hidden_dim*self.window_dim) # B T N*H
            y=self.proj_out(y)
            #import sys
            #sys.exit(0)

        elif self.implementation == "sigmoid":
                        #b h t c
            hidden_x = torch.zeros(B, C, self.window_dim).to(hidden_states.device)    # B C WD        
            y = []
            #print(hidden_x.shape)
            
            for i in range(self.window_dim,self.window_dim+T):

                A_qk = torch.sigmoid(q[:,i,:,:]) # B C WD 
                B_norm = F.pad((1 - torch.sum(A_qk,-1)/self.window_dim).unsqueeze(-1),(0,self.window_dim-1))

                A_qk = F.pad(A_qk.unsqueeze(-1),(0,self.window_dim-1))  # B C WD WD
                A_qk = self.A_temp+A_qk
                #A_qk = A_qk.reshape(B*C,T+self.window_dim, self.window_dim, self.window_dim)
                
                hidden_x = torch.einsum('bci, bcij -> bcj',hidden_x,A_qk) + B_norm*v[:,:,:,i]*nl[:,:,:,i]
                #hidden_x = torch.einsum('bi, bij -> bj',hidden_x,A_qk[:,i,:,:])+ v[:,:,i]*nl[:,:,i] 
                # import sys
                # print(v[:,:,:,i]*nl[:,:,:,i])
                # sys.exit(0)                             
                y.append(hidden_x[:,:,0]) 
                # y.append(hidden_x[:,0].reshape(B,C)) 
            #print(A_window)
            y=torch.stack(y, dim=1)
            #y = y.reshape(B,C,T).transpose(1,2)
            #y = rearrange(y, 'b h l d -> b l (h d)')
            #print( (hidden_x@A_qk[:,i,:,:]).shape)
            #import sys
            #sys.exit(0)
        #     
        #  
        elif self.implementation == "hopscan_custom":
            
            # softmax normalization of coeff A and a_0
            A_t = torch.softmax(gates,-1) # B T N m m+1
            # gated input a_0*v 
            a0v = A_t[:,:,:,:,-1]*v[:,:,:,:] # B T N m 
            # transition matrix A_t 
            A_t = A_t[:,:,:,:,:-1] # B T N m m

            # parallel scan
            y=hopscan_opt(a0v, A_t) # B T N m

            # reshape back 
            y=y.reshape(B,T,self.hidden_dim*self.window_dim) # B T N*m

            # out projection from hidden state (later it goes to mlp)
            y=self.proj_out(y) # B T N

        elif self.implementation == "custom_hopscan_autotune":

            # identical recurrence to `hopscan_custom`, but the gating + custom
            # hopscan scan is wrapped in torch.compile(mode="max-autotune",
            # dynamic=False) so inductor can fuse the surrounding ops.
            if _hopscan_custom_recurrence_autotune is None:
                raise ImportError("hopscan_opt unavailable; cannot build custom_hopscan_autotune")
            y = _hopscan_custom_recurrence_autotune(gates, v) # B T N m

            # reshape back
            y = y.reshape(B,T,self.hidden_dim*self.window_dim) # B T N*m

            # out projection from hidden state (later it goes to mlp)
            y = self.proj_out(y) # B T N

        elif self.implementation == "affine_scan_torch_impl":

            # softmax normalization of coeff A and a_0 (same gating as `orig`)
            A_t = torch.softmax(gates,-1) # B T N m m+1
            # gated input a_0*v
            a0v = A_t[:,:,:,:,-1]*v[:,:,:,:] # B T N m
            # transition matrix A_t
            A_t = A_t[:,:,:,:,:-1] # B T N m m

            # parallel affine scan (associative_scan over time, autograd through HOP).
            # scan_parallel(W, x) solves h_t = W_t @ h_{t-1} + x_t.
            y=affine_scan_parallel(A_t, a0v) # B T N m

            # reshape back
            y=y.reshape(B,T,self.hidden_dim*self.window_dim) # B T N*m

            # out projection from hidden state (later it goes to mlp)
            y=self.proj_out(y) # B T N

        elif self.implementation in _TRITON_SCAN_MODES:

            # softmax normalization of coeff A and a_0 (same gating as `orig`)
            A_t = torch.softmax(gates,-1) # B T N m m+1
            # gated input a_0*v
            a0v = A_t[:,:,:,:,-1]*v[:,:,:,:] # B T N m
            # transition matrix A_t
            A_t = A_t[:,:,:,:,:-1] # B T N m m

            # triton affine scan expects a flattened (batch*block) layout
            BB = B*self.hidden_dim
            A_bb = A_t.permute(0,2,1,3,4).reshape(BB, T, self.window_dim, self.window_dim)
            b_bb = a0v.permute(0,2,1,3).reshape(BB, T, self.window_dim)

            y = triton_affine_scan(A_bb, b_bb, _TRITON_SCAN_MODES[self.implementation]) # BB T m

            # reshape back
            y = y.reshape(B, self.hidden_dim, T, self.window_dim).permute(0,2,1,3)
            y = y.reshape(B, T, self.hidden_dim*self.window_dim) # B T N*m

            # out projection from hidden state (later it goes to mlp)
            y = self.proj_out(y) # B T N

        elif self.implementation in _TRITON_COMPILED_MODES:

            # identical math to the branch above; the gating and layout chains
            # are compiled, and the scan stays an opaque op between them.
            A_bb, b_bb = _triton_gate_inputs_autotune(gates, v, self.window_dim)

            y = triton_affine_scan(A_bb, b_bb,
                                   _TRITON_COMPILED_MODES[self.implementation]) # BB T m

            y = _triton_scan_output_autotune(
                y, B, T, self.hidden_dim, self.window_dim) # B T N*m

            # out projection from hidden state (later it goes to mlp)
            y = self.proj_out(y) # B T N

        elif self.implementation == "triton_fused_gates":

            # gating computed inside the scan kernel, so neither the softmax
            # output nor A is ever materialised, and the (B,T,N,m) result makes
            # the reshape below a free view. Widths without a kernel fall back to
            # the compiled A-based path above.
            if fused_gate_scan is not None and fused_gate_supported(self.window_dim):
                y = fused_gate_scan(gates, v) # B T N m
                y = y.reshape(B, T, self.hidden_dim*self.window_dim) # B T N*m
            else:
                A_bb, b_bb = _triton_gate_inputs_autotune(gates, v, self.window_dim)
                y = triton_affine_scan(A_bb, b_bb, "auto_v2") # BB T m
                y = _triton_scan_output_autotune(
                    y, B, T, self.hidden_dim, self.window_dim) # B T N*m

            # out projection from hidden state (later it goes to mlp)
            y = self.proj_out(y) # B T N

        elif self.implementation == "sigmoid_l1":

            #b t n*h
            #b h t c
            hidden_x = torch.zeros(B, self.hidden_dim, self.window_dim).to(hidden_states.device)      # B H       
            y = []
            #print(hidden_x.shape)
            A_qk = torch.sigmoid(gates) # B T N H H+1
            A_qk = A_qk/torch.sum(A_qk,-1,keepdim=True)
            nl = A_qk[:,:,:,:,-1]*v[:,:,:,:] # B T N H 
            A_qk = A_qk[:,:,:,:,:-1] # B T N H H

            for i in range(T):
                
                # check order
                hidden_x = torch.einsum('bci, bcji -> bcj',hidden_x,A_qk[:,i,:,:,:])+ nl[:,i,:,:]
                y.append(hidden_x) 
                # y.append(hidden_x) 
            #print(A_window)
            y=torch.stack(y, dim=1).reshape(B, T, self.hidden_dim*self.window_dim) # B T N*H
            y=self.proj_out(y)

        elif self.implementation == "sigmoid_l1_save_dynamics":

            #b t n*h
            #b h t c
            hidden_x = torch.zeros(B, self.hidden_dim, self.window_dim).to(hidden_states.device)      # B H       
            y = []
            #print(hidden_x.shape)
            A_qk = torch.sigmoid(gates) # B T N H H+1
            A_qk = A_qk/torch.sum(A_qk,-1,keepdim=True)
            nl = A_qk[:,:,:,:,-1]*v[:,:,:,:] # B T N H 
            A_qk = A_qk[:,:,:,:,:-1] # B T N H H

            self.hidden_dynamics = []
            self.hidden_dynamics.append(A_qk)

            for i in range(T):
                
                # check order
                hidden_x = torch.einsum('bci, bcji -> bcj',hidden_x,A_qk[:,i,:,:,:])+ nl[:,i,:,:]
                y.append(hidden_x) 

                if hidden_x.requires_grad:
                    hidden_x.retain_grad()
                self.hidden_dynamics.append(hidden_x)
                # y.append(hidden_x) 
            #print(A_window)
            y=torch.stack(y, dim=1).reshape(B, T, self.hidden_dim*self.window_dim) # B T N*H
            y=self.proj_out(y)

        elif self.implementation == "sigmoid_l1_hopscan_custom":

            #b t n*h
            A_qk = torch.sigmoid(gates) # B T N H H+1
            A_qk = A_qk/torch.sum(A_qk,-1,keepdim=True)
            nl = A_qk[:,:,:,:,-1]*v[:,:,:,:] # B T N H 
            A_qk = A_qk[:,:,:,:,:-1] # B T N H H


            y=hopscan_opt(nl, A_qk) # B T N H

            # reshape back 
            y=y.reshape(B,T,self.hidden_dim*self.window_dim) # B T N*H

            y=self.proj_out(y)

        elif self.implementation == "relu_l1":

            #b t n*h
            #b h t c
            hidden_x = torch.zeros(B, self.hidden_dim, self.window_dim).to(hidden_states.device)      # B H       
            y = []
            #print(hidden_x.shape)
            A_qk = torch.relu(gates) # B T N H H+1
            A_qk = A_qk/torch.sum(A_qk+0.001,-1,keepdim=True)
            nl = A_qk[:,:,:,:,-1]*v[:,:,:,:] # B T N H 
            A_qk = A_qk[:,:,:,:,:-1] # B T N H H

            for i in range(T):
                
                # check order
                hidden_x = torch.einsum('bci, bcji -> bcj',hidden_x,A_qk[:,i,:,:,:])+ nl[:,i,:,:]
                y.append(hidden_x) 
                # y.append(hidden_x) 
            #print(A_window)
            y=torch.stack(y, dim=1).reshape(B, T, self.hidden_dim*self.window_dim) # B T N*H
            y=self.proj_out(y)

        elif self.implementation == "relu_l1_save_dynamics":

            #b t n*h
            #b h t c
            hidden_x = torch.zeros(B, self.hidden_dim, self.window_dim).to(hidden_states.device)      # B H       
            y = []
            #print(hidden_x.shape)
            A_qk = torch.relu(gates) # B T N H H+1
            A_qk = A_qk/torch.sum(A_qk+0.001,-1,keepdim=True)
            # A_qk = torch.tanh(gates) # B T N H H+1
            # A_qk = 4*A_qk/torch.sum(torch.abs(A_qk),-1,keepdim=True)
            nl = A_qk[:,:,:,:,-1]*v[:,:,:,:] # B T N H 
            A_qk = A_qk[:,:,:,:,:-1] # B T N H H

            self.hidden_dynamics = []
            self.hidden_dynamics.append(A_qk)

            for i in range(T):
                
                # check order
                hidden_x = torch.einsum('bci, bcji -> bcj',hidden_x,A_qk[:,i,:,:,:])+ nl[:,i,:,:]
                y.append(hidden_x) 

                if hidden_x.requires_grad:
                    hidden_x.retain_grad()
                self.hidden_dynamics.append(hidden_x)
                # y.append(hidden_x) 
            #print(A_window)
            y=torch.stack(y, dim=1).reshape(B, T, self.hidden_dim*self.window_dim) # B T N*H
            y=self.proj_out(y)

        elif self.implementation == "relu_l1_hopscan_custom":

            #b t n*h

            A_qk = torch.relu(gates) # B T N H H+1
            A_qk = A_qk/torch.sum(A_qk+0.001,-1,keepdim=True)
            nl = A_qk[:,:,:,:,-1]*v[:,:,:,:] # B T N H 
            A_qk = A_qk[:,:,:,:,:-1] # B T N H H


            y=hopscan_opt(nl, A_qk) # B T N H

            # reshape back 
            y=y.reshape(B,T,self.hidden_dim*self.window_dim) # B T N*H

            y=self.proj_out(y.real)

        elif self.implementation == "nonorm":

            #b t n*h
            #b h t c
            hidden_x = torch.zeros(B, self.hidden_dim, self.window_dim).to(hidden_states.device)      # B H       
            y = []
            #print(hidden_x.shape)
            A_qk = gates # B T N H H+1
            #A_qk = A_qk/torch.sum(A_qk,-1,keepdim=True)
            nl = A_qk[:,:,:,:,-1]*v[:,:,:,:] # B T N H 
            A_qk = A_qk[:,:,:,:,:-1] # B T N H H

            for i in range(T):
                
                # check order
                hidden_x = torch.einsum('bci, bcji -> bcj',hidden_x,A_qk[:,i,:,:,:])+ nl[:,i,:,:]
                y.append(hidden_x) 
                # y.append(hidden_x) 
            #print(A_window)
            y=torch.stack(y, dim=1).reshape(B, T, self.hidden_dim*self.window_dim) # B T N*H
            y=self.proj_out(y)

        elif self.implementation == "nonorm_save_dynamics":

            #b t n*h
            #b h t c
            hidden_x = torch.zeros(B, self.hidden_dim, self.window_dim).to(hidden_states.device)      # B H       
            y = []
            #print(hidden_x.shape)
            A_qk = gates # B T N H H+1
            #A_qk = A_qk/torch.sum(A_qk,-1,keepdim=True)
            nl = A_qk[:,:,:,:,-1]*v[:,:,:,:] # B T N H 
            A_qk = A_qk[:,:,:,:,:-1] # B T N H H

            self.hidden_dynamics = []
            self.hidden_dynamics.append(A_qk)

            for i in range(T):
                
                # check order
                hidden_x = torch.einsum('bci, bcji -> bcj',hidden_x,A_qk[:,i,:,:,:])+ nl[:,i,:,:]
                y.append(hidden_x) 

                if hidden_x.requires_grad:
                    hidden_x.retain_grad()
                self.hidden_dynamics.append(hidden_x)
                # y.append(hidden_x) 
            #print(A_window)
            y=torch.stack(y, dim=1).reshape(B, T, self.hidden_dim*self.window_dim) # B T N*H
            y=self.proj_out(y)

        elif self.implementation == "nonorm_hopscan_custom":

            #b t n*h
            A_qk = gates# B T N H H+1
            #A_qk = A_qk/torch.sum(A_qk,-1,keepdim=True)
            nl = A_qk[:,:,:,:,-1]*v[:,:,:,:] # B T N H 
            A_qk = A_qk[:,:,:,:,:-1] # B T N H H


            y=hopscan_opt(nl, A_qk) # B T N H

            # reshape back 
            y=y.reshape(B,T,self.hidden_dim*self.window_dim) # B T N*H

            y=self.proj_out(y)

        elif self.implementation == "tanh_l1":

            #b t n*h
            #b h t c
            hidden_x = torch.zeros(B, self.hidden_dim, self.window_dim).to(hidden_states.device)      # B H       
            y = []
            #print(hidden_x.shape)
            A_qk = torch.tanh(gates) # B T N H H+1
            A_qk = A_qk/torch.sum(torch.abs(A_qk),-1,keepdim=True)
            nl = A_qk[:,:,:,:,-1]*v[:,:,:,:] # B T N H 
            A_qk = A_qk[:,:,:,:,:-1] # B T N H H

            for i in range(T):
                
                # check order
                hidden_x = torch.einsum('bci, bcji -> bcj',hidden_x,A_qk[:,i,:,:,:])+ nl[:,i,:,:]
                y.append(hidden_x) 
                # y.append(hidden_x) 
            #print(A_window)
            y=torch.stack(y, dim=1).reshape(B, T, self.hidden_dim*self.window_dim) # B T N*H
            y=self.proj_out(y)

        elif self.implementation == "tanh_l1_save_dynamics":

            #b t n*h
            #b h t c
            hidden_x = torch.zeros(B, self.hidden_dim, self.window_dim).to(hidden_states.device)      # B H       
            y = []
            #print(hidden_x.shape)
            A_qk = torch.tanh(gates) # B T N H H+1
            A_qk = A_qk/torch.sum(torch.abs(A_qk),-1,keepdim=True)
            nl = A_qk[:,:,:,:,-1]*v[:,:,:,:] # B T N H 
            A_qk = A_qk[:,:,:,:,:-1] # B T N H H

            self.hidden_dynamics = []
            self.hidden_dynamics.append(A_qk)

            for i in range(T):
                
                # check order
                hidden_x = torch.einsum('bci, bcji -> bcj',hidden_x,A_qk[:,i,:,:,:])+ nl[:,i,:,:]
                y.append(hidden_x) 

                if hidden_x.requires_grad:
                    hidden_x.retain_grad()
                self.hidden_dynamics.append(hidden_x)
                # y.append(hidden_x) 
            #print(A_window)
            y=torch.stack(y, dim=1).reshape(B, T, self.hidden_dim*self.window_dim) # B T N*H
            y=self.proj_out(y)

        elif self.implementation == "tanh_l1_hopscan_custom":

            #b t n*h
            A_qk = torch.tanh(gates) # B T N H H+1
            A_qk = A_qk/torch.sum(torch.abs(A_qk),-1,keepdim=True)
            nl = A_qk[:,:,:,:,-1]*v[:,:,:,:] # B T N H 
            A_qk = A_qk[:,:,:,:,:-1] # B T N H H


            y=hopscan_opt(nl, A_qk) # B T N H

            # reshape back 
            y=y.reshape(B,T,self.hidden_dim*self.window_dim) # B T N*H

            y=self.proj_out(y)

        elif self.implementation == "lin_l1":

            #b t n*h
            #b h t c
            hidden_x = torch.zeros(B, self.hidden_dim, self.window_dim).to(hidden_states.device)      # B H       
            y = []
            #print(hidden_x.shape)
            A_qk = gates # B T N H H+1
            A_qk = A_qk/torch.sum(torch.abs(A_qk),-1,keepdim=True)
            nl = A_qk[:,:,:,:,-1]*v[:,:,:,:] # B T N H 
            A_qk = A_qk[:,:,:,:,:-1] # B T N H H

            for i in range(T):
                
                # check order
                hidden_x = torch.einsum('bci, bcji -> bcj',hidden_x,A_qk[:,i,:,:,:])+ nl[:,i,:,:]
                y.append(hidden_x) 
                # y.append(hidden_x) 
            #print(A_window)
            y=torch.stack(y, dim=1).reshape(B, T, self.hidden_dim*self.window_dim) # B T N*H
            y=self.proj_out(y)

        elif self.implementation == "lin_l1_save_dynamics":

            #b t n*h
            #b h t c
            hidden_x = torch.zeros(B, self.hidden_dim, self.window_dim).to(hidden_states.device)      # B H       
            y = []
            #print(hidden_x.shape)
            A_qk = gates # B T N H H+1
            A_qk = A_qk/torch.sum(torch.abs(A_qk),-1,keepdim=True)
            nl = A_qk[:,:,:,:,-1]*v[:,:,:,:] # B T N H 
            A_qk = A_qk[:,:,:,:,:-1] # B T N H H

            self.hidden_dynamics = []
            self.hidden_dynamics.append(A_qk)

            for i in range(T):
                
                # check order
                hidden_x = torch.einsum('bci, bcji -> bcj',hidden_x,A_qk[:,i,:,:,:])+ nl[:,i,:,:]
                y.append(hidden_x) 

                if hidden_x.requires_grad:
                    hidden_x.retain_grad()
                self.hidden_dynamics.append(hidden_x)
                # y.append(hidden_x) 
            #print(A_window)
            y=torch.stack(y, dim=1).reshape(B, T, self.hidden_dim*self.window_dim) # B T N*H
            y=self.proj_out(y)

        elif self.implementation == "lin_l1_hopscan_custom":

            #b t n*h
            A_qk = gates # B T N H H+1
            A_qk = A_qk/torch.sum(torch.abs(A_qk),-1,keepdim=True)
            nl = A_qk[:,:,:,:,-1]*v[:,:,:,:] # B T N H 
            A_qk = A_qk[:,:,:,:,:-1] # B T N H H


            y=hopscan_opt(nl, A_qk) # B T N H

            # reshape back 
            y=y.reshape(B,T,self.hidden_dim*self.window_dim) # B T N*H

            y=self.proj_out(y)

        else: 
            raise ValueError(f"Parallel implementation {self.implementation} not supported")

        return y.to(hidden_states.dtype)
