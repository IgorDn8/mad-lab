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
    from mad.model.layers.ops.scans.hopscan import hopscan, hopscan_opt
except ImportError:
    hopscan, hopscan_opt = None, None

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
            
        elif self.implementation == "hopscan":
            #b h t c
            #hidden_x = torch.zeros(B, self.hidden_dim, self.window_dim).to(hidden_states.device)    # B C WD        
            #print(hidden_x.shape)
            A_qk = torch.softmax(gates,-1) # B T C WD+1
            nl = F.pad(A_qk[:,:,:,-1:]*v[:,:,:].unsqueeze(-1),(0,self.window_dim-1)) # B T C WD
            A_qk = self.A_temp + F.pad(A_qk[:,:,:,:-1].unsqueeze(-1),(0,self.window_dim-1)) # B T C WD' WD

            # prepare for hopscan
            nl=nl.permute(0,2,3,1).reshape(B*self.hidden_dim,self.window_dim,T) # B*C WD T
            A_qk = A_qk.permute(0,2,4,3,1).reshape(B*self.hidden_dim,self.window_dim,self.window_dim,T) # B*C WD WD' T 

            y=hopscan(nl, A_qk)[:,0,:] # B*C WD T

            # reshape back 
            y=y.reshape(B,self.hidden_dim,T).permute(0,2,1) 
        
        elif self.implementation == "hopscan_opt":

            # softmax normalization of coeff A and a_0
            A_t = torch.softmax(gates,-1) # B T N m+1
            # gated input a_0*v  (input vector is padded with zeros)
            a0v = F.pad(A_t[:,:,:,-1:]*v[:,:,:].unsqueeze(-1),(0,self.window_dim-1)) # B T N m
            # transition matrix A_t (to get companion form we pad and add A_temp which is structred 1-off diagonal matrix)
            A_t = self.A_temp + F.pad(A_t[:,:,:,:-1].unsqueeze(-1),(0,self.window_dim-1)) # B T N m m

            # parallel scan
            y=hopscan_opt(a0v, A_t) # B T N m

            # reshape back 
            y=y.reshape(B,T,self.hidden_dim*self.window_dim)

            #y=self.proj_out(y) # B T N

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

        elif self.implementation == "sigmoid_l1_hopscan_opt":
                        #b h t c
            A_qk = torch.sigmoid(gates) # B T C WD+1
            A_qk = A_qk/torch.sum(A_qk,-1,keepdim=True)
            nl = F.pad(A_qk[:,:,:,-1:]*v[:,:,:].unsqueeze(-1),(0,self.window_dim-1)) # B T C WD
            A_qk = self.A_temp + F.pad(A_qk[:,:,:,:-1].unsqueeze(-1),(0,self.window_dim-1)) # B T C WD' WD

            y=hopscan_opt(nl, A_qk) # B T C WD

            # reshape back 
            y=y.reshape(B,T,self.hidden_dim*self.window_dim)

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

        elif self.implementation == "relu_l1_hopscan_opt":
                        #b h t c
            A_qk = torch.relu(gates) # B T C WD+1
            A_qk = A_qk/torch.sum(A_qk+0.001,-1,keepdim=True)
            nl = F.pad(A_qk[:,:,:,-1:]*v[:,:,:].unsqueeze(-1),(0,self.window_dim-1)) # B T C WD
            A_qk = self.A_temp + F.pad(A_qk[:,:,:,:-1].unsqueeze(-1),(0,self.window_dim-1)) # B T C WD' WD

            y=hopscan_opt(nl, A_qk) # B T C WD

            # reshape back 
            y=y.reshape(B,T,self.hidden_dim*self.window_dim)

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
        
        elif self.implementation == "nonorm_hopscan_opt":
                        #b h t c
            A_qk = gates # B T C WD+1
            # A_qk = A_qk/torch.sum(A_qk,-1,keepdim=True)
            nl = F.pad(A_qk[:,:,:,-1:]*v[:,:,:].unsqueeze(-1),(0,self.window_dim-1)) # B T C WD
            A_qk = self.A_temp + F.pad(A_qk[:,:,:,:-1].unsqueeze(-1),(0,self.window_dim-1)) # B T C WD' WD

            y=hopscan_opt(nl, A_qk) # B T C WD

            # reshape back 
            y=y.reshape(B,T,self.hidden_dim*self.window_dim)

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

        elif self.implementation == "tanh_l1_hopscan_opt":
                        #b h t c
            A_qk = torch.tanh(gates) # B T C WD+1
            A_qk = A_qk/torch.sum(torch.abs(A_qk),-1,keepdim=True)
            nl = F.pad(A_qk[:,:,:,-1:]*v[:,:,:].unsqueeze(-1),(0,self.window_dim-1)) # B T C WD
            A_qk = self.A_temp + F.pad(A_qk[:,:,:,:-1].unsqueeze(-1),(0,self.window_dim-1)) # B T C WD' WD

            y=hopscan_opt(nl, A_qk) # B T C WD

            # reshape back 
            y=y.reshape(B,T,self.hidden_dim*self.window_dim)

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

        elif self.implementation == "lin_l1_hopscan_opt":
                        #b h t c
            A_qk = gates # B T C WD+1
            A_qk = A_qk/torch.sum(torch.abs(A_qk),-1,keepdim=True)
            nl = F.pad(A_qk[:,:,:,-1:]*v[:,:,:].unsqueeze(-1),(0,self.window_dim-1)) # B T C WD
            A_qk = self.A_temp + F.pad(A_qk[:,:,:,:-1].unsqueeze(-1),(0,self.window_dim-1)) # B T C WD' WD

            y=hopscan_opt(nl, A_qk) # B T C WD

            # reshape back 
            y=y.reshape(B,T,self.hidden_dim*self.window_dim)

        else: 
            raise ValueError(f"Parallel implementation {self.implementation} not supported")

        return self.proj_out(y)
