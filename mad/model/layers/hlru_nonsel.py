"""Non-selective higher-order LRU layer.

This variant uses learned recurrent parameters shared across all sequence
positions instead of input-dependent gates. It corresponds to the formerly named
``HLRU_o0_v1`` implementation.
"""

import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange

#@torch.compile(mode="max-autotune", dynamic=False)
class HLRU_nonsel(nn.Module):
    """H-LRU non-selective variant with fixed learned state mixing."""

    def __init__(
        self,
        dim: int,
        eps: float = 1e-12,
        implementation: str="parallel",
        window_dim: int = 16,
        hidden_dim: int = 128,
        **kwargs
    ):
        super().__init__()
        
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.window_dim = window_dim
        self.eps = eps
        self.implementation = implementation

        # initialize projections and feature map
        self.proj_v = nn.Linear(self.dim, self.hidden_dim, bias=False)
        #self.proj_out = nn.Linear(self.hidden_dim, self.dim, bias=False)

        self.register_parameter('A_w', torch.nn.Parameter(torch.rand(self.hidden_dim,self.window_dim+1))) # N WD+1
        self.register_buffer("A_temp", torch.diag(torch.ones(self.window_dim-1), 1))
        self.register_buffer("Id", torch.diag(torch.ones(self.window_dim)))

        # if self.implementation=="save_grads":
        #     self.hidden_grads = nn.ParameterList([torch.nn.Parameter(torch.zeros(self.max_length, self.hidden_dim,self.window_dim)) for _ in self.max_length])
        # else:
        self.hidden_dynamics = []
    
    def forward(self, 
        hidden_states: torch.Tensor,
        *args, **kwargs
    ):
        """
        x (torch.Tensor): tensor of shape (b, t, c)
        y (torch.Tensor): tensor of shape (b, t, c)
        """
        B, T, _ = hidden_states.size()

        v = self.proj_v(hidden_states)

        if self.implementation == "orig":
            #b h t c
            hidden_x = torch.zeros(B, self.hidden_dim, self.window_dim).to(hidden_states.device)    # B C WD        
            y = []

            A_qk = torch.softmax(self.A_w,-1) # N WD+1
            nl = F.pad(A_qk[:,-1:]*v[:,:,:].unsqueeze(-1),(0,self.window_dim-1)) # B T+WD C WD
            A_qk = self.A_temp + F.pad(A_qk[:,:-1].unsqueeze(-1),(0,self.window_dim-1)) # C WD WD 
            
            for i in range(T):

                hidden_x = torch.einsum('bci, cij -> bcj',hidden_x, A_qk) + nl[:,i,:,:]

                y.append(hidden_x[:,:,0]) 
            y=torch.stack(y, dim=1) # B T C

        elif self.implementation == "orig_save_dynamics":
            #b h t c
            hidden_x = torch.zeros(B, self.hidden_dim, self.window_dim).to(hidden_states.device)    # B C WD        
            y = []

            A_qk = torch.softmax(self.A_w,-1) # N WD+1
            nl = F.pad(A_qk[:,-1:]*v[:,:,:].unsqueeze(-1),(0,self.window_dim-1)) # B T+WD C WD
            A_qk = self.A_temp + F.pad(A_qk[:,:-1].unsqueeze(-1),(0,self.window_dim-1)) # C WD WD 
            
            self.hidden_dynamics = []
            for i in range(T):

                hidden_x = torch.einsum('bci, cij -> bcj',hidden_x, A_qk) + nl[:,i,:,:]

                # hidden_x.requires_grad = True
                if hidden_x.requires_grad:
                    hidden_x.retain_grad()
                self.hidden_dynamics.append(hidden_x)

                y.append(hidden_x[:,:,0]) 
            y=torch.stack(y, dim=1) # B T C

        elif self.implementation == "sigmoid_l1":
            #b h t c
            hidden_x = torch.zeros(B, self.hidden_dim, self.window_dim).to(hidden_states.device)    # B C WD        
            y = []

            A_qk = torch.sigmoid(self.A_w) # N WD+1
            A_qk = A_qk/torch.sum(A_qk,-1,keepdim=True)
            #A_qk = A_qk/A_qk.norm(p=2,dim=-2,keepdim=True)
            nl = F.pad(A_qk[:,-1:]*v[:,:,:].unsqueeze(-1),(0,self.window_dim-1)) # B T+WD C WD
            A_qk = self.A_temp + F.pad(A_qk[:,:-1].unsqueeze(-1),(0,self.window_dim-1)) # C WD WD 
            
            for i in range(T):

                hidden_x = torch.einsum('bci, cij -> bcj',hidden_x, A_qk) + nl[:,i,:,:]

                y.append(hidden_x[:,:,0]) 
            y=torch.stack(y, dim=1) # B T C

        elif self.implementation == "sigmoid_l1_save_dynamics":
            #b h t c
            hidden_x = torch.zeros(B, self.hidden_dim, self.window_dim).to(hidden_states.device)    # B C WD        
            y = []

            A_qk = torch.sigmoid(self.A_w) # N WD+1
            A_qk = A_qk/torch.sum(A_qk,-1,keepdim=True)
            nl = F.pad(A_qk[:,-1:]*v[:,:,:].unsqueeze(-1),(0,self.window_dim-1)) # B T+WD C WD
            A_qk = self.A_temp + F.pad(A_qk[:,:-1].unsqueeze(-1),(0,self.window_dim-1)) # C WD WD 
            
            self.hidden_dynamics = []
            for i in range(T):

                hidden_x = torch.einsum('bci, cij -> bcj',hidden_x, A_qk) + nl[:,i,:,:]

                # hidden_x.requires_grad = True
                if hidden_x.requires_grad:
                    hidden_x.retain_grad()
                self.hidden_dynamics.append(hidden_x)

                y.append(hidden_x[:,:,0]) 
            y=torch.stack(y, dim=1) # B T C

        elif self.implementation == "relu_l1":
            #b h t c
            hidden_x = torch.zeros(B, self.hidden_dim, self.window_dim).to(hidden_states.device)    # B C WD        
            y = []

            A_qk = F.relu(self.A_w) # N WD+1
            A_qk = A_qk/torch.sum(torch.abs(A_qk),-1,keepdim=True)
            nl = F.pad(A_qk[:,-1:]*v[:,:,:].unsqueeze(-1),(0,self.window_dim-1)) # B T+WD C WD
            A_qk = self.A_temp + F.pad(A_qk[:,:-1].unsqueeze(-1),(0,self.window_dim-1)) # C WD WD 
            
            for i in range(T):

                hidden_x = torch.einsum('bci, cij -> bcj',hidden_x, A_qk) + nl[:,i,:,:]

                y.append(hidden_x[:,:,0]) 
            y=torch.stack(y, dim=1) # B T C

        elif self.implementation == "relu_l1_save_dynamics":
            #b h t c
            hidden_x = torch.zeros(B, self.hidden_dim, self.window_dim).to(hidden_states.device)    # B C WD        
            y = []

            A_qk = F.relu(self.A_w) # N WD+1
            A_qk = A_qk/torch.sum(torch.abs(A_qk),-1,keepdim=True)
            nl = F.pad(A_qk[:,-1:]*v[:,:,:].unsqueeze(-1),(0,self.window_dim-1)) # B T+WD C WD
            A_qk = self.A_temp + F.pad(A_qk[:,:-1].unsqueeze(-1),(0,self.window_dim-1)) # C WD WD 
            
            self.hidden_dynamics = []
            for i in range(T):

                hidden_x = torch.einsum('bci, cij -> bcj',hidden_x, A_qk) + nl[:,i,:,:]

                # hidden_x.requires_grad = True
                if hidden_x.requires_grad:
                    hidden_x.retain_grad()
                self.hidden_dynamics.append(hidden_x)

                y.append(hidden_x[:,:,0]) 
            y=torch.stack(y, dim=1) # B T C
        
        elif self.implementation == "no_norm":
            #b h t c
            hidden_x = torch.zeros(B, self.hidden_dim, self.window_dim).to(hidden_states.device)    # B C WD        
            y = []

            A_qk = self.A_w/(self.window_dim+1) # N WD+1
            # A_qk = A_qk/torch.sum(A_qk,-1,keepdim=True)
            nl = F.pad(A_qk[:,-1:]*v[:,:,:].unsqueeze(-1),(0,self.window_dim-1)) # B T+WD C WD
            A_qk = self.A_temp + F.pad(A_qk[:,:-1].unsqueeze(-1),(0,self.window_dim-1)) # C WD WD 
            
            for i in range(T):

                hidden_x = torch.einsum('bci, cij -> bcj',hidden_x, A_qk) + nl[:,i,:,:]

                y.append(hidden_x[:,:,0]) 
            y=torch.stack(y, dim=1) # B T C

        elif self.implementation == "no_norm_save_dynamics":
            #b h t c
            hidden_x = torch.zeros(B, self.hidden_dim, self.window_dim).to(hidden_states.device)    # B C WD        
            y = []

            A_qk = self.A_w # N WD+1
            nl = F.pad(A_qk[:,-1:]*v[:,:,:].unsqueeze(-1),(0,self.window_dim-1)) # B T+WD C WD
            A_qk = self.A_temp + F.pad(A_qk[:,:-1].unsqueeze(-1),(0,self.window_dim-1)) # C WD WD 
            
            self.hidden_dynamics = []
            for i in range(T):

                hidden_x = torch.einsum('bci, cij -> bcj',hidden_x, A_qk) + nl[:,i,:,:]

                # hidden_x.requires_grad = True
                if hidden_x.requires_grad:
                    hidden_x.retain_grad()
                self.hidden_dynamics.append(hidden_x)

                y.append(hidden_x[:,:,0]) 
            y=torch.stack(y, dim=1) # B T C
        
        else: 
            raise ValueError(f"Parallel implementation {self.implementation} not supported")
        
        return y.to(hidden_states.dtype) #self.proj_out(y).to(hidden_states.dtype)
