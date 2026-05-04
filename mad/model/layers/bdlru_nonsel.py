"""Non-selective block-diagonal LRU layer.

This variant uses learned recurrent parameters shared across all sequence
positions instead of input-dependent gates. It corresponds to the formerly named
``BDLRU_o0_v1`` implementation.
"""

import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange

class BDLRU_nonsel(nn.Module):
    """BD-LRU non-selective variant with fixed learned state mixing."""

    def __init__(
        self,
        dim: int,
        window_dim: int = 1,
        eps: float = 1e-12,
        implementation: str="parallel",
        max_length: int = 256,
        hidden_dim: int = 10,
        **kwargs
    ):
        super().__init__()
        
        self.dim = dim
        self.window_dim = window_dim
        self.eps = eps
        self.implementation = implementation
        self.max_length= max_length
        self.hidden_dim= hidden_dim

        # initialize projections and feature map
        self.proj_v = nn.Linear(self.dim, self.window_dim*self.hidden_dim, bias=False)
        self.proj_out = nn.Linear(self.window_dim*self.hidden_dim, self.dim, bias=False)

        self.register_parameter('A_w', torch.nn.Parameter(torch.rand(self.hidden_dim,self.window_dim,self.window_dim+1))) # N H H+1


    
    def forward(self, 
        hidden_states: torch.Tensor,
        *args, **kwargs
    ):
        """
        x (torch.Tensor): tensor of shape (b, t, c)
        y (torch.Tensor): tensor of shape (b, t, c)
        """
        # q is gate, k is non-linear proj, v is linear proj
        B, T, C = hidden_states.size()
        assert T <= self.max_length, f'T is {T} but should be less than {self.max_length}'


        v = self.proj_v(hidden_states) # B T N*H
        v = v.reshape(B,T,self.hidden_dim,self.window_dim) 

        if self.implementation == "orig":
            #b h t c

            hidden_x = torch.zeros(B, self.hidden_dim, self.window_dim).to(hidden_states.device)   # [B H]*N       
            y = []

            A_qk = torch.softmax(self.A_w,-1) # N H H+1
            nl = A_qk[:,:,-1]*v # B T N H
            A_qk = A_qk[:,:,:-1] # N H H
            
            for i in range(T):

                hidden_x = torch.einsum('bni, nji -> bnj',hidden_x, A_qk) + nl[:,i,:,:]

                y.append(hidden_x) 
            
            y=torch.stack(y, dim=1).reshape(B, T, self.hidden_dim*self.window_dim) # B T N*H
            y=self.proj_out(y) # B T C
            #y = rearrange(y, 'b h l d -> b l (h d)')
        elif self.implementation == "sigmoid_l1":
             #b h t c
            hidden_x = torch.zeros(B, self.hidden_dim, self.window_dim).to(hidden_states.device)   # [B H]*N       
            y = []

            A_qk = torch.sigmoid(self.A_w) # N H H+1
            A_qk = A_qk/torch.sum(A_qk,-1,keepdim=True)
            nl = A_qk[:,:,-1]*v # B T N H
            A_qk = A_qk[:,:,:-1] # N H H
            
            for i in range(T):

                hidden_x = torch.einsum('bni, nji -> bnj',hidden_x, A_qk) + nl[:,i,:,:]

                y.append(hidden_x) 
            #print(A_window)
            
            y=torch.stack(y, dim=1).reshape(B, T, self.hidden_dim*self.window_dim) # B T N*H
            y=self.proj_out(y) # B T C
            #y = rearrange(y, 'b h l d -> b l (h d)')

        
        elif self.implementation == "relu_l1":
             #b h t c
            #hidden_x = torch.zeros(B, self.window_dim, C).to(hidden_states.device)
            hidden_x = torch.zeros(B, self.hidden_dim, self.window_dim).to(hidden_states.device)   # [B H]*N       
            y = []

            A_qk = torch.relu(self.A_w) # N H H+1
            A_qk = A_qk/torch.sum(A_qk,-1,keepdim=True)
            nl = A_qk[:,:,-1]*v # B T N H
            A_qk = A_qk[:,:,:-1] # N H H
            
            for i in range(T):

                hidden_x = torch.einsum('bni, nji -> bnj',hidden_x, A_qk) + nl[:,i,:,:]

                y.append(hidden_x) 
            #print(A_window)
            
            y=torch.stack(y, dim=1).reshape(B, T, self.hidden_dim*self.window_dim) # B T N*H
            y=self.proj_out(y) # B T C
            #y = rearrange(y, 'b h l d -> b l (h d)')


        elif self.implementation == "no_norm":
             #b h t c
            #hidden_x = torch.zeros(B, self.window_dim, C).to(hidden_states.device)
            hidden_x = torch.zeros(B, self.hidden_dim, self.window_dim).to(hidden_states.device)   # [B H]*N       
            y = []

            A_qk = self.A_w/(self.window_dim+1) #(self.window_dim+1) # N H H+1
            nl = A_qk[:,:,-1]*v # B T N H
            A_qk = A_qk[:,:,:-1] # N H H
            
            for i in range(T):

                hidden_x = torch.einsum('bni, nji -> bnj',hidden_x, A_qk) + nl[:,i,:,:]

                y.append(hidden_x) 
            #print(A_window)
            
            y=torch.stack(y, dim=1).reshape(B, T, self.hidden_dim*self.window_dim) # B T N*H
            y=self.proj_out(y) # B T C
            #y = rearrange(y, 'b h l d -> b l (h d)')

       
        else: 
            raise ValueError(f"Parallel implementation {self.implementation} not supported")

        return y.to(hidden_states.dtype)
