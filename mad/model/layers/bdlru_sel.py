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
    from mad.model.layers.ops.scans.hopscan import hopscan, hopscan_opt
except ImportError:
    hopscan, hopscan_opt = None, None

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
        elif self.implementation == "hopscan":

            #b t n*h
            A_qk = torch.softmax(gates,-1) # B T N H H+1
            nl = A_qk[:,:,:,:,-1]*v[:,:,:,:] # B T N H 
            A_qk = A_qk[:,:,:,:,:-1] # B T N H H

            # prepare for hopscan
            nl=nl.permute(0,2,3,1).reshape(B*self.hidden_dim,self.window_dim,T) # B*N H T
            A_qk = A_qk.permute(0,2,4,3,1).reshape(B*self.hidden_dim,self.window_dim,self.window_dim,T) # B*N H H' T

            y=hopscan(nl, A_qk) # B*N H T

            # reshape back 
            y=y.reshape(B,self.hidden_dim*self.window_dim,T).permute(0,2,1) # B T N*H

            y=self.proj_out(y)

        elif self.implementation == "hopscan_opt":
            
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

        elif self.implementation == "sigmoid_l1_hopscan_opt":

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

        elif self.implementation == "relu_l1_hopscan_opt":

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

        elif self.implementation == "nonorm_hopscan_opt":

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

        elif self.implementation == "tanh_l1_hopscan_opt":

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

        elif self.implementation == "lin_l1_hopscan_opt":

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
