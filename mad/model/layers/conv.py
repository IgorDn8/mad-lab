# adpated from:
# https://github.com/sustcsonglin/flash-linear-attention/blob/main/fla/layers/linear_attn.py
# https://github.com/HazyResearch/based/blob/main/based/models/mixers/linear_attention.py

import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange

class Conv(nn.Module):
    def __init__(
        self,
        dim: int,
        kernel_size: int = 1,
        implementation: str="orig",
        **kwargs
    ):
        super().__init__()
        
        self.dim = dim
        self.implementation = implementation
        self.kernel_size= kernel_size

        self.proj_v = nn.Linear(self.dim, self.dim)
        self.conv1d = nn.Conv1d(self.dim,self.dim,kernel_size)

    
    def forward(self, 
        hidden_states: torch.Tensor,
        *args, **kwargs
    ):
        """
        x (torch.Tensor): tensor of shape (b, t, c)
        y (torch.Tensor): tensor of shape (b, t, c)
        """
        B, T, C = hidden_states.size()  
        hidden_states = F.pad(hidden_states,(0,0,self.kernel_size-1,0))       
        v = self.proj_v(hidden_states).permute(0,2,1) # B C T
        
        if self.implementation == "orig":
            #b t c
            y = self.conv1d(v).permute(0,2,1) # B T C          
        else: 
            raise ValueError(f"Parallel implementation {self.implementation} not supported")

        return y.to(hidden_states.dtype)

if __name__ == '__main__':
    import torch
    x = torch.randn(2, 128, 128).to(torch.bfloat16).cuda().requires_grad_(True)

    model = Conv(dim=12).to(torch.bfloat16).cuda()
    y = model(x)
    print(y.shape)
    y.sum().backward()
    print(x.grad.shape)