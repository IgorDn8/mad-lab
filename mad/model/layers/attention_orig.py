# adpated from:
# https://github.com/sustcsonglin/flash-linear-attention/blob/main/fla/layers/linear_attn.py
# https://github.com/HazyResearch/based/blob/main/based/models/mixers/linear_attention.py

import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange
from torch.autograd import Variable

try:
    from flash_attn import flash_attn_func
except ImportError:
    flash_attn_func = None

class AttentionOrig(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 16,
        expand_k: int = 1,
        expand_v: int = 1,
        eps: float = 1e-12,
        implementation: str="parallel",
        **kwargs
    ):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.key_dim = int(self.dim * expand_k)
        self.value_dim = int(self.dim * expand_v)
        self.head_qk_dim = self.key_dim // self.num_heads
        self.head_v_dim = self.value_dim // self.num_heads
        self.eps = eps
        self.implementation = implementation

        # initialize projections and feature map
        self.proj_q = nn.Linear(self.dim, self.key_dim, bias=False)
        self.proj_k = nn.Linear(self.dim, self.key_dim, bias=False)
        self.proj_v = nn.Linear(self.dim, self.value_dim , bias=False)
        self.out_proj = nn.Linear(self.value_dim, self.dim, bias=False)

    def forward(self, 
        hidden_states: torch.Tensor,
        *args, **kwargs
    ):
        """
        x (torch.Tensor): tensor of shape (b, d, l)
        y (torch.Tensor): tensor of shape (b, d, l)
        """
        # q is gate, k is non-linear proj, v is linear proj
        b, l, _ = hidden_states.size()
        casual_mask = torch.tril(torch.ones(l, l)).bool().to(hidden_states.device)

        q, k, v = self.proj_q(hidden_states), self.proj_k(hidden_states), self.proj_v(hidden_states)
        

        if self.implementation == "A_softmax":
            #b h t c
            v = v.view(b, l, self.num_heads, self.head_v_dim).transpose(1, 2)
            q = q.view(b, l, self.num_heads, self.head_qk_dim).transpose(1, 2)
            k = k.view(b, l, self.num_heads, self.head_qk_dim).transpose(1, 2) #b h t c

            A_qk = torch.einsum("bhnd,bhmd->bhnm", q, k)/math.sqrt(self.head_qk_dim)
            A_qk = torch.softmax(A_qk.masked_fill_(casual_mask == 0, float('-inf')), dim=-1)      
            y = torch.einsum("bhnm,bhme->bhne", A_qk, v)
            y = rearrange(y, 'b h l d -> b l (h d)')

        elif self.implementation == "A_lin":

            v = v.view(b, l, self.num_heads, self.head_v_dim).transpose(1, 2)
            q = q.view(b, l, self.num_heads, self.head_qk_dim).transpose(1, 2)
            k = k.view(b, l, self.num_heads, self.head_qk_dim).transpose(1, 2) #b h t c

            #b h t c
            A_qk = torch.einsum("bhnd,bhmd->bhnm", q, k) 
            A_qk = torch.tril(A_qk)   
            y = torch.einsum("bhnm,bhme->bhne", A_qk, v)
            z = 1 / (torch.einsum("bhld->bhl", A_qk) + self.eps)
            y = y * z[..., None]
            y = rearrange(y, 'b h l d -> b l (h d)')
            
            #flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, window_size=(-1, -1), alibi_slopes=None, deterministic=False):
        elif self.implementation == "A_flash":

            v = v.view(b, l, self.num_heads, self.head_v_dim)
            q = q.view(b, l, self.num_heads, self.head_qk_dim)
            k = k.view(b, l, self.num_heads, self.head_qk_dim)

            y=flash_attn_func(q, k, v, dropout_p=0.0, causal=True)

            y = y.view(b, l, self.num_heads*self.head_v_dim)

        else: 
            raise ValueError(f"Parallel implementation {self.implementation} not supported")

        return self.out_proj(y.to(q.dtype))


if __name__ == '__main__':
    import torch
    x = torch.randn(2, 128, 128).to(torch.bfloat16).cuda().requires_grad_(True)
    for fm in [
        'elu',
        'relu',
        'hedgehog',
        'taylor',
        't2r',
        'dpfp',
        'identity',
        'elementwise_product'
    ]:
        print(f'Testing rg-lru forward with {fm} feature map...')
        model = kernelRNN(dim=128, feature_map=fm).to(torch.bfloat16).cuda()
        y = model(x)
        print(y.shape)
        y.sum().backward()
        print(x.grad.shape)