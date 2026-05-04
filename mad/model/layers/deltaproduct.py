import torch
import torch.nn as nn
import torch.nn.functional as F
# from fla.layers import GatedDeltaProduct

try:
    from fla.layers import GatedDeltaProduct  # pscan implementation
except ImportError:
    print(f"fla is not installed! ")
    GatedDeltaProduct = None

class dproduct(nn.Module):
    """
    DeltaProduct
    """

    def __init__(
        self,
        dim: int,
        head_dim: int,
        num_heads: int,
        expand_v: int = 1, 
        gated: bool = True,
        negative: bool = True,
        rank: int = 1,
        mode: str = 'chunk',
        dropout_rate: float = 0.0,
        **kwargs
    ):
        super().__init__()
        self.dim = dim


        self.layer = GatedDeltaProduct(
                hidden_size = dim,
                head_dim = head_dim,
                num_heads = num_heads,
                expand_v = expand_v,
                num_householder = rank,
                use_forget_gate = gated,
                allow_neg_eigval = negative,
                mode=mode,
            )
        #self.drop = nn.Dropout(p=dropout_rate)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ):

        y = self.layer(hidden_states)[0]
        #y = self.drop(y)

        return y

