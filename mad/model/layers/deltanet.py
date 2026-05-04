import torch
import torch.nn as nn
import torch.nn.functional as F
# from fla.layers import (
#     DeltaNet,
#     GatedDeltaNet,
# )

try:
    from fla.layers import (
    DeltaNet,
    GatedDeltaNet,
)
except ImportError:
    print(f"fla is not installed! ")
    DeltaNet, GatedDeltaNet = None, None

class dnet(nn.Module):
    """
    DeltaNet
    """

    def __init__(
        self,
        dim: int,
        head_dim: int,
        num_heads: int,
        expand_v: int = 1, 
        gated: bool = True,
        negative: bool = True,
        mode: str = 'chunk',
        dropout_rate: float = 0.0,
        **kwargs
    ):
        super().__init__()
        self.dim = dim

        if gated:
            # key_dim = head_dim*num_heads
            self.layer = GatedDeltaNet(
                    hidden_size = dim,
                    head_dim = head_dim,
                    num_heads = num_heads,
                    expand_v = expand_v,
                    allow_neg_eigval = negative,
                    mode=mode,
                )
        else:
            # key_dim = hidden*v_expand
            self.layer = DeltaNet(
                    hidden_size = dim,
                    num_heads = num_heads,
                    expand_k = expand_v,
                    expand_v = expand_v,
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

