import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba2, Mamba


class MambaV2(nn.Module):
    """
    Mamba
    """

    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        version2: bool = True,
        dropout_rate: float = 0.0,
        **kwargs
    ):
        super().__init__()
        self.dim = dim

        if version2:
            self.layer = Mamba2(
                    d_model = dim,
                    d_state = d_state,
                    d_conv = d_conv,
                    expand = expand,
                )
        else:
            self.layer = Mamba(
                    d_model = dim,
                    d_state = d_state,
                    d_conv = d_conv,
                    expand = expand,
                )
        #self.drop = nn.Dropout(p=dropout_rate)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ):

        y = self.layer(hidden_states)
        #y = self.drop(y)

        return y

