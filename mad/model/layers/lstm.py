
import torch
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange

class LSTM(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        **kwargs
    ):
        super().__init__()
        
        self.dim = dim
        self.hidden_dim = hidden_dim

        #self.proj_v = nn.Linear(self.dim, self.dim)
        self.lstm = nn.LSTM(input_size=self.dim ,hidden_size=self.hidden_dim, batch_first=True)
        self.proj_out = nn.Linear(self.hidden_dim, self.dim, bias=False)

    
    def forward(self, 
        x: torch.Tensor,
        *args, **kwargs
    ):
        """
        x (torch.Tensor): tensor of shape (b, t, c)
        y (torch.Tensor): tensor of shape (b, t, c)
        """   
        B, T, C = x.size()
        hidden_h = torch.zeros(1,B, self.hidden_dim).to(x.device)
        hidden_c = torch.zeros(1,B, self.hidden_dim).to(x.device)

        y, _ = self.lstm(input=x, hx=(hidden_h, hidden_c))
        y = self.proj_out(y) # B T C       

        return y

if __name__ == '__main__':
    import torch
    x = torch.randn(2, 128, 128).to(torch.bfloat16).cuda().requires_grad_(True)

    model = LSTM(dim=12, hidden_dim =12).to(torch.bfloat16).cuda()
    y = model(x)
    print(y.shape)
    y.sum().backward()
    print(x.grad.shape)