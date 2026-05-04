import torch
# pscan
from torch.autograd.function import Function, FunctionCtx
from torch._higher_order_ops.associative_scan import associative_scan

### Eager Reference Implementations
def scan_ref_fwd(inputs:torch.Tensor, coeffs:torch.Tensor, reverse=False):
    outputs = torch.zeros_like(inputs)
    prev = torch.zeros_like(outputs[..., 0])

    for i in range(0, inputs.shape[-1])[::-1 if reverse else 1]:
        outputs[..., i] = torch.einsum('bij,bj->bi', coeffs[..., i], prev) + inputs[..., i]
        prev = outputs[..., i].clone()
        
    return outputs

def shift(input, shifts, fillval=0):
    # torch.roll without the copy of the wrap-around section
    if shifts > 0:
        output = torch.cat([torch.full_like(input[..., :shifts], fillval), input[..., :-shifts]], dim=-1)
    if shifts < 0:
        output = torch.cat([input[..., -shifts:], torch.full_like(input[..., shifts:], fillval)], dim=-1)
    return output

def scan_ref_bwd(d_outputs:torch.Tensor, coeffs:torch.Tensor, outputs:torch.Tensor, reverse=False):
    coeffs_bwd = shift(coeffs, -1 if not reverse else 1, fillval=0).permute(0,2,1,3)
    d_inputs = scan_ref_fwd(inputs=d_outputs, coeffs=coeffs_bwd, reverse=(not reverse))
    d_coeffs =  torch.einsum('bij,bkj->bikj',d_inputs,shift(outputs, shifts=1 if not reverse else -1, fillval=0))
    return d_inputs, d_coeffs


class ScanRefFn(Function):
    @staticmethod
    def forward(ctx:FunctionCtx, inputs:torch.Tensor, coeffs:torch.Tensor, reverse:bool=False) -> torch.Tensor:
        outputs = scan_ref_fwd(inputs=inputs, coeffs=coeffs, reverse=reverse)
        ctx.save_for_backward(coeffs, outputs)
        ctx.reverse = reverse
        return outputs
    
    @staticmethod
    def backward(ctx:FunctionCtx, d_outputs:torch.Tensor):
        coeffs, outputs = ctx.saved_tensors
        d_inputs, d_coeffs = scan_ref_bwd(d_outputs=d_outputs, coeffs=coeffs, outputs=outputs, reverse=ctx.reverse)
        return d_inputs, d_coeffs, None

#@torch.compile
def refpscan(inputs:torch.Tensor, coeffs:torch.Tensor):
    return ScanRefFn.apply(inputs, coeffs)

### Eager Higher-Order Op Implementations
def scan_hop_fwd(inputs:torch.Tensor, coeffs:torch.Tensor, reverse=False):

    def op(acc:dict, curr:dict):
        c = torch.einsum('bij,bjk->bik',curr['c'],acc['c'])
        x = curr['x'] + torch.einsum('bij,bj->bi',curr['c'],acc['x'])
        return dict(x=x, c=c)
    
    outputs = associative_scan(op, dict(x=inputs, c=coeffs), dim=-1, reverse=reverse, combine_mode='generic')['x']
    return outputs

def scan_hop_bwd(d_outputs:torch.Tensor, coeffs:torch.Tensor, outputs:torch.Tensor, reverse=False):
    coeffs_bwd = shift(coeffs, -1 if not reverse else 1, fillval=0).permute(0,2,1,3)
    d_inputs = scan_hop_fwd(inputs=d_outputs, coeffs=coeffs_bwd, reverse=(not reverse))
    d_coeffs = torch.einsum('bij,bkj->bikj',d_inputs, shift(outputs, shifts=1 if not reverse else -1, fillval=0))
    return d_inputs, d_coeffs

class ScanHopFn(Function):
    @staticmethod
    def forward(ctx:FunctionCtx, inputs:torch.Tensor, coeffs:torch.Tensor, reverse:bool=False) -> torch.Tensor:
        outputs = scan_hop_fwd(inputs=inputs, coeffs=coeffs, reverse=reverse)
        ctx.save_for_backward(coeffs, outputs)
        ctx.reverse = reverse
        return outputs
    
    @staticmethod
    def backward(ctx:FunctionCtx, d_outputs:torch.Tensor):
        coeffs, outputs = ctx.saved_tensors
        d_inputs, d_coeffs = scan_hop_bwd(d_outputs=d_outputs, coeffs=coeffs, outputs=outputs, reverse=ctx.reverse)
        return d_inputs, d_coeffs, None

def hopscan(inputs:torch.Tensor, coeffs:torch.Tensor):
    return ScanHopFn.apply(inputs, coeffs)


# hop ein opt
def shift_opt(input, shifts, fillval=0):
    # torch.roll without the copy of the wrap-around section
    if shifts > 0:
        output = torch.cat([torch.full_like(input[:, :shifts,...], fillval), input[:, :-shifts,...]], dim=1)
    if shifts < 0:
        output = torch.cat([input[:, -shifts:,...], torch.full_like(input[:, shifts:,...], fillval)], dim=1)
    return output

### Eager Higher-Order Op Implementations
def scan_hop_opt_fwd(inputs:torch.Tensor, coeffs:torch.Tensor, reverse=False):

    def op(acc:dict, curr:dict):
        c = torch.einsum('bcij,bcjk->bcik',curr['c'],acc['c'])
        x = curr['x'] + torch.einsum('bcij,bcj->bci',curr['c'],acc['x'])
        return dict(x=x, c=c)
    
    outputs = associative_scan(op, dict(x=inputs, c=coeffs), dim=1, reverse=reverse, combine_mode='generic')['x']
    return outputs

def scan_hop_opt_bwd(d_outputs:torch.Tensor, coeffs:torch.Tensor, outputs:torch.Tensor, reverse=False):
    coeffs_bwd = shift_opt(coeffs, -1 if not reverse else 1, fillval=0).permute(0,1,2,4,3)
    d_inputs = scan_hop_opt_fwd(inputs=d_outputs, coeffs=coeffs_bwd, reverse=(not reverse))
    d_coeffs = torch.einsum('btci,btck->btcik',d_inputs, shift_opt(outputs, shifts=1 if not reverse else -1, fillval=0))
    return d_inputs, d_coeffs


class ScanHopOptFn(Function):
    @staticmethod
    def forward(ctx:FunctionCtx, inputs:torch.Tensor, coeffs:torch.Tensor, reverse:bool=False) -> torch.Tensor:
        outputs = scan_hop_opt_fwd(inputs=inputs, coeffs=coeffs, reverse=reverse)
        ctx.save_for_backward(coeffs, outputs)
        ctx.reverse = reverse
        return outputs
    
    @staticmethod
    def backward(ctx:FunctionCtx, d_outputs:torch.Tensor):
        coeffs, outputs = ctx.saved_tensors
        d_inputs, d_coeffs = scan_hop_opt_bwd(d_outputs=d_outputs, coeffs=coeffs, outputs=outputs, reverse=ctx.reverse)
        return d_inputs, d_coeffs, None

#@torch.compile(mode="max-autotune", dynamic=False)
def hopscan_opt(inputs:torch.Tensor, coeffs:torch.Tensor):
    return ScanHopOptFn.apply(inputs, coeffs)