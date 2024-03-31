import torch
import triton
from distance_jit import _bundle_distance_fwd, _chamfer_distance_fwd

class ChamferDistance(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x:torch.Tensor, y:torch.Tensor, squared:bool):
        assert x.is_cuda and y.is_cuda
        assert x.dtype == y.dtype
        if not (x.is_contiguous() and y.is_contiguous()):
            x = x.contiguous()
            y = y.contiguous()
        B, N, ndim = x.shape
        B, M, ndim = y.shape
        assert ndim == 3
        output = torch.zeros((B,N), dtype=x.dtype,device=x.device)
        indices = torch.zeros((B,N),device=x.device).long()
        grid = lambda meta: (B,triton.cdiv(N, meta['X_BLOCK_SIZE']),)
        
        with torch.cuda.device(x.device):
            _chamfer_distance_fwd[grid](x, y, output, indices, N=N,M=M, X_BLOCK_SIZE=8, Y_BLOCK_SIZE=256)
        assert indices.max() < M
        ctx.save_for_backward(x, y, indices)
        ctx.squared = squared

        if squared:
            return output
        else:
            return torch.sqrt(output)

    @staticmethod
    def backward(ctx, grad_output):
        x, y, indices = ctx.saved_tensors
        grad_y = torch.zeros_like(y)
        
        y = torch.gather(y,1,indices[...,None].expand(-1,-1,y.shape[2]))
        if ctx.squared:
            grad_x = (x-y)*grad_output[...,None]
            grad_y[:,indices,:] = -grad_x
        else:
            grad_x = (x-y)*(grad_output[...,None]/torch.norm(x-y,dim=-1,keepdim=True))
            grad_y[:,indices,:] = -grad_x
        return grad_x, grad_y, None
    
def chamfer_distance(x: torch.Tensor, y: torch.Tensor, squared: bool = False):
    """
    Calculates the chamfer distance between two point clouds.

    Args:
        x (torch.Tensor): The first point cloud with the shape (B, N, ndim).
        y (torch.Tensor): The second point cloudwith the shape (B, M, ndim).
        squared (bool, optional): If True, returns the squared chamfer distance. 
            If False, returns the non-squared chamfer distance. Defaults to False.

    Returns:
        torch.Tensor: The chamfer distance between the two point clouds.
    """
    return ChamferDistance.apply(x, y, squared)

class BundleDistance(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x:torch.Tensor, y:torch.Tensor, squared:bool):
        assert x.is_cuda and y.is_cuda
        assert x.dtype == y.dtype
        if not (x.is_contiguous() and y.is_contiguous()):
            x = x.contiguous()
            y = y.contiguous()
        B, N, P, ndim = x.shape
        B, M, P, ndim = y.shape
        assert ndim == 3
        output = torch.zeros((B,N), dtype=x.dtype,device=x.device)
        indices = torch.zeros((B,N),device=x.device).long()
        flip = torch.zeros((B,N),dtype=bool,device=x.device)
        grid = lambda meta: (B,triton.cdiv(N, meta['X_BLOCK_SIZE']),)
        
        with torch.cuda.device(x.device):
            _bundle_distance_fwd[grid](x, y, output, indices,flip,
                                       N, M, P,
                                       X_BLOCK_SIZE=8, Y_BLOCK_SIZE=128,P_next_power_of_2=triton.next_power_of_2(P))
        assert indices.max() < M
        ctx.save_for_backward(x, y, indices,flip)
        ctx.squared = squared

        if squared:
            return output
        else:
            return torch.sqrt(output)

    @staticmethod
    def backward(ctx, grad_output):
        x, y, indices,flip = ctx.saved_tensors
        B, N, P, ndim = x.shape
        grad_y = torch.zeros_like(y)
        y = torch.gather(y,1,indices[...,None,None].expand(-1,-1,y.shape[2],y.shape[3]))
        y = torch.where(flip[...,None,None],y.flip(2),y)
        if ctx.squared:
            grad_x = (x-y)*grad_output[...,None,None]
            grad_y[:,indices] = torch.where(flip[...,None,None], -grad_x.flip(2), -grad_x)
        else:
            grad_x = (x-y)*grad_output[...,None,None]/torch.norm(x-y,dim=-1,keepdim=True)/P
            grad_y[:,indices] = torch.where(flip[...,None,None], -grad_x.flip(2), -grad_x)
        return grad_x, grad_y, None
def bundle_distance(x: torch.Tensor, y: torch.Tensor, squared: bool = False):
    """
    Minimum average Direct-Flip (MDF) streamline distances (Garyfallidis et al., 2012).

    Args:
        x (torch.Tensor): The first bundle with the shape (B, N, P, ndim).
        y (torch.Tensor): The second bundle with the shape (B, M, P, ndim).
        squared (bool, optional): If True, returns the squared bundle distance. 
            If False, returns the non-squared bundle distance. Defaults to False.

    Returns:
        torch.Tensor: The bundle distance between the two bundles.
    """
    return BundleDistance.apply(x, y, squared)