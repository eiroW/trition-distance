import torch
import triton
import triton.language as tl

@triton.jit
def _chamfer_distance_fwd(
    X_pointer, Y_pointer, 
    output_pointer, index_pointer,
    N:tl.constexpr, M:tl.constexpr,
    X_BLOCK_SIZE:tl.constexpr=128,
    Y_BLOCK_SIZE:tl.constexpr=128,
    
    ndim:tl.constexpr=3,
):
    pid = tl.program_id(1)
    batch = tl.program_id(0)
    offset = batch * N + pid * X_BLOCK_SIZE + tl.arange(0,X_BLOCK_SIZE) % N
    offx = offset[:,None] * ndim + tl.arange(0,4)[None,:] % ndim
    # print("offx",offx)
    dmask = tl.arange(0,4) < ndim
    num_mask = (pid * X_BLOCK_SIZE + tl.arange(0,X_BLOCK_SIZE)) < N 

    X = tl.load(X_pointer + offx,mask = num_mask[:,None] & dmask[None,:], other=0.0)
    # print("X",X)
    results_min = tl.full((X_BLOCK_SIZE,), 1e9,  dtype=tl.float32)
    indices_min = tl.zeros((X_BLOCK_SIZE,), dtype=tl.int64)
    offy = (batch * M + tl.arange(0,Y_BLOCK_SIZE)[:,None])* ndim + tl.arange(0,4)[None,:] % ndim
    for i in range(tl.cdiv(M,Y_BLOCK_SIZE)):
        
        mask = (tl.arange(0,Y_BLOCK_SIZE) + i*Y_BLOCK_SIZE) < M
        Y = tl.load(Y_pointer+offy ,
                     mask = dmask[None,:]&mask[:,None],other=0.0)
        diff = X[:,None,:]-Y[None,:,:]
        result = tl.sum(diff*diff,axis = 2)

        result = tl.where(mask[None,:],result,1e9)
        result,indices = tl.min(result, axis=1,return_indices=True)
        mask = result<results_min
        indices_min = tl.where(mask, indices + i * Y_BLOCK_SIZE, indices_min,)
        results_min = tl.where(mask, result, results_min,)
        offy += Y_BLOCK_SIZE * ndim
        
    tl.store(output_pointer + offset , results_min.to(X.dtype),mask=num_mask)
    tl.store(index_pointer + offset , indices_min,mask=num_mask)

@triton.jit
def _bundle_distance_fwd(
    X_pointer, Y_pointer, 
    output_pointer,index_pointer,flip_pointer,
    N:tl.constexpr,M:tl.constexpr, P:tl.constexpr=16,
    X_BLOCK_SIZE:tl.constexpr=128,
    Y_BLOCK_SIZE:tl.constexpr=128,
    P_next_power_of_2:tl.constexpr=16,
    
    ndim:tl.constexpr=3,
):
    pid = tl.program_id(1)
    batch = tl.program_id(0)
    # P_next_power_of_2 = 1 << (P-1).bit_length()
    offset = batch * N + pid * X_BLOCK_SIZE + tl.arange(0,X_BLOCK_SIZE) % N
    offx = offset[:,None,None] * ndim * P + tl.arange(0, P_next_power_of_2)[None,:,None] % P * ndim +tl.arange(0,4)[None,None,:] % ndim
    
    dmask = tl.arange(0,4) < ndim
    num_mask = (pid * X_BLOCK_SIZE + tl.arange(0,X_BLOCK_SIZE)) < N 
    pmask = tl.arange(0, P_next_power_of_2) < P
    
    X = tl.load(X_pointer + offx,mask = num_mask[:,None,None] & dmask[None,None,:] & pmask[None,:, None], other=0.0)

    results_min = tl.full((X_BLOCK_SIZE,), 1e9,  dtype=tl.float32)
    indices_min = tl.zeros((X_BLOCK_SIZE,), dtype=tl.int64)
    flip_min = tl.zeros((X_BLOCK_SIZE,), dtype=tl.int1)
    
    for i in range(tl.cdiv(M,Y_BLOCK_SIZE)):
        mask = (tl.arange(0,Y_BLOCK_SIZE) + i*Y_BLOCK_SIZE) < M
        
        # calculate the direct distance between X and Y
        offy = batch * M  * P * ndim\
            + (i*Y_BLOCK_SIZE + tl.arange(0,Y_BLOCK_SIZE)[:,None,None] % M ) * P * ndim\
            + (tl.arange(0, P_next_power_of_2)[None,:,None] % P)  * ndim\
            + tl.arange(0,4)[None,None,:] % ndim
        # print('offy',offy)  
        Y = tl.load(Y_pointer+offy ,
                     mask = dmask[None,None,:] & pmask[None,:,None] & mask[:,None,None],other=0.0)
        
        diff = X[:,None,:,:] - Y[None,:,:,:]
        result_direct = tl.sum(diff*diff,axis = 3)
        result_direct = tl.sum(result_direct,axis = 2)
        
        # calculate the flip distance between X and Y
        offy = batch * M  * P * ndim\
            + (i*Y_BLOCK_SIZE + tl.arange(0,Y_BLOCK_SIZE)[:,None,None] % M ) * P * ndim\
            + (P - tl.arange(0, P_next_power_of_2)[None,:,None] % P - 1) * ndim\
            + tl.arange(0,4)[None,None,:] % ndim
        Y = tl.load(Y_pointer+offy,
                     mask = dmask[None,None,:] & pmask[None,:,None] & mask[:,None,None],other=0.0)
        
        diff = X[:,None,:,:] - Y[None,:,:,:]
        result_flip =  tl.sum(diff*diff,axis = 3)
        
        result_flip = tl.sum(result_flip,axis = 2)
        

        
        flip = result_direct > result_flip
        result = tl.where(flip,result_flip,result_direct)
        
        result = tl.where(mask[None,:],result,1e9)
        result,indices = tl.min(result, axis=1,return_indices=True)
        
        flip = _get_values_on_indices(flip, indices, length=Y_BLOCK_SIZE)
        
        mask = result<results_min
        indices_min = tl.where(mask, indices + i * Y_BLOCK_SIZE, indices_min,)
        results_min = tl.where(mask, result, results_min,)
        flip_min = tl.where(mask, flip, flip_min,)
        
        
    tl.store(output_pointer + offset , results_min.to(X.dtype),mask=num_mask)
    tl.store(index_pointer + offset , indices_min,mask=num_mask)
    tl.store(flip_pointer + offset , flip_min,mask=num_mask)
@triton.jit
def _get_values_on_indices(
    X, indices,length
):
    
    mask = tl.arange(0, length)[None,:] == indices[:,None]
    return tl.sum(mask & X,axis=1)

if __name__ == "__main__":
    N = 16
    M = 16
    B = 1
    P = 3
    
    device = torch.device("cuda")
    x = (torch.randn(B, N, 3)).to(device)
    y = (torch.randn(B, M, 3)).to(device)
    output = torch.zeros((B,N),dtype=x.dtype,device=x.device)
    indices = torch.zeros((B,N),dtype=torch.int64,device=x.device)
    flip = torch.zeros((B,N),dtype=bool,device=x.device)
    grid = lambda meta: (B,triton.cdiv(N, meta['X_BLOCK_SIZE']),)
    _chamfer_distance_fwd[grid](x, y, output, indices, N, M, X_BLOCK_SIZE=2, Y_BLOCK_SIZE=2,)
    # assert torch.allclose(((x-y[:,indices,:])**2).sum(-1),output)
