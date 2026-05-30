import tilelang
import tilelang.language as T 
from tilelang import jit 

@jit 
def copy2d_to_shared(A, block_M, block_N):
    M, N = T.const("M, N")
    A: T.Tensor((M, N), "float16")
    B = T.empty((M, N), "float16")

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads= 128) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_N), "float16")
        T.copy(A[by*block_M:(by+1)*block_M, bx*block_N:(bx+1)*block_N], A_shared)
        T.copy(A_shared, B[by*block_M:(by+1)*block_M, bx*block_N:(bx+1)*block_N])
    return B

import torch
A = torch.rand(1024, 2048, dtype=torch.float16, device="cuda")
B = copy2d_to_shared(A, block_M=128, block_N=64)
print(torch.allclose(A, B))