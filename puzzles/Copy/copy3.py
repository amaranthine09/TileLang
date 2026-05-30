import tilelang
import tilelang.language as T 
from tilelang import jit 

@jit 
def copy_1D_parallel(A, block_N: int):
    N =T.const("N")
    A: T.Tensor((N,), "float16")
    B = T.empty((N,), "float16")

    with T.Kernel(T.ceildiv(N, block_N), threads = 256) as bx:
        T.copy(
            A[bx*block_N:(bx+1)*block_N],
            B[bx*block_N:(bx+1)*block_N]
            )
    return B

import torch
A = torch.rand(4096, dtype=torch.float16, device="cuda")
B = copy_1D_parallel(A, block_N=256)
print(torch.allclose(A, B))