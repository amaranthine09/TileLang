import tilelang 
import tilelang.language as T 
from tilelang import jit 
import torch 

@jit 
def vectadd(A, B, block_N: int):
    N = T.const("N")
    A: T.Tensor((N,), "float16")
    B: T.tensor((N,), "float16")
    C = T.empty((N,), "float16")

    with T.Kernel(T.ceildiv(N, block_N), threads = 128) as bx:
        base_idx = bx*block_N
        for i in T.Parallel(block_N):
            C[base_idx+i] = A[base_idx+i]+ B[base_idx+i]
        return C
A = torch.rand(1024, dtype=torch.float16, device="cuda")
B = torch.rand(1024, dtype=torch.float16, device="cuda")
C = vectadd(A, B, block_N= 128)
print(torch.allclose(C, A + B))
