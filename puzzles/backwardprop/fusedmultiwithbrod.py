import tilelang 
import tilelang.language as T 
import torch 
from tilelang import jit

@jit
def tl_mul_relu_bcast(A, B, block_N: int, block_M: int):
    N, M = T.const("N, M")
    dtype = T.float16
    A: T.Tensor((N, M), dtype)
    B: T.Tensor((M,), dtype)
    C = T.empty((N, M), dtype)
    
    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads = 256) as (bx, by):
        A_shared = T.alloc_shared((block_N, block_M), "float16")
        B_shared = T.alloc_shared((block_M,), "float16")
        C_local = T.alloc_fragment((block_N, block_M), "float32")

        T.clear(C_local)

        T.copy(A[bx*block_N: (bx+1)*block_N, by*block_M:(by+1)*block_M], A_shared)
        T.copy(B[by*block_M: (by+1)*block_M], B_shared)

        for i, j in T.Parallel(block_N, block_M):
            temp = A_shared[i, j]*B_shared[j]
            C_local[i,j] = T.maximum(temp, T.float32(0))
        T.copy(C_local, C[bx*block_N:(bx+1)*block_N, by*block_M:(by+1)*block_M])
    return C

A = torch.randn(512, 1024, dtype=torch.float16, device="cuda")
B = torch.randn(1024, dtype=torch.float16, device="cuda")
C = tl_mul_relu_bcast(A, B, block_N=128, block_M=64)

torch_out = torch.maximum(A * B[None, :], torch.tensor(0., dtype=torch.float16, device="cuda"))
print(torch.allclose(C, torch_out, atol=1e-3))
print("✅ Puzzle 5: Broadcast Mul + ReLU with Shared Memory - PASSED!")