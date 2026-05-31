import tilelang
import tilelang.language as T 
import torch 
from tilelang import jit 
@jit 
def tl_mul_relu_bwd(A, B, dC, BLOCK_N: int, BLOCK_M: int):
    N, M = T.const("N, M")
    dtype = T.float16
    A: T.Tensor((N, M), dtype)
    B: T.Tensor((M,), dtype)
    dC: T.Tensor((N, M), dtype)
    dA = T.empty((N, M), dtype)

    with T.Kernel(T.ceildiv(N,BLOCK_N), T.ceildiv(M, BLOCK_M), threads = 128) as (bx,by):

        A_shared = T.alloc_fragment((BLOCK_N, BLOCK_M), "float16")
        B_shared = T.alloc_fragment((BLOCK_M,), "float16")
        dC_shared = T.alloc_fragment((BLOCK_N, BLOCK_M), "float16")

        dA_local = T.alloc_fragment((BLOCK_N, BLOCK_M), "float16")
        
        T.copy(A[bx*BLOCK_N, BLOCK_M], A_shared)
        T.copy(B[by*BLOCK_M], B_shared)
        T.copy(dC[bx*BLOCK_N, BLOCK_M], dC_shared)

        for i, j in T.Parallel(BLOCK_N, BLOCK_M):
            z = A_shared[i,j]*B_shared[j]
            relu_grad = T.if_then_else(z>0, T.float16(1), T.float16(0))
            dA_local[i, j] = dC_shared[i, j]* B_shared[j]*relu_grad
        T.copy( dA_local, dA[bx*BLOCK_N, by*BLOCK_M])
    return dA

N, M = 512, 1024
A = torch.randn(N, M, dtype=torch.float16, device="cuda")
B = torch.randn(M, dtype=torch.float16, device="cuda")
dC = torch.randn(N, M, dtype=torch.float16, device="cuda")

dA = tl_mul_relu_bwd(A, B, dC, BLOCK_N=128, BLOCK_M=64)
torch_out = dC * B[None, :] * ((A * B[None, :]) > 0).to(torch.float16)

print(torch.allclose(dA, torch_out, atol=1e-3))
print("✅ Puzzle 6: Backward of Broadcast Mul + ReLU - PASSED!")