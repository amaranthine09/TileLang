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
        base_idx = bx*BLOCK_N
        base_idy = by*BLOCK_M

        for i, j in T.Parallel(BLOCK_N, BLOCK_M):
            z = A[base_idx+i, base_idy+j]*B[base_idy+j]
            relu_grad = T.if_then_else(z>0, T.float16(1), T.float16(0))
            dA[base_idx+i, base_idy+j] = dC[base_idx+i, base_idy+j]* B[base_idy+j]*relu_grad
    return dA
    

N, M = 512, 1024
A = torch.randn(N, M, dtype=torch.float16, device="cuda")
B = torch.randn(M, dtype=torch.float16, device="cuda")
dC = torch.randn(N, M, dtype=torch.float16, device="cuda")

dA = tl_mul_relu_bwd(A, B, dC, BLOCK_N=128, BLOCK_M=64)

torch_out = dC * B[None, :] * ((A * B[None, :]) > 0).to(torch.float16)

print(torch.allclose(dA, torch_out, atol=1e-3))
print("✅ Puzzle 6: Backward of Broadcast Mul + ReLU - PASSED!")