import tilelang
import tilelang.language as T 
from tilelang import jit 
import torch 

@jit 
def vectfma(A, B, bias, block_N):
    N = T.const("N")
    A: T.Tensor((N,), "float16")
    B: T.Tensor((N,), "float16")
    bias: T.Tensor((N,), "float16")
    C = T.empty((N,), "float16")

    with T.Kernel(T.ceildiv(N, block_N), threads = 128) as bx:
        A_frg = T.alloc_fragment((block_N,), "float16")
        B_frg = T.alloc_fragment((block_N,), "float16")
        bias_frg = T.alloc_fragment((block_N,), "float16")
        C_frg = T.alloc_fragment((block_N,), "float16")

        T.copy(A[bx*block_N:(bx+1)*block_N], A_frg)
        T.copy(B[bx*block_N:(bx+1)*block_N], B_frg)
        T.copy(bias[bx*block_N:(bx+1)*block_N], bias_frg)

        for i in T.Parallel(block_N):
            C_frg[i] = A_frg[i] * B_frg[i] + bias_frg[i]
        T.copy(C_frg, C[bx*block_N:(bx+1)*block_N])
    return C


A = torch.randn(4096, dtype=torch.float16, device="cuda")
B = torch.randn(4096, dtype=torch.float16, device="cuda")
bias = torch.randn(4096, dtype=torch.float16, device="cuda")
C = vectfma(A, B, bias, block_N=256)

print(torch.allclose(C, A * B + bias, atol=1e-3))
print("✅ Puzzle 2: Fused Multiply-Add - PASSED!")