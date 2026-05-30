import tilelang
import tilelang.language as T 
from tilelang import jit 
import torch 

@jit 
def vectaddwithrelu(A, B, block_N):
    N = T.const("N")
    A: T.Tensor((N,), "float16")
    B: T.Tensor((N,), "float16")
    C = T.empty((N,), "float16")

    with T.Kernel(T.ceildiv(N, block_N), threads = 128) as bx:
        A_frg = T.alloc_fragment((block_N,), "float16")
        B_frg = T.alloc_fragment((block_N,), "float16")
        C_frg = T.alloc_fragment((block_N,), "float16")

        T.copy(A[bx*block_N:(bx+1)*block_N], A_frg)
        T.copy(B[bx*block_N:(bx+1)*block_N], B_frg)

        for i in T.Parallel(block_N):
            temp = A_frg[i] * B_frg[i]
            C_frg[i] = T.if_then_else(temp > 0, temp, T.float16(0))
        T.copy(C_frg, C[bx*block_N:(bx+1)*block_N])
    return C


A = torch.randn(1024, dtype=torch.float16, device="cuda")
B = torch.randn(1024, dtype=torch.float16, device="cuda")
C = vectaddwithrelu(A, B, block_N=128)

torch_out = torch.maximum(A * B, torch.tensor(0., dtype=torch.float16, device="cuda"))
print(torch.allclose(C, torch_out, atol=1e-3))
print("✅ Correct!")