import tilelang
import tilelang.language as T 
import torch 
from tilelang import jit 

@jit 
def vectaddwithrelu(A, B, block_N):
    N = T.const("N")
    A: T.Tensor((N,), "float16")
    B: T.Tensor((N,), "float16")
    C = T.empty((N,), "float16")

    with T.Kernel(T.ceildiv(N, block_N), threads = 128) as bx:
        base_idx = bx*block_N
        for i in T.Parallel(block_N):
            temp = A[base_idx+i]* B[base_idx+i]
            C[base_idx+i] = T.maximum(temp, T.float16(0))
    return C

A = torch.randn(1024, dtype = torch.float16, device ="cuda")
B = torch.randn(1024, dtype = torch.float16, device ="cuda")
C = vectaddwithrelu(A, B, block_N= 128)

torch_out = torch.maximum(A * B, torch.tensor(0., dtype=torch.float16, device="cuda"))
print(torch.allclose(C, torch_out, atol=1e-3))
print("✅ Correct!")
    