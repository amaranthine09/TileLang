import tilelang
import tilelang.language as T
from tilelang import jit

@jit  
def add(N: int, block: int = 256, dtype: str = 'float32'):

    @T.prim_func
    def add_kernel(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
        C: T.Tensor((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block), threads=block) as bx:
            for i in T.Parallel(block):
                gi = bx * block + i
                C[gi] = A[gi] + B[gi]

    return add_kernel

import torch
N = 1 << 20
A = torch.randn(N, device='mps', dtype=torch.float32)
B = torch.randn(N, device='mps', dtype=torch.float32)
C = torch.empty(N, device='mps', dtype=torch.float32)

kernel = add(N)
kernel(A, B, C)  
torch.testing.assert_close(C, A + B)

print("✅ Kernel executed successfully on Metal (MPS)!")
print("-" * 50)
print(f"Tensor A (first 5): {A[:5].cpu().numpy()}")
print(f"Tensor B (first 5): {B[:5].cpu().numpy()}")
print(f"Tensor C (Result):  {C[:5].cpu().numpy()}")
print("-" * 50)
print("Vector addition verification passed!")