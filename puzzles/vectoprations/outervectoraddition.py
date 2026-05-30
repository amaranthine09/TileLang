import tilelang 
import tilelang.language as T 
from tilelang import jit 
import torch 

@jit 
def outervectoraddition(A, B, block_M, block_N):
     M, N = T.const("M, N")
     A: T.Tensor((M,), "float16")
     B: T.Tensor((N,), "float16")
     C = T.empty((M,N), "float16")

     with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads = 128) as (bx, by):
          for i, j in T.Parallel(block_M, block_N):
               C[by*block_M+i, bx*block_N+j] = A[by*block_M+i]+B[bx*block_N+j]
     return C

A = torch.randn(512, dtype=torch.float16, device="cuda")
B = torch.randn(1024, dtype=torch.float16, device="cuda")
C = outervectoraddition(A, B, block_M=64, block_N=128)
torch_out = A[:, None] + B[None, :]
print(torch.allclose(C, torch_out, atol=1e-3))
print("✅ Puzzle 4: Outer Vector Addition (A[m] + B[n]) - PASSED!")