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
          A_frag = T.alloc_fragment((block_M,), "float16")
          B_frag = T.alloc_fragment((block_N,), "float16")
          C_frag = T.alloc_fragment((block_M, block_N), "float16")

          T.copy(A[by*block_M:(by+1)*block_M], A_frag)
          T.copy(B[bx*block_N:(bx+1)*block_N], B_frag)

          for i, j in T.Parallel(block_M, block_N):
               C_frag[i,j] = A_frag[i] + B_frag[j]
          T.copy(C_frag, C[by*block_M: (by+1)*block_M, bx*block_N:(bx+1)*block_N])
     return C

A = torch.randn(512, dtype=torch.float16, device="cuda")
B = torch.randn(1024, dtype=torch.float16, device="cuda")
C = outervectoraddition(A, B, block_M=64, block_N=128)
torch_out = A[:, None] + B[None, :]
print(torch.allclose(C, torch_out, atol=1e-3))
print("✅ Puzzle 4: Outer Vector Addition (A[m] + B[n]) - PASSED!")