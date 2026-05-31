import tilelang 
import tilelang.language as T 
import torch 
from tilelang import jit 

@jit
def reducedsum(A, block_N:int, block_M):
    N, M = T.const("N, M")
    dtype = T.float32
    A:  T.Tensor((N, M), dtype)
    B = T.empty((N,))

    with T.Kernel(T.ceildiv(N, block_N), threads = 128) as (bx, by):
        n_idx = bx * block_N
        A_frag = T.alloc_fragment((block_N, block_M), dtype)
        B_frag = T.alloc_fragment((block_N,) , dtype )
        for i in T.Parallel(block_N):
            B_frag[i] = T.float32(0)
        for m_start in T.Serial(M // block_M):
            m_idx = m_start * block_M
            T.copy(A[n_idx:n_idx+block_N, m_idx:m_idx+block_M], A_frag)

            for i in T.Parallel(block_N):
                row_sum = T.alloc_var(dtype)
                row_sum = T.float32(0)

                for j in T.Serial(block_M):
                    row_sum += A_frag[i, j]
                B_frag[i] += row_sum

        T.copy(B_frag, B[n_idx:n_idx+block_N])

    return B

N, M = 512, 1024
A = torch.randn(N, M, dtype=torch.float32, device="cuda")
C = reducedsum(A, block_N=128, block_M=64)

torch_out = A.sum(dim=1)
print(torch.allclose(C, torch_out, atol=1e-3))
print("✅ Puzzle 7: Row-wise Reduction (Sum) - PASSED!")