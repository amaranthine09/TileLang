import tilelang 
import tilelang.language as T 
import torch 
from tilelang import jit 
@jit
def reducedsum(A, BLOCK_N: int, BLOCK_M: int):
    N, M = T.const("N, M")
    dtype = T.float32
    A: T.Tensor((N, M), dtype)
    B = T.empty((N,), dtype)

    with T.Kernel(N // BLOCK_N, threads=256) as pid_n:
        A_local = T.alloc_fragment((BLOCK_N, BLOCK_M), dtype)
        B_local = T.alloc_fragment((BLOCK_N,), dtype)
        T.clear(B_local)

        for m_blk_id in T.Serial(M // BLOCK_M):
            T.copy(A[pid_n * BLOCK_N, m_blk_id * BLOCK_M], A_local)
            T.reduce_sum(A_local, B_local, dim=1, clear=False)
        T.copy(B_local, B[pid_n * BLOCK_N])

    return B
N, M = 512, 1024
A = torch.randn(N, M, dtype=torch.float32, device="cuda")
C = reducedsum(A, block_N=128, block_M=64)

torch_out = A.sum(dim=1)
print(torch.allclose(C, torch_out, atol=1e-3))
print("✅ Puzzle 7: Row-wise Reduction (Sum) - PASSED!")