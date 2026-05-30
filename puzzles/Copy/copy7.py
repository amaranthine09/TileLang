import tilelang 
import tilelang.language as T 
from tilelang import jit

@jit 
def copy2d_to_shared(A, block_M, block_N):
    M, N = T.const("M, N")
    A: T.Tensor((M, N), "float16")

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads= 128) as (bx, by):
        A_shared = T.alloc_shared( block_M, block_N)
        T.copy(A[by*block_M:(by+1)*block_M, bx*block_N:(bx+1)*block_N], A_shared)
    return A_shared