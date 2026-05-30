import tilelang.language as T 
from tilelang import jit 

@jit 
def copy_1D_parallel(A, block_N: int):
    N: T.const("N")
    A: T.Tensor((N,), T.float16)
    B: T.empty((N,), T.float16)

    with T.kernel(T.ceildiv(N, block_N), threads = 256) as bx:
        T.copy(
            A[bx*block_N:(bx+1)*block_N],
            B[bx*block_N:(bx+1)*block_N]
            )
    return B
    