import tilelang
import tilelang as T 
from tilelang import jit 

@jit 
def copy2d_serial_multithread(A, block_M: int, block_N: int):
    M, N = T.const("M, N")
    A: T.Tensor((M, N), "float16")
    B = T.empty((M, N), "float16")

    with  T.Kernel( T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads = 128) as (bx, by):
        T.copy(
            A[by*block_M: (by+1)*block_M, bx*block_N: (bx+1)*block_N],
            B[by*block_M: (by+1)*block_M, bx*block_N: (bx+1)*block_N]
            )
    return B