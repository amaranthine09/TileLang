import tilelang
import tilelang.language as T 
import torch 
from tilelang import jit 

@jit
def elementwiseaddition(
    M:int,
    N:int,
    block_M: int = 128,
    block_N: int = 128,
    dtyep: str = "float18",
    acc_dtype: str = "float32"
):
    @T.prim_func
    def addkernel(
        A: T.Tensor((M,N), dtype),
        B: T.Tensor((M,N), dtype),
        C: T.Tensor((M,N), acc_dtype)
    ):
        with T.kernel( T.ceildiv(N, block_N),T.ceildive(M, block_M), threads = 128) as (bx, by):
            bx_s = bx*block_N 
            by_s = by*block_M

            for i, j in T.Parallel(block_M, block_N):
                C[by_s+i, bx+j] = A[by_s+i, bx+j] + B[by_s+i, bx_s+j]
    return addkernel
