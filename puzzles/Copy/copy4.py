import tilelang
import tilelang as T 
from tilelang import jit 

@jit 
def copy2d_serial_singlethread(A):
    M, N = T.const("M, N")
    A: T.Tensor((M, N), "float16")
    B = T.empty((M, N), "float16")

    with  T.Kernel(1, threads = 1) as _:
        T.copy(A, B)
    return B