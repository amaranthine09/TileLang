import tilelang
import tilelang.language as T 
from tilelang import jit 

@jit 
def copy_1D_serial_multithread(A):
    N =T.const("N")
    A: T.Tensor((N,), "float16")
    B = T.empty((N,), "float16")

    with T.Kernel(1, threads = 256) as _:
        T.copy(A, B)
    return B 