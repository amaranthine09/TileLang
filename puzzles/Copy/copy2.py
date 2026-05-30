import tilelang.language as T 
from tilelang import jit 

@jit 
def copy_serial_multithread(A):
    N: T.const("N")
    A: T.Tensor((N,), T.float16)
    B = T.empty((N,), T.float16)

    with T.kernel(1, threads = 256) as _:
        T.copy(A, B)
    return B 