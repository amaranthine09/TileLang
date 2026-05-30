import tilelang.language as T 
from tilelang import jit 

@jit 
def copy_1d_serial(A):
    N: T.const("N") 
    A: T.Tensor((N,), T.float16)
    B = T.empty((N,), T.float16)

    with T.kernel(1, threads= 1,) as _:
        T.copy(A, B)
    return B 



