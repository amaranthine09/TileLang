import tilelang
import tilelang.language as T 
from tilelang import jit 

@jit 
def copy2d_serial_multithread(A):
    M, N = T.const("M, N")
    A: T.Tensor((M, N), "float16")
    B = T.empty((M, N), "float16")

    with T.Kernel(1, threads = 128) as _:
        T.copy(A, B)
    return B

import torch
A = torch.rand(1024, 2048, dtype=torch.float16, device="cuda")
B = copy2d_serial_multithread(A)
print(torch.allclose(A, B))