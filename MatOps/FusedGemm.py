import tilelang
import tilelang.language as T
from tilelang import jit
import torch
import math

@jit
def fused_matmul_bias_activation(
    M: int, N: int, K: int,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 32,
    dtype: str = "float16",
    accum_dtype: str = "float32",
    activation: str = "silu"     
):
    @T.prim_func
    def kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        Bias: T.Tensor((N,), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(
            T.ceildiv(N, block_N),
            T.ceildiv(M, block_M),
            threads=128
        ) as (bx, by):

            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local  = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            for i, j in T.Parallel(block_M, block_N):
                val = C_local[i, j] + Bias[bx * block_N + j]

                if activation == "relu":
                    val = T.max(val, 0)
                elif activation == "gelu":
                    val = 0.5 * val * (1 + T.tanh(0.79788456 * (val + 0.044715 * val * val * val)))
                elif activation == "silu":
                    val = val / (1 + T.exp(-val))

                C_local[i, j] = val

            T.copy(C_local, C[by * block_M, bx * block_N])

    return kernel

M, N, K = 1024, 1024, 1024

A = torch.randn(M, K, device="cuda", dtype=torch.float16)
B = torch.randn(K, N, device="cuda", dtype=torch.float16)
Bias = torch.randn(N, device="cuda", dtype=torch.float16)
C = torch.empty(M, N, device="cuda", dtype=torch.float16)


kernel = fused_matmul_bias_activation(M, N, K, activation="silu")
kernel(A, B, Bias, C)

C_ref = torch.nn.functional.silu(A @ B + Bias)
print("Max error:", (C - C_ref).abs().max().item())
print("Fused GEMM + SiLU successful! ✅")