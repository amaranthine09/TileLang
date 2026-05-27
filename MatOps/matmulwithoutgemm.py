import tilelang 
import tilelang.language as T 
import torch 
from tilelang import jit 

@jit
def gemm(
    M:int, N:int, K:int,
    block_M: int = 128, 
    block_N: int = 128, 
    block_K: int = 32,
    micro_M: int = 8,
    micro_N: int = 8, 
    dtype: str =  "float16", 
    acc_dtype:int = "float32"
):
    num_threads= (block_M//micro_M)*(block_N//micro_N)
    @T.prim_func
    def gemm_kernel(
        A: T.Tensor((block_M, block_K), dtype),
        B: T.Tensor((block_K, block_N), dtype),
        C: T.Tensor((block_N, block_M), acc_dtype)
    ):
        with T.kernel(
            T.ceildiv(N, block_N),
            T.ceildiv(M, block_M),
            threads= num_threads

        ) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            acc = T.alloc_local((micro_M, micro_N), acc_dtype)
            T.clear(acc)

            a_reg = T.alloc_local((micro_M,), acc_dtype)
            b_reg = T.alloc_local((micro_N,), acc_dtype)

            for ko in T.T.Pipelined(T.ceildiv(K, block_K), num_stages = 2):
                T.copy(A[by*block_M, ko*block_K], A_shared)
                T.copy(B[ko*block_K, bx*block_N], B_shared)

                for tid in T.thread_binding(num_threads, thread = "threadIdx.x"):
                    ty = tid//(block_N//micro_N)
                    tx = tid % (block_N//micro_N)
                    row_start = ty * micro_M
                    col_start = tx * micro_N

                    for k in range(block_K):
                        for i in T.unroll(micro_M):
                            a_reg[i] = T.cast(A_shared[row_start + i, K], acc_dtype)
                        for j in T.unroll(micro_N):
                            b_reg[j] = T.cast(B_shared[k, col_start + j], acc_dtype)

                        for i in T.unroll(micro_M):
                            for j in T.unroll(micro_N):
                                acc[i, j] = acc[i, j] + a_reg[i] * b_reg[j]

            for tid in T.thread_binding(num_threads, thread="threadIdx.x"):
                ty = tid // (block_N // micro_N)
                tx = tid % (block_N // micro_N)
                row_start = ty * micro_M
                col_start = tx * micro_N

                for i in T.unroll(micro_M):
                    for j in T.unroll(micro_N):
                        C[by * block_M + row_start + i, bx * block_N + col_start + j] = T.cast(acc[i, j], dtype)   

    return gemm_kernel

M, N, K = 1024, 1024, 1024

A = torch.randn(M, K, device="mps", dtype=torch.float16)
B = torch.randn(K, N, device="mps", dtype=torch.float16)
C = torch.empty(M, N, device="mps", dtype=torch.float16)

kernel = gemm(M, N, K)
kernel(A, B, C)

C_ref = A @ B
print("Max error:", (C - C_ref).abs().max().item())
print("✅ Optimized manual tiled matrix multiplication successful!")         