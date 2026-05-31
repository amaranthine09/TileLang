import tilelang
import tilelang.language as T 
import torch 
from tilelang import jit 

@jit 
def softmax(A, block_N, block_M):
    log2_e = 1.4426950408889634
    N, M = T.const("N, M")
    dtype = T.float32
    A: T.Tensor((N, M), dtype)
    B = T.empty((N, M), dtype)

    with T.Kernel(T.ceildiv(N, block_N), threads=256) as pid_n:
        n_idx = pid_n * block_N

        A_local = T.alloc_fragment((block_N, block_M), dtype)
        max_val = T.alloc_fragment((block_N,), dtype)
        sum_val = T.alloc_fragment((block_N,), dtype)
        B_local = T.alloc_fragment((block_N, block_M), dtype)

        T.fill(max_val, -T.infinity(dtype))
        for m_start in T.serial(T.ceildiv(M, block_M)):
            m_idx = m_start * block_M
            T.copy(A[n_idx:n_idx + block_N, m_idx:m_idx + block_M], A_local)
            cur_max = T.alloc_fragment((block_N,), dtype)
            T.reduce_max(A_local, cur_max, dim=1, clear=True)
            for i in T.Parallel(block_N):
                max_val[i] = T.maximum(max_val[i], cur_max[i])

        T.clear(sum_val)
        for m_start in T.serial(T.ceildiv(M, block_M)):
            m_idx = m_start * block_M
            T.copy(A[n_idx:n_idx + block_N, m_idx:m_idx + block_M], A_local)
            for i, j in T.Parallel(block_N, block_M):
                B_local[i, j] = T.exp2((A_local[i, j] - max_val[i]) * log2_e)
            T.reduce_sum(B_local, sum_val, dim=1, clear=False)

        for m_start in T.serial(T.ceildiv(M, block_M)):
            m_idx = m_start * block_M
            T.copy(A[n_idx:n_idx + block_N, m_idx:m_idx + block_M], A_local)
            for i, j in T.Parallel(block_N, block_M):
                val = T.exp2((A_local[i, j] - max_val[i]) * log2_e)
                B_local[i, j] = val / sum_val[i]
            T.copy(B_local, B[n_idx:n_idx + block_N, m_idx:m_idx + block_M])

    return B

if __name__ == "__main__":
    torch.manual_seed(42)
    N, M = 256, 512
    A = torch.randn(N, M, dtype=torch.float32, device="cuda")

    B_tile = softmax(A, block_N=64, block_M=64)
    B_ref = torch.softmax(A, dim=1)

    max_diff = (B_tile - B_ref).abs().max().item()
    print(f"Max diff: {max_diff:.2e}")
    print("✅ Correct!" if torch.allclose(B_tile, B_ref, atol=1e-5) else "❌ Mismatch")