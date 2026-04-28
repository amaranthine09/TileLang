import tilelang
from tilelang import language as T
import torch

print("=" * 60)
print("TileLang - Real GPU Programming Language")
print("=" * 60)


HAS_CUDA = torch.cuda.is_available()
HAS_MPS = torch.backends.mps.is_available()

TARGET = "cuda" if HAS_CUDA else "cuda -arch=sm_80"

print(f"\nCUDA available: {HAS_CUDA}")
print(f"MPS available:  {HAS_MPS}")
if not HAS_CUDA:
    print("⚠ No CUDA GPU detected. Will generate TIR program only.")


@tilelang.jit(target=TARGET)
def gemm_relu(
    M, N, K,
    block_M: int = 64,
    block_N: int = 64,
    block_K: int = 64,
    dtype=T.float16,
    accum_dtype=T.float32,
):
    """
    GEMM with ReLU activation: C = relu(A @ B)
    Lazy-style: the outer function takes config params,
    the inner @T.prim_func defines the kernel.
    """
    @T.prim_func
    def kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            for i, j in T.Parallel(block_M, block_N):
                C_local[i, j] = T.max(C_local[i, j], 0)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return kernel


@tilelang.jit(target=TARGET)
def simple_gemm(
    M, N, K,
    block_M: int = 64,
    block_N: int = 64,
    block_K: int = 64,
    dtype=T.float16,
    accum_dtype=T.float32,
):
    """Simple GEMM: C = A @ B"""
    @T.prim_func
    def kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return kernel


M, N, K = 256, 256, 256

if HAS_CUDA:
    # ---- Full execution on GPU ----
    device = "cuda"
    a = torch.randn(M, K, dtype=torch.float16, device=device)
    b = torch.randn(K, N, dtype=torch.float16, device=device)
    c_ref = a @ b

    print("\n--- Running Simple GEMM on GPU ---")
    kernel = simple_gemm(M, N, K)
    c = kernel(a, b)

    print(f"Input A shape: {a.shape}")
    print(f"Input B shape: {b.shape}")
    print(f"Output C shape: {c.shape}")

    try:
        torch.testing.assert_close(c, c_ref, rtol=1e-2, atol=1e-2)
        print("✓ Results match! Simple GEMM kernel works correctly.")
    except Exception as e:
        print(f"⚠ Verification: {e}")

    print("\n--- Running GEMM with ReLU on GPU ---")
    kernel_relu = gemm_relu(M, N, K)
    c_relu = kernel_relu(a, b)
    c_ref_relu = torch.relu(a @ b)

    try:
        torch.testing.assert_close(c_relu, c_ref_relu, rtol=1e-2, atol=1e-2)
        print("✓ Results match! GEMM ReLU kernel works correctly.")
    except Exception as e:
        print(f"⚠ Verification: {e}")

    print("\n--- Kernel Source Code Preview ---")
    source = kernel.get_kernel_source()
    print(source[:800] + "...")

else:
    # ---- TIR generation only (no GPU available) ----
    print("\n--- Generating TIR for Simple GEMM ---")
    tir_func = simple_gemm.get_tir(M, N, K)
    print("✓ TIR program generated successfully!\n")
    tir_source = tir_func.script()
    print(tir_source[:2000])
    if len(tir_source) > 2000:
        print(f"\n... ({len(tir_source)} chars total, truncated)")

    print("\n--- Generating TIR for GEMM with ReLU ---")
    tir_relu = gemm_relu.get_tir(M, N, K)
    print("✓ GEMM ReLU TIR generated successfully!\n")
    relu_source = tir_relu.script()
    print(relu_source[:1000])
    if len(relu_source) > 1000:
        print(f"\n... ({len(relu_source)} chars total, truncated)")


print("\n" + "=" * 60)
print("TileLang Program Complete!")
print("=" * 60)
print("\nSupported targets for full execution:")
print("  - NVIDIA GPU:  target='cuda'")
print("  - AMD GPU:     target='hip'")
print("  - Apple Metal: target='metal' (experimental)")