# 🧩 TileLang Programs

A collection of GPU kernel programs written in **[TileLang](https://github.com/tile-ai/tilelang)** — a high-level, Pythonic domain-specific language for writing efficient GPU kernels using tile-based programming abstractions.

This repository documents my journey of learning and experimenting with TileLang, from basic building blocks to real-world high-performance kernels.

---

## 📖 What is TileLang?

TileLang is a DSL (Domain-Specific Language) built on top of [Apache TVM](https://tvm.apache.org/) that lets you write GPU kernels in Python using a **tile-based programming model**. Instead of dealing with raw CUDA/HIP thread-level code, TileLang provides high-level abstractions like:

| Concept | TileLang API | What It Does |
|---|---|---|
| **Shared Memory Tiles** | `T.alloc_shared()` | Allocates a tile in GPU shared memory for fast data reuse across threads within a block |
| **Register Fragments** | `T.alloc_fragment()` | Allocates a tile in per-thread registers for accumulation (e.g., matrix multiply accumulators) |
| **Tiled Copy** | `T.copy()` | Copies a 2D slice from global memory → shared memory or shared → registers, handling bounds and layout automatically |
| **Tiled GEMM** | `T.gemm()` | Performs a matrix-multiply-accumulate on tiles, mapping to Tensor Core instructions on supported hardware |
| **Pipelined Loops** | `T.Pipelined()` | Overlaps memory loads with compute using software pipelining (double/triple buffering) for latency hiding |
| **Parallel Loops** | `T.Parallel()` | Expresses element-wise operations across a tile, distributed over the thread block |
| **Kernel Launch** | `T.Kernel()` | Defines the GPU grid dimensions and threads-per-block for the kernel launch configuration |

TileLang compiles these abstractions down to optimized CUDA/HIP/Metal code through TVM's compilation pipeline, generating kernels that can rival hand-tuned implementations.

---

## 🗂️ Repository Structure

```
TileLang/
├── README.md              # This file
├── .gitignore             # Git ignore rules
├── Basic_programs         # Basic TileLang programs (getting started)
└── tilelang_real.py       # Full GEMM kernels with verification
```

---

## 🚀 Programs

### 1. `tilelang_real.py` — GEMM Kernels (Matrix Multiplication)

This is the main program implementing two GPU kernels using TileLang's **lazy-style JIT API**:

#### 🔹 Simple GEMM — `C = A × B`

A clean, tiled matrix multiplication kernel demonstrating the core TileLang workflow:

```python
@tilelang.jit(target=TARGET)
def simple_gemm(M, N, K, block_M=64, block_N=64, block_K=64, dtype=T.float16, accum_dtype=T.float32):
    @T.prim_func
    def kernel(A: T.Tensor((M, K), dtype), B: T.Tensor((K, N), dtype), C: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local  = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[by * block_M, bx * block_N])
    return kernel
```

**How it works step-by-step:**

1. **Grid Launch** — The output matrix `C (M×N)` is divided into tiles of size `block_M × block_N`. Each tile is assigned to one thread block on the GPU.
2. **Shared Memory Allocation** — Each block allocates shared memory tiles `A_shared` and `B_shared` to stage data from global memory, enabling fast reuse across all 128 threads.
3. **Accumulator Init** — A register-level fragment `C_local` is allocated for each thread's portion of the output tile and cleared to zero.
4. **Pipelined K-Loop** — The inner dimension `K` is traversed in chunks of `block_K`. Using `T.Pipelined` with `num_stages=3` (triple buffering), the next iteration's memory loads overlap with the current iteration's compute — hiding memory latency.
5. **Tile GEMM** — `T.gemm()` performs the matrix multiply-accumulate on the shared memory tiles, mapping to Tensor Core `mma` instructions on NVIDIA GPUs.
6. **Write Back** — The accumulated result in registers is copied back to global memory.

#### 🔹 GEMM + ReLU — `C = max(A × B, 0)`

Extends the simple GEMM with a **fused ReLU activation**, demonstrating how to add element-wise operations after the matrix multiply without a separate kernel launch:

```python
# After the GEMM loop, before writing back:
for i, j in T.Parallel(block_M, block_N):
    C_local[i, j] = T.max(C_local[i, j], 0)
```

This fuses the activation directly into the GEMM kernel, avoiding an extra global memory round-trip that a separate ReLU kernel would require — a common optimization in deep learning inference.

#### ⚙️ Execution Modes

The program automatically detects the available hardware:

| Environment | Behavior |
|---|---|
| **NVIDIA GPU (CUDA)** | Compiles & runs both kernels, verifies results against PyTorch `torch.matmul` |
| **No GPU (CPU only)** | Generates and prints the TIR (Tensor IR) intermediate representation for inspection |

---

### 2. `Basic_programs` — Getting Started

Placeholder for basic TileLang programs covering fundamental concepts. Will be populated with:
- Vector addition kernels
- Element-wise operations
- Simple reduction patterns
- Memory hierarchy exploration

---

## 🛠️ Setup & Requirements

### Prerequisites

- **Python** 3.8+
- **PyTorch** (for tensor creation and verification)
- **TileLang** (`pip install tilelang`)
- **NVIDIA GPU** with CUDA support (for execution; TIR generation works without GPU)

### Installation

```bash
# Clone this repository
git clone https://github.com/amaranthine09/TileLang.git
cd TileLang

# Install dependencies
pip install torch tilelang
```

### Running

```bash
# Run the GEMM kernels
python tilelang_real.py
```

**Expected output (with GPU):**
```
============================================================
TileLang - Real GPU Programming Language
============================================================

CUDA available: True
MPS available:  False

--- Running Simple GEMM on GPU ---
Input A shape: torch.Size([256, 256])
Input B shape: torch.Size([256, 256])
Output C shape: torch.Size([256, 256])
✓ Results match! Simple GEMM kernel works correctly.

--- Running GEMM with ReLU on GPU ---
✓ Results match! GEMM ReLU kernel works correctly.
```

**Expected output (without GPU):**
```
============================================================
TileLang - Real GPU Programming Language
============================================================

CUDA available: False
MPS available:  True
⚠ No CUDA GPU detected. Will generate TIR program only.

--- Generating TIR for Simple GEMM ---
✓ TIR program generated successfully!
<TIR source code...>
```

---

## 🎯 Supported GPU Targets

| Target | Flag | Status |
|---|---|---|
| NVIDIA (CUDA) | `target="cuda"` | ✅ Fully supported |
| AMD (ROCm/HIP) | `target="hip"` | ✅ Supported |
| Apple Metal | `target="metal"` | 🧪 Experimental |

---

## 📚 Key Concepts & Resources

- **Tile-Based Programming** — Instead of thinking about individual threads, you think about tiles (2D blocks of data) and how they move through the memory hierarchy.
- **Software Pipelining** — `T.Pipelined` automatically generates double/triple buffered loops so that memory loads and computation overlap on the GPU.
- **Operator Fusion** — By adding element-wise operations (like ReLU) inside the kernel before the final write-back, we avoid extra kernel launches and global memory traffic.

### Useful Links

- [TileLang GitHub](https://github.com/tile-ai/tilelang)
- [TileLang Documentation](https://tilelang.github.io/tilelang/)
- [Apache TVM](https://tvm.apache.org/)
- [NVIDIA Tensor Cores](https://developer.nvidia.com/tensor-cores)

---

## 📌 Roadmap

- [x] Simple GEMM kernel
- [x] Fused GEMM + ReLU kernel
- [x] Automatic hardware detection (CUDA / CPU fallback)
- [ ] Attention kernel (Flash Attention style)
- [ ] Convolution kernels
- [ ] Reduction and softmax kernels
- [ ] Performance benchmarks vs cuBLAS
- [ ] Multi-head attention for transformer inference

---

## 🤝 About

This repository is maintained by **[@amaranthine09](https://github.com/amaranthine09)** as a learning resource and reference for writing GPU kernels with TileLang. More programs will be added as I continue exploring tile-based GPU programming.

If you find this helpful, feel free to ⭐ the repo!

---

*Built with [TileLang](https://github.com/tile-ai/tilelang) — Tile-based GPU programming made Pythonic.*
