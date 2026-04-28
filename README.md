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
└── <program_folder>/      # Each program category gets its own folder
    └── *.py               # TileLang kernel implementations
```

> **Note:** Programs will be organized into dedicated folders as they are added — each folder will contain related kernels, utilities, and examples for a specific topic (e.g., GEMM, attention, convolution, etc.).

---

## 🚀 Programs

Programs are organized into **dedicated folders** by topic. Each folder contains TileLang kernels, utilities, and examples for a specific area of GPU programming.

| Folder | Topic | Status |
|---|---|---|
| `gemm/` | Matrix multiplication kernels (GEMM, fused GEMM+ReLU) | 🔜 Coming soon |
| `basics/` | Fundamental kernels (vector add, element-wise ops) | 🔜 Coming soon |
| `attention/` | Attention kernels (Flash Attention style) | 🔜 Planned |
| `convolution/` | Convolution kernels for deep learning | 🔜 Planned |
| `reductions/` | Reduction & softmax kernels | 🔜 Planned |

> More folders will be added as new programs are written. Each folder will include its own README with detailed explanations of the kernels inside.

### What to Expect in Each Program

Every program in this repo will follow this pattern using TileLang's **lazy-style JIT API**:

```python
import tilelang
from tilelang import language as T

@tilelang.jit(target="cuda")
def my_kernel(M, N, block_M=64, block_N=64, dtype=T.float16):
    @T.prim_func
    def kernel(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            # 1. Allocate shared memory & register tiles
            # 2. Load data from global → shared memory
            # 3. Compute on tiles (GEMM, element-wise, etc.)
            # 4. Write results back to global memory
            pass
    return kernel
```

Each program will include:
- ✅ Detailed inline comments explaining every step
- ✅ Automatic hardware detection (CUDA / CPU fallback for TIR generation)
- ✅ Verification against PyTorch reference implementations
- ✅ Step-by-step explanations in the folder's README

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
# Navigate into a program folder and run
python <folder>/<script>.py

# Example (once GEMM programs are added):
python gemm/simple_gemm.py
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

- [ ] GEMM kernels (simple, fused GEMM+ReLU)
- [ ] Basic kernels (vector add, element-wise ops)
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
