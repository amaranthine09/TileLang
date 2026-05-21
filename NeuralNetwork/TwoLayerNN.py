import torch
import torch.nn as nn
from tilelang import jit
import tilelang.language as T

@jit
def fused_linear_silu(M, N, K):
    @T.prim_func
    def kernel(
        A: T.Tensor((M, K), "float16"),
        W: T.Tensor((K, N), "float16"),
        Bias: T.Tensor((N,), "float16"),
        C: T.Tensor((M, N), "float16"),
    ):
        with T.Kernel(T.ceildiv(N, 128), T.ceildiv(M, 128), threads=128) as (bx, by):
            A_shared = T.alloc_shared((128, 32), "float16")
            W_shared = T.alloc_shared((32, 128), "float16")
            C_local  = T.alloc_fragment((128, 128), "float32")

            T.clear(C_local)

            for ko in T.Pipelined(T.ceildiv(K, 32), num_stages=2):
                T.copy(A[by*128 : (by+1)*128, ko*32 : (ko+1)*32], A_shared)
                T.copy(W[ko*32 : (ko+1)*32, bx*128 : (bx+1)*128], W_shared)
                T.gemm(A_shared, W_shared, C_local)

            for i, j in T.Parallel(128, 128):
                val = C_local[i, j] + Bias[bx*128 + j]
                val = val / (1 + T.exp(-val))          
                C_local[i, j] = val

            T.copy(C_local, C[by*128 : (by+1)*128, bx*128 : (bx+1)*128])
    return kernel


@jit
def fused_linear(M, N, K):
    @T.prim_func
    def kernel(
        A: T.Tensor((M, K), "float16"),
        W: T.Tensor((K, N), "float16"),
        Bias: T.Tensor((N,), "float16"),
        C: T.Tensor((M, N), "float16"),
    ):
        with T.Kernel(T.ceildiv(N, 128), T.ceildiv(M, 128), threads=128) as (bx, by):
            A_shared = T.alloc_shared((128, 32), "float16")
            W_shared = T.alloc_shared((32, 128), "float16")
            C_local  = T.alloc_fragment((128, 128), "float32")

            T.clear(C_local)

            for ko in T.Pipelined(T.ceildiv(K, 32), num_stages=2):
                T.copy(A[by*128 : (by+1)*128, ko*32 : (ko+1)*32], A_shared)
                T.copy(W[ko*32 : (ko+1)*32, bx*128 : (bx+1)*128], W_shared)
                T.gemm(A_shared, W_shared, C_local)

            for i, j in T.Parallel(128, 128):
                C_local[i, j] = C_local[i, j] + Bias[bx*128 + j]

            T.copy(C_local, C[by*128 : (by+1)*128, bx*128 : (bx+1)*128])
    return kernel

class TileLangMLP(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64, output_dim=10):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(input_dim, hidden_dim, device="cuda", dtype=torch.float16) * 0.01)
        self.b1 = nn.Parameter(torch.zeros(hidden_dim, device="cuda", dtype=torch.float16))
        self.W2 = nn.Parameter(torch.randn(hidden_dim, output_dim, device="cuda", dtype=torch.float16) * 0.01)
        self.b2 = nn.Parameter(torch.zeros(output_dim, device="cuda", dtype=torch.float16))

    def forward(self, x):
        hidden = fused_linear_silu(x.shape[0], 64, x.shape[1])(x, self.W1, self.b1)
        out = fused_linear(x.shape[0], 10, 64)(hidden, self.W2, self.b2)
        return out


model = TileLangMLP().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

x = torch.randn(32, 128, device="cuda", dtype=torch.float16)
y = torch.randint(0, 10, (32,), device="cuda")

print("Starting training...")

for epoch in range(5):
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")

print("\nTraining completed successfully!")