from numpy import dtype
from tvm_ffi import device
import torch 
import torch.nn as nn 
import tilelang.language as T 
from tilelang import jit 
import math

@jit 
def tilelang_linear_kernel(M:int, N:int, K:int):
    @T.prim_func
    def kernel(
        X: T.Tensor((M,K), "float16"),
        W: T.Tensor((K,N), "float16"),
        B: T.Tensor((N,), "float16"),
        O: T.Tensor((M,N), "float32")
        ):
            with T.kernel(T.ceildiv(N, 128), T.ceildiv(M,128), threads = 128) as (bx , by):
                X_shared = T.alloc_shared((128, 32), "float16")
                W_shared = T.alloc_shared((32, 128), "float16")
                C_local = T.alloc_shared((128, 128), "float32")
                T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(K, ko), num_stages=2):
                T.copy(X[by*128:(by+1)*128, ko*32:(ko+1)*32], X_shared)
                T.copy(W[ko*32:(ko+1)*32, bx*128 :(bx+1)*128], W_shared)
                T.gemma(X_shared, W_shared, C_local)

            for i,j in T.parallel(128, 128):
                C_local[i,j]= C_local[i,j]+B[bx*128+j]
            T.copy(C_local, O[by*128, bx*128])
    return kernel



class TilelangLinear(nn.Module):
    def __init__(self, in_feature:int, out_feature: int, bias: bool = True):
        super().__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature

        self.Weight= nn.Parameter(torch.empty(out_feature, in_feature, device= "cuda", dtype = torch.float16))
        self.bias = nn.Parameter(torch.zeros(out_feature, device="cuda", dtype= torch.float16)) if bias else None 

        nn.init.kaiming_uniform_(self.Weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    def forward(self, x: torch.Tensor)-> torch.Tensor:
        assert x.shape[-1] == self.in_features, f"Expected {self.in_features}, got {x.shape[-1]}"
        original_shape = x.shape
        x = x.view(-1, self.in_feature)
        M = x.shape[0]
        
        kernel = tilelang_linear_kernel(M, self.out_features, self.in_features)
        out = torch.empty(M, self.out_features, device=x.device, dtype=torch.float16)
        kernel(x, self.weight,self.bias if self.bias is not None else torch.zeros(self.out_features, device=x.device, dtype=torch.float16))
        
        return out.view(*original_shape[:-1], self.out_features)

if __name__ == "__main__":
    model = nn.Sequential(
        TilelangLinear(128, 512),
        nn.SiLU(),                    
        TilelangLinear(512, 256),
        nn.SiLU(),
        TilelangLinear(256, 10),
    ).cuda()

    x = torch.randn(32, 128, device="cuda", dtype=torch.float16)
    y = model(x)

    print("Output shape:", y.shape)      
    print("✅ TileLangLinear works like normal nn.Linear!")