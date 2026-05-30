import tilelang 
import tilelang.language as T 
import torch
import torch.nn as nn 
from tilelang import jit 

class attentionblock(nn.Module):
    def __init__(self, d_model: int, dim_head: int):
        super().__init__()
        self.d_model = d_model
        