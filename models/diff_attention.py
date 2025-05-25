import torch
import torch.nn.functional as F
from torch import nn, Tensor
from math import sqrt

class DiffAttn(nn.Module):
    """
    Differential Attention module.
    
    This module computes attention weights based on the difference between two sets of queries and keys.
    
    Attributes:
    - d (int): The dimensionality of the attention weights.
    - embedding_dim (int): The dimensionality of the input embeddings.
    - W_q (nn.Linear): Linear layer for transforming queries.
    - W_k (nn.Linear): Linear layer for transforming keys.
    - W_v (nn.Linear): Linear layer for transforming values.
    """
    def __init__(self, d: int, embedding_dim: int):
        super(DiffAttn, self).__init__()
        self.d = d
        self.W_q = nn.Linear(embedding_dim, 2 * d)
        self.W_k = nn.Linear(embedding_dim, 2 * d)
        self.W_v = nn.Linear(embedding_dim, d)  # Changed to output d dimensions

    def forward(self, X1: Tensor, X2: Tensor, λ: float) -> Tensor:
        """
        Forward pass of the Differential Attention module.
        
        Args:
        - X (Tensor): Input tensor.
        - λ (float): Scaling factor for the difference.
        
        Returns:
        - Tensor: Output tensor.
        """
        
        Q1 = self.W_q(X1)
        K1 = self.W_k(X1)
        Q2 = self.W_q(X2)
        K2 = self.W_k(X2)      
        X = torch.cat([X1, X2], dim=1)
        V = self.W_v(X)

        # Q1, Q2 = self.split(Q)
        # K1, K2 = self.split(K)

        s = 1 / sqrt(self.d)
        
        A1 = (Q1 @ K1.transpose(-1, -2)) * s
        A2 = (Q2 @ K2.transpose(-1, -2)) * s
        
        A1_softmax = F.softmax(A1, dim=-1)
        A2_softmax = F.softmax(A2, dim=-1)
        
        result = (A1_softmax - λ * A2_softmax) @ V
        return result

    def split(X: Tensor) -> (Tensor, Tensor):
        """
        Splits the input tensor into two halves along the last dimension.
        
        Args:
        - X (Tensor): Input tensor.
        
        Returns:
        - Tuple[Tensor, Tensor]: Two tensors, each containing half of the input dimensions.
        """
        half_dim = X.shape[-1] // 2
        return X[..., :half_dim], X[..., half_dim:]


class MultiHeadDifferentialAttention(nn.Module):
    """
    Multi-Head Differential Attention module.
    
    This module applies the Differential Attention mechanism multiple times in parallel.
    
    Attributes:
    - h (int): The number of attention heads.
    - d (int): The dimensionality of the attention weights.
    - embedding_dim (int): The dimensionality of the input embeddings.
    - λinit (float): The initial scaling factor for the difference.
    - diff_attn_heads (nn.ModuleList): List of Differential Attention modules.
    - W_o (nn.Linear): Linear layer for output transformation.
    - norm (nn.LayerNorm): Layer normalization module.
    """
    def __init__(self, h: int, d: int, embedding_dim: int, λinit: float):
        super(MultiHeadDifferentialAttention, self).__init__()
        self.h = h
        self.d = d
        self.λinit = λinit
        self.embedding_dim = embedding_dim
        self.diff_attn_heads = nn.ModuleList([DiffAttn(d, embedding_dim) for _ in range(h)])
        self.W_o = nn.Linear(h * d, embedding_dim)  # Changed to h * d
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, X: Tensor, λ: float) -> Tensor:
        """
        Forward pass of the Multi-Head Differential Attention module.
        
        Args:
        - X (Tensor): Input tensor.
        - λ (float): Scaling factor for the difference.
        
        Returns:
        - Tensor: Output tensor.
        """

        O_list = [head(X, λ) for head in self.diff_attn_heads]
        
        O_concat = torch.cat(O_list, dim=-1)

        # Apply the output transformation
        result = self.W_o(O_concat)

        # Apply LayerNorm
        result = self.norm(result)

        # Scale by λinit
        result = result * (1 - self.λinit)

        return result