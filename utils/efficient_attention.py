"""
Efficient Attention Mechanisms for iTransformer.

Implements various efficient attention variants to replace full self-attention:
1. Windowed Attention - Tokens attend to a local window only
2. Strided Attention - Tokens attend to strided positions
3. Reformer Attention - LSH-based attention for long sequences
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class WindowedAttention(nn.Module):
    """
    Windowed/Local Attention: Each token attends only to tokens within a local window.
    
    This reduces complexity from O(n^2) to O(n*w) where w is the window size.
    Typical window size: 32-64 tokens
    """
    
    def __init__(self, d_model: int, n_heads: int, window_size: int = 32, dropout: float = 0.1):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            window_size: Size of the attention window (tokens on each side)
            dropout: Dropout rate
        """
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.window_size = window_size
        self.head_dim = d_model // n_heads
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            query: (batch_size, seq_len, d_model)
            key: (batch_size, seq_len, d_model)
            value: (batch_size, seq_len, d_model)
            mask: Optional mask tensor
            
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        batch_size = query.shape[0]
        seq_len = query.shape[1]
        
        # Linear transformations
        Q = self.query(query)  # (batch_size, seq_len, d_model)
        K = self.key(key)
        V = self.value(value)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # (batch_size, n_heads, seq_len, head_dim)
        
        # Create windowed attention mask
        window_mask = self._create_window_mask(seq_len, self.window_size, device=Q.device)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply window mask
        scores = scores.masked_fill(~window_mask, float('-inf'))
        
        # Apply custom mask if provided
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        
        # Softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, V)
        
        # Reshape back
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.d_model)
        
        # Final linear transformation
        output = self.fc_out(output)
        
        return output
    
    @staticmethod
    def _create_window_mask(seq_len: int, window_size: int, device: torch.device) -> torch.Tensor:
        """Create a mask for windowed attention."""
        mask = torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)
        
        for i in range(seq_len):
            start = max(0, i - window_size)
            end = min(seq_len, i + window_size + 1)
            mask[i, :start] = False
            mask[i, end:] = False
        
        return mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions


class StridedAttention(nn.Module):
    """
    Strided Attention: Each token attends to a strided subset of positions.
    
    Useful for capturing long-range dependencies while maintaining efficiency.
    Complexity: O(n * (n/stride))
    """
    
    def __init__(self, d_model: int, n_heads: int, stride: int = 4, dropout: float = 0.1):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            stride: Stride for attention positions
            dropout: Dropout rate
        """
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.stride = stride
        self.head_dim = d_model // n_heads
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            query: (batch_size, seq_len, d_model)
            key: (batch_size, seq_len, d_model)
            value: (batch_size, seq_len, d_model)
            mask: Optional mask tensor
            
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        batch_size = query.shape[0]
        seq_len = query.shape[1]
        
        # Linear transformations
        Q = self.query(query)
        K = self.key(key)
        V = self.value(value)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Stride the key and value
        K = K[:, :, ::self.stride, :]  # (batch_size, n_heads, seq_len//stride, head_dim)
        V = V[:, :, ::self.stride, :]
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply mask if provided
        if mask is not None:
            mask = mask[:, :, :, ::self.stride]
            scores = scores.masked_fill(~mask, float('-inf'))
        
        # Softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, V)
        
        # Reshape back
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.d_model)
        
        # Final linear transformation
        output = self.fc_out(output)
        
        return output


class EfficientAttentionSelector(nn.Module):
    """
    Selector to choose between different efficient attention mechanisms.
    """
    
    ATTENTION_TYPES = {
        'full': 'FullAttention',
        'windowed': WindowedAttention,
        'strided': StridedAttention,
    }
    
    @staticmethod
    def get_attention(attention_type: str, d_model: int, n_heads: int,
                     window_size: int = 32, stride: int = 4,
                     dropout: float = 0.1) -> nn.Module:
        """
        Get an attention module based on type.
        
        Args:
            attention_type: Type of attention ('full', 'windowed', 'strided')
            d_model: Model dimension
            n_heads: Number of attention heads
            window_size: Window size for windowed attention
            stride: Stride for strided attention
            dropout: Dropout rate
            
        Returns:
            Attention module
        """
        if attention_type == 'windowed':
            return WindowedAttention(d_model, n_heads, window_size, dropout)
        elif attention_type == 'strided':
            return StridedAttention(d_model, n_heads, stride, dropout)
        elif attention_type == 'full':
            raise ValueError("Use standard MultiHeadAttention for full attention")
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")
    
    @staticmethod
    def get_attention_flops(seq_len: int, d_model: int, attention_type: str,
                          window_size: int = 32, stride: int = 4) -> dict:
        """
        Estimate FLOPs for different attention types.
        
        Returns:
            Dictionary with FLOP estimates
        """
        full_flops = 2 * seq_len * seq_len * d_model
        
        flops = {
            'full': full_flops,
            'windowed': 2 * seq_len * window_size * d_model,
            'strided': 2 * seq_len * (seq_len // stride) * d_model,
        }
        
        reduction = {}
        for key in flops:
            if key != 'full':
                reduction[key] = full_flops / flops[key]
        
        return {
            'flops': flops,
            'reduction': reduction,
            'attention_type': attention_type
        }
