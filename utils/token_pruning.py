"""
Token Pruning and Variate Selection for iTransformer.

Implements techniques to reduce unnecessary computations:
1. Token Pruning - Remove low-importance tokens based on attention scores
2. Dynamic Token Pruning - Prune tokens adaptively based on uncertainty
3. Variate Selection - Select only the most relevant features/variates
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List


class TokenPruning(nn.Module):
    """
    Token Pruning: Remove low-importance tokens to reduce sequence length.
    
    Based on: "Vision Transformer Slimming" paper
    - Computes token importance scores
    - Prunes tokens with low importance
    - Reduces computational cost of subsequent layers
    """
    
    def __init__(self, pruning_ratio: float = 0.1, metric: str = 'norm'):
        """
        Args:
            pruning_ratio: Ratio of tokens to prune (0.0 - 1.0)
            metric: Metric for importance ('norm', 'entropy', 'attention')
        """
        super().__init__()
        self.pruning_ratio = pruning_ratio
        self.metric = metric
        assert 0.0 <= pruning_ratio < 1.0, "pruning_ratio must be in [0, 1)"
        
    def forward(self, tokens: torch.Tensor, attention_weights: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            tokens: (batch_size, seq_len, d_model)
            attention_weights: Optional (batch_size, n_heads, seq_len, seq_len) for attention-based pruning
            
        Returns:
            pruned_tokens: (batch_size, pruned_len, d_model)
            keep_mask: (batch_size, seq_len) boolean mask
        """
        batch_size, seq_len, d_model = tokens.shape
        
        # Compute importance scores
        if self.metric == 'norm':
            importance = self._compute_norm_importance(tokens)
        elif self.metric == 'entropy':
            importance = self._compute_entropy_importance(tokens)
        elif self.metric == 'attention':
            if attention_weights is None:
                raise ValueError("attention_weights required for attention-based pruning")
            importance = self._compute_attention_importance(attention_weights)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
        
        # Determine which tokens to keep
        num_keep = max(1, int(seq_len * (1 - self.pruning_ratio)))
        _, keep_indices = torch.topk(importance, num_keep, dim=1)
        
        # Create mask and gather tokens
        keep_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=tokens.device)
        keep_mask.scatter_(1, keep_indices, True)
        
        # Gather pruned tokens
        pruned_tokens = tokens[keep_mask].view(batch_size, num_keep, d_model)
        
        return pruned_tokens, keep_mask
    
    @staticmethod
    def _compute_norm_importance(tokens: torch.Tensor) -> torch.Tensor:
        """Compute importance as L2 norm of token embeddings."""
        importance = torch.norm(tokens, p=2, dim=-1)  # (batch_size, seq_len)
        return importance
    
    @staticmethod
    def _compute_entropy_importance(tokens: torch.Tensor) -> torch.Tensor:
        """Compute importance based on token entropy."""
        # Normalize tokens to [0, 1]
        tokens_normalized = F.softmax(tokens, dim=-1)
        # Entropy: -sum(p * log(p))
        entropy = -(tokens_normalized * torch.log(tokens_normalized + 1e-10)).sum(dim=-1)
        return entropy
    
    @staticmethod
    def _compute_attention_importance(attention_weights: torch.Tensor) -> torch.Tensor:
        """
        Compute importance based on attention patterns.
        
        Args:
            attention_weights: (batch_size, n_heads, seq_len, seq_len)
            
        Returns:
            importance: (batch_size, seq_len)
        """
        # Average attention across heads
        attention_mean = attention_weights.mean(dim=1)  # (batch_size, seq_len, seq_len)
        
        # Token importance = sum of incoming attention
        importance = attention_mean.sum(dim=-2)  # (batch_size, seq_len)
        
        return importance


class DynamicTokenPruning(nn.Module):
    """
    Dynamic Token Pruning: Adaptively prune tokens based on uncertainty.
    
    Different pruning ratios for different layers and batches.
    """
    
    def __init__(self, base_pruning_ratio: float = 0.1, metric: str = 'norm'):
        """
        Args:
            base_pruning_ratio: Base pruning ratio to start with
            metric: Metric for importance
        """
        super().__init__()
        self.base_pruning_ratio = base_pruning_ratio
        self.metric = metric
        self.pruner = TokenPruning(base_pruning_ratio, metric)
        
    def forward(self, tokens: torch.Tensor, layer_id: int, num_layers: int,
                attention_weights: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            tokens: (batch_size, seq_len, d_model)
            layer_id: Current layer ID (0-indexed)
            num_layers: Total number of layers
            attention_weights: Optional attention weights
            
        Returns:
            pruned_tokens: (batch_size, pruned_len, d_model)
            keep_mask: (batch_size, seq_len)
        """
        # Adaptive pruning ratio: increase towards deeper layers
        # Early layers: low pruning (preserve information)
        # Later layers: higher pruning (remove redundancy)
        adaptive_ratio = self.base_pruning_ratio * (layer_id / max(1, num_layers - 1))
        
        # Temporarily update pruning ratio
        old_ratio = self.pruner.pruning_ratio
        self.pruner.pruning_ratio = min(adaptive_ratio, 0.5)  # Cap at 50%
        
        pruned_tokens, keep_mask = self.pruner(tokens, attention_weights)
        
        # Restore original ratio
        self.pruner.pruning_ratio = old_ratio
        
        return pruned_tokens, keep_mask


class VariateSelection(nn.Module):
    """
    Variate Selection: Select only the most relevant features for forecasting.
    
    Useful for multivariate forecasting where not all features are equally important.
    """
    
    def __init__(self, input_dim: int, num_select: int, method: str = 'learned'):
        """
        Args:
            input_dim: Number of input variates/features
            num_select: Number of variates to select
            method: Selection method ('learned', 'correlation', 'attention')
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_select = min(num_select, input_dim)
        self.method = method
        
        if method == 'learned':
            # Learnable gating network
            self.gate = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, input_dim),
                nn.Sigmoid()
            )
        elif method == 'attention':
            self.query = nn.Linear(input_dim, 1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch_size, seq_len, input_dim) or (batch_size, input_dim)
            
        Returns:
            selected: Tensor with selected variates
            importance_scores: (batch_size, input_dim) or (input_dim,) importance scores
        """
        if self.method == 'learned':
            return self._learned_selection(x)
        elif self.method == 'correlation':
            return self._correlation_selection(x)
        elif self.method == 'attention':
            return self._attention_selection(x)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _learned_selection(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Learned selection using gating network."""
        if x.dim() == 3:
            # (batch_size, seq_len, input_dim)
            batch_size, seq_len, input_dim = x.shape
            x_flat = x.mean(dim=1)  # Average across time: (batch_size, input_dim)
        else:
            x_flat = x
        
        # Compute importance scores
        importance = self.gate(x_flat)  # (batch_size, input_dim)
        
        # Select top variates
        _, select_indices = torch.topk(importance, self.num_select, dim=-1)
        
        # Gather selected variates
        if x.dim() == 3:
            selected = x[:, :, select_indices]  # (batch_size, seq_len, num_select)
        else:
            selected = x[:, select_indices]  # (batch_size, num_select)
        
        return selected, importance
    
    def _correlation_selection(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Selection based on correlation with target."""
        if x.dim() == 3:
            # Compute correlation across time
            x_mean = x.mean(dim=1)  # (batch_size, input_dim)
            x_std = x.std(dim=1) + 1e-10
            x_normalized = (x - x_mean.unsqueeze(1)) / x_std.unsqueeze(1)
        else:
            x_normalized = x
        
        # Compute importance as max absolute correlation
        importance = torch.abs(x_normalized).max(dim=0)[0] if x.dim() == 3 else torch.abs(x).max(dim=0)[0]
        
        # Select top variates
        _, select_indices = torch.topk(importance, self.num_select)
        
        if x.dim() == 3:
            selected = x[:, :, select_indices]
        else:
            selected = x[:, select_indices]
        
        return selected, importance
    
    def _attention_selection(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Selection using attention mechanism."""
        if x.dim() == 3:
            # (batch_size, seq_len, input_dim)
            batch_size, seq_len, input_dim = x.shape
            importance = self.query(x).squeeze(-1)  # (batch_size, seq_len)
            importance = importance.mean(dim=1)  # (batch_size,)
        else:
            importance = self.query(x).squeeze(-1)  # (batch_size,) or scalar
        
        # Select top variates
        _, select_indices = torch.topk(importance, self.num_select)
        
        if x.dim() == 3:
            selected = x[:, :, select_indices]
        else:
            selected = x[:, select_indices]
        
        return selected, importance


class VariateSelectionGate(nn.Module):
    """
    Variate Selection Gate for iTransformer.
    
    Learns to select the most relevant variates at the beginning of the model.
    """
    
    def __init__(self, input_dim: int, num_select: Optional[int] = None):
        """
        Args:
            input_dim: Number of input variates
            num_select: Number of variates to select (None = use all with learned weights)
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_select = num_select if num_select is not None else input_dim
        
        # Learnable importance scores for each variate
        self.importance_scores = nn.Parameter(torch.ones(input_dim))
        
        # Learnable selection threshold
        self.selection_gate = nn.Linear(input_dim, input_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch_size, seq_len, input_dim)
            
        Returns:
            selected: (batch_size, seq_len, num_select)
            weights: (input_dim,) learned importance weights
        """
        # Apply learned importance scores
        weights = F.softmax(self.importance_scores, dim=0)
        
        # Apply gating
        gate_scores = torch.sigmoid(self.selection_gate(torch.eye(self.input_dim, device=x.device)))
        
        # Combine weights and gate scores
        final_weights = weights * gate_scores.squeeze(-1)
        
        # Select top variates
        _, select_indices = torch.topk(final_weights, self.num_select)
        
        # Gather selected variates
        selected = x[:, :, select_indices]  # (batch_size, seq_len, num_select)
        
        return selected, final_weights
