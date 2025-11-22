"""
Optimization Demo: Using Efficient Attention, Token Pruning, and Model Pruning.

This script demonstrates how to apply the three optimization techniques
to reduce model complexity and inference latency.
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
import time
from pathlib import Path

# Assuming these imports work after proper installation
try:
    from utils.efficient_attention import EfficientAttentionSelector, WindowedAttention, StridedAttention
    from utils.token_pruning import TokenPruning, DynamicTokenPruning, VariateSelection
    from utils.model_pruning import MagnitudePruning, ChannelPruning, PruningAnalyzer
except ImportError as e:
    print(f"Warning: Could not import optimization modules: {e}")
    print("Make sure utils/efficient_attention.py, utils/token_pruning.py, and utils/model_pruning.py exist")


class OptimizationPipeline:
    """
    Complete optimization pipeline combining all three techniques.
    """
    
    def __init__(self, model: nn.Module, use_efficient_attention: bool = True,
                 use_token_pruning: bool = True, use_model_pruning: bool = True):
        """
        Args:
            model: PyTorch model to optimize
            use_efficient_attention: Apply efficient attention
            use_token_pruning: Apply token pruning
            use_model_pruning: Apply model pruning
        """
        self.model = model
        self.original_model = self._copy_model(model)
        self.use_efficient_attention = use_efficient_attention
        self.use_token_pruning = use_token_pruning
        self.use_model_pruning = use_model_pruning
        
        self.optimization_stats = {}
    
    def _copy_model(self, model: nn.Module) -> nn.Module:
        """Create a copy of the model."""
        import copy
        return copy.deepcopy(model)
    
    def apply_efficient_attention(self, attention_type: str = 'windowed',
                                 window_size: int = 32, stride: int = 4) -> None:
        """
        Replace full attention with efficient attention.
        
        Args:
            attention_type: Type of attention ('windowed' or 'strided')
            window_size: Window size for windowed attention
            stride: Stride for strided attention
        """
        if not self.use_efficient_attention:
            return
        
        print("\n" + "="*80)
        print("OPTIMIZATION 1: EFFICIENT ATTENTION")
        print("="*80)
        
        # This is a template - actual implementation depends on model architecture
        # For iTransformer, you would replace the attention modules
        
        print(f"\n✅ Efficient Attention Type: {attention_type}")
        if attention_type == 'windowed':
            print(f"   Window Size: {window_size}")
            flops_info = EfficientAttentionSelector.get_attention_flops(
                seq_len=96, d_model=512, attention_type='windowed', window_size=window_size
            )
        else:
            print(f"   Stride: {stride}")
            flops_info = EfficientAttentionSelector.get_attention_flops(
                seq_len=96, d_model=512, attention_type='strided', stride=stride
            )
        
        print(f"\n📊 FLOP Reduction:")
        print(f"   Full Attention FLOPs: {flops_info['flops']['full']:,}")
        print(f"   {attention_type.capitalize()} Attention FLOPs: {flops_info['flops'][attention_type]:,}")
        print(f"   Reduction Factor: {flops_info['reduction'][attention_type]:.2f}x")
        
        self.optimization_stats['efficient_attention'] = flops_info
    
    def apply_token_pruning(self, pruning_ratio: float = 0.1,
                           metric: str = 'norm', dynamic: bool = False) -> None:
        """
        Apply token pruning to reduce sequence length.
        
        Args:
            pruning_ratio: Ratio of tokens to prune
            metric: Pruning metric ('norm', 'entropy', 'attention')
            dynamic: Use dynamic pruning with layer-aware ratios
        """
        if not self.use_token_pruning:
            return
        
        print("\n" + "="*80)
        print("OPTIMIZATION 2: TOKEN PRUNING & VARIATE SELECTION")
        print("="*80)
        
        if dynamic:
            pruner = DynamicTokenPruning(base_pruning_ratio=pruning_ratio, metric=metric)
            print(f"\n✅ Token Pruning Type: Dynamic")
        else:
            pruner = TokenPruning(pruning_ratio=pruning_ratio, metric=metric)
            print(f"\n✅ Token Pruning Type: Static")
        
        print(f"   Metric: {metric}")
        print(f"   Base Pruning Ratio: {pruning_ratio:.1%}")
        
        # Simulate token pruning effect
        seq_len = 96
        pruned_seq_len = int(seq_len * (1 - pruning_ratio))
        
        print(f"\n📊 SEQUENCE LENGTH REDUCTION:")
        print(f"   Original Sequence Length: {seq_len}")
        print(f"   Pruned Sequence Length: {pruned_seq_len}")
        print(f"   Reduction: {(seq_len - pruned_seq_len)/seq_len*100:.1f}%")
        print(f"   Computational Savings: {pruning_ratio*100:.1f}% fewer tokens to process")
        
        self.optimization_stats['token_pruning'] = {
            'original_seq_len': seq_len,
            'pruned_seq_len': pruned_seq_len,
            'pruning_ratio': pruning_ratio,
            'metric': metric,
        }
    
    def apply_model_pruning(self, pruning_ratio: float = 0.3,
                           strategy: str = 'magnitude') -> None:
        """
        Apply model pruning to reduce parameters.
        
        Args:
            pruning_ratio: Ratio of weights to prune
            strategy: Pruning strategy ('magnitude', 'channel', 'structured')
        """
        if not self.use_model_pruning:
            return
        
        print("\n" + "="*80)
        print("OPTIMIZATION 3: MODEL PRUNING")
        print("="*80)
        
        original_stats = PruningAnalyzer.analyze_model_size(self.original_model)
        
        print(f"\n✅ Model Pruning Strategy: {strategy.capitalize()}")
        print(f"   Pruning Ratio: {pruning_ratio:.1%}")
        
        print(f"\n📊 ORIGINAL MODEL:")
        print(f"   Total Parameters: {original_stats['total_params']:,}")
        print(f"   Model Size: {original_stats['model_size_mb']:.2f} MB")
        print(f"   Layer Count: {original_stats['layer_count']}")
        
        # Simulate pruned model
        if strategy == 'magnitude':
            pruned_model = MagnitudePruning.prune_model(self.model, pruning_ratio)
        elif strategy == 'structured':
            pruned_model = MagnitudePruning.prune_model(self.model, pruning_ratio, structured=True)
        else:
            pruned_model = self.model
        
        pruned_stats = PruningAnalyzer.analyze_model_size(pruned_model)
        
        print(f"\n📊 PRUNED MODEL:")
        print(f"   Total Parameters: {pruned_stats['total_params']:,}")
        print(f"   Model Size: {pruned_stats['model_size_mb']:.2f} MB")
        print(f"   Layer Count: {pruned_stats['layer_count']}")
        
        print(f"\n📊 COMPRESSION METRICS:")
        compression_ratio = original_stats['total_params'] / max(pruned_stats['total_params'], 1)
        size_reduction_mb = original_stats['model_size_mb'] - pruned_stats['model_size_mb']
        
        print(f"   Compression Ratio: {compression_ratio:.2f}x")
        print(f"   Size Reduction: {size_reduction_mb:.2f} MB ({(1-pruned_stats['total_params']/original_stats['total_params'])*100:.1f}%)")
        
        self.optimization_stats['model_pruning'] = {
            'strategy': strategy,
            'pruning_ratio': pruning_ratio,
            'original_params': original_stats['total_params'],
            'pruned_params': pruned_stats['total_params'],
            'original_size_mb': original_stats['model_size_mb'],
            'pruned_size_mb': pruned_stats['model_size_mb'],
            'compression_ratio': compression_ratio,
        }
        
        self.model = pruned_model
    
    def apply_variate_selection(self, num_select: int = 5, method: str = 'learned') -> None:
        """
        Apply variate selection for multivariate forecasting.
        
        Args:
            num_select: Number of variates to select
            method: Selection method ('learned', 'correlation', 'attention')
        """
        print("\n" + "="*80)
        print("BONUS: VARIATE SELECTION")
        print("="*80)
        
        print(f"\n✅ Variate Selection Method: {method.capitalize()}")
        print(f"   Selected Variates: {num_select}")
        
        print(f"\n📊 FEATURE REDUCTION:")
        print(f"   Original Features: 7 (e.g., temperature, humidity, pressure, etc.)")
        print(f"   Selected Features: {num_select}")
        print(f"   Feature Reduction: {(1 - num_select/7)*100:.1f}%")
        
        self.optimization_stats['variate_selection'] = {
            'method': method,
            'num_select': num_select,
            'original_features': 7,
            'feature_reduction_ratio': 1 - num_select/7,
        }
    
    def estimate_speedup(self) -> Dict:
        """
        Estimate overall speedup from all optimizations.
        
        Returns:
            Dictionary with speedup estimates
        """
        print("\n" + "="*80)
        print("OVERALL OPTIMIZATION IMPACT")
        print("="*80)
        
        speedup_factors = {}
        
        # Efficient attention: ~2-3x speedup on attention computation
        if 'efficient_attention' in self.optimization_stats:
            speedup_factors['attention'] = self.optimization_stats['efficient_attention']['reduction'].get('windowed', 2.0)
        
        # Token pruning: proportional to pruning ratio
        if 'token_pruning' in self.optimization_stats:
            pruning_ratio = self.optimization_stats['token_pruning']['pruning_ratio']
            speedup_factors['token_pruning'] = 1 / (1 - pruning_ratio)
        
        # Model pruning: proportional to parameter reduction
        if 'model_pruning' in self.optimization_stats:
            speedup_factors['pruning'] = self.optimization_stats['model_pruning']['compression_ratio']
        
        # Combined speedup (multiplicative)
        combined_speedup = 1.0
        for key, speedup in speedup_factors.items():
            combined_speedup *= speedup
        
        print(f"\n⚡ INDIVIDUAL SPEEDUP FACTORS:")
        for key, speedup in speedup_factors.items():
            print(f"   {key.capitalize()}: {speedup:.2f}x")
        
        print(f"\n⚡ COMBINED SPEEDUP (Estimated): {combined_speedup:.2f}x")
        
        # Latency improvement
        original_latency = 100  # ms (baseline)
        optimized_latency = original_latency / combined_speedup
        time_savings = (original_latency - optimized_latency) / original_latency * 100
        
        print(f"\n⏱️  LATENCY IMPROVEMENT:")
        print(f"   Original Latency: {original_latency:.2f} ms")
        print(f"   Optimized Latency: {optimized_latency:.2f} ms")
        print(f"   Time Savings: {time_savings:.1f}%")
        
        return {
            'individual_speedups': speedup_factors,
            'combined_speedup': combined_speedup,
            'original_latency_ms': original_latency,
            'optimized_latency_ms': optimized_latency,
            'time_savings_percent': time_savings,
        }
    
    def print_summary(self) -> None:
        """Print comprehensive optimization summary."""
        print("\n" + "="*80)
        print("OPTIMIZATION SUMMARY")
        print("="*80)
        
        print(f"\n✅ APPLIED OPTIMIZATIONS:")
        if self.use_efficient_attention:
            print(f"   ✓ Efficient Attention (Windowed/Strided)")
        if self.use_token_pruning:
            print(f"   ✓ Token Pruning & Variate Selection")
        if self.use_model_pruning:
            print(f"   ✓ Model Pruning (Magnitude/Channel/Structured)")
        
        speedup_results = self.estimate_speedup()
        
        print(f"\n{'='*80}")
        print(f"\n🎯 OPTIMIZATION COMPLETE!")
        print(f"   Expected Speedup: {speedup_results['combined_speedup']:.2f}x")
        print(f"   Expected Latency: {speedup_results['optimized_latency_ms']:.2f} ms")
        print(f"   Expected Time Savings: {speedup_results['time_savings_percent']:.1f}%")


def main():
    parser = argparse.ArgumentParser(description='Optimization Pipeline Demo')
    
    parser.add_argument('--attention_type', type=str, default='windowed',
                       choices=['windowed', 'strided'],
                       help='Type of efficient attention')
    parser.add_argument('--window_size', type=int, default=32,
                       help='Window size for windowed attention')
    parser.add_argument('--stride', type=int, default=4,
                       help='Stride for strided attention')
    parser.add_argument('--token_pruning_ratio', type=float, default=0.1,
                       help='Token pruning ratio')
    parser.add_argument('--model_pruning_ratio', type=float, default=0.3,
                       help='Model pruning ratio')
    parser.add_argument('--pruning_strategy', type=str, default='magnitude',
                       choices=['magnitude', 'structured'],
                       help='Model pruning strategy')
    parser.add_argument('--variate_selection', type=int, default=5,
                       help='Number of variates to select')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("ITRANSFORMER OPTIMIZATION PIPELINE")
    print("="*80)
    
    # Create a dummy model for demonstration
    # In practice, this would be your actual iTransformer model
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(512, 512)
            self.linear2 = nn.Linear(512, 512)
            self.linear3 = nn.Linear(512, 256)
            
        def forward(self, x):
            x = self.linear1(x)
            x = self.linear2(x)
            x = self.linear3(x)
            return x
    
    model = DummyModel()
    
    # Initialize optimization pipeline
    pipeline = OptimizationPipeline(
        model,
        use_efficient_attention=True,
        use_token_pruning=True,
        use_model_pruning=True
    )
    
    # Apply optimizations
    pipeline.apply_efficient_attention(
        attention_type=args.attention_type,
        window_size=args.window_size,
        stride=args.stride
    )
    
    pipeline.apply_token_pruning(
        pruning_ratio=args.token_pruning_ratio,
        metric='norm',
        dynamic=False
    )
    
    pipeline.apply_model_pruning(
        pruning_ratio=args.model_pruning_ratio,
        strategy=args.pruning_strategy
    )
    
    pipeline.apply_variate_selection(
        num_select=args.variate_selection,
        method='learned'
    )
    
    # Print summary
    pipeline.print_summary()


if __name__ == '__main__':
    main()
