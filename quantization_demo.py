"""
Demonstration script for Model Quantization and Performance Profiling.
Shows how to quantize the iTransformer model and compare performance metrics.
"""

import torch
import torch.nn as nn
import argparse
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from data_provider.data_factory import data_provider
from experiments.exp_quantized import Exp_Quantized_Long_Term_Forecast
from utils.profiling import PerformanceProfiler, BatchProfiler
from utils.quantization import get_model_size, list_quantizable_layers


def create_sample_model(args):
    """Create a sample iTransformer model"""
    from model import iTransformer
    model = iTransformer.Model(args)
    return model


def run_quantization_demo(args):
    """Run a complete quantization demonstration"""
    
    print("\n" + "="*80)
    print("ITRANSFORMER QUANTIZATION & PROFILING DEMONSTRATION")
    print("="*80)
    
    # Initialize model
    print("\n[1/5] Loading model...")
    model = create_sample_model(args)
    device = torch.device('cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Get data loaders
    print("[2/5] Loading data...")
    _, train_loader = data_provider(args, 'train')
    _, test_loader = data_provider(args, 'test')
    
    # Initialize quantization helper
    print("[3/5] Initializing quantization framework...")
    quant_helper = Exp_Quantized_Long_Term_Forecast()
    
    # Print model analysis
    print("\n" + "-"*80)
    print("ORIGINAL MODEL ANALYSIS")
    print("-"*80)
    quant_helper.print_model_analysis(model)
    
    # Apply dynamic quantization
    print("-"*80)
    print("QUANTIZATION PROCESS")
    print("-"*80)
    quantized_model = quant_helper.quantize_model_dynamic(model)
    
    # Move quantized model to CPU for inference (quantization requires CPU)
    quantized_model = quantized_model.to('cpu')
    
    # Profile inference performance
    print("-"*80)
    print("INFERENCE PROFILING")
    print("-"*80)
    
    # Get a test batch
    batch_x, batch_y, batch_x_mark, batch_y_mark = next(iter(test_loader))
    
    # Profile original model
    print("\n[4/5] Profiling original model inference...")
    original_metrics = quant_helper.profile_inference(
        model.to(device),
        batch_x, batch_x_mark,
        torch.zeros_like(batch_y[:, -args.pred_len:, :]),
        batch_y_mark,
        device=str(device),
        num_iterations=5
    )
    
    # Profile quantized model
    print("[5/5] Profiling quantized model inference...")
    quantized_metrics = quant_helper.profile_inference(
        quantized_model,
        batch_x, batch_x_mark,
        torch.zeros_like(batch_y[:, -args.pred_len:, :]),
        batch_y_mark,
        device='cpu',
        num_iterations=5
    )
    
    # Print comparison results
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON RESULTS")
    print("="*80)
    
    speedup = original_metrics['elapsed_time_per_iteration'] / quantized_metrics['elapsed_time_per_iteration']
    
    print(f"\nTiming Analysis:")
    print(f"  Original Model - Time/Iteration: {original_metrics['elapsed_time_per_iteration']*1000:.4f} ms")
    print(f"  Quantized Model - Time/Iteration: {quantized_metrics['elapsed_time_per_iteration']*1000:.4f} ms")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Time Savings: {(1-1/speedup)*100:.2f}%")
    
    print(f"\nMemory Analysis:")
    print(f"  Original Model - Memory Delta: {original_metrics['memory_delta_mb']:+.2f} MB")
    print(f"  Quantized Model - Memory Delta: {quantized_metrics['memory_delta_mb']:+.2f} MB")
    memory_saving = ((original_metrics['memory_delta_mb'] - quantized_metrics['memory_delta_mb']) / 
                     abs(original_metrics['memory_delta_mb']) * 100) if original_metrics['memory_delta_mb'] != 0 else 0
    print(f"  Memory Savings: {memory_saving:.2f}%")
    
    if 'gpu_memory_allocated_mb' in original_metrics:
        print(f"\nGPU Memory Analysis:")
        print(f"  Original - Allocated: {original_metrics.get('gpu_memory_allocated_mb', 0):.2f} MB")
        print(f"  Quantized - Allocated: {quantized_metrics.get('gpu_memory_allocated_mb', 0):.2f} MB")
    
    # Model size comparison
    print(f"\nModel Size Comparison:")
    orig_size = get_model_size(model)
    quant_size = get_model_size(quantized_model)
    print(f"  Original Model Size: {orig_size['total_size_mb']:.4f} MB")
    print(f"  Quantized Model Size: {quant_size['total_size_mb']:.4f} MB")
    compression_ratio = orig_size['total_size_mb'] / quant_size['total_size_mb']
    print(f"  Compression Ratio: {compression_ratio:.2f}x")
    print(f"  Size Reduction: {(1 - quant_size['total_size_mb']/orig_size['total_size_mb'])*100:.2f}%")
    
    # Save results
    print(f"\n" + "="*80)
    print("PROFILING METRICS SUMMARY")
    print("="*80)
    
    metrics_summary = {
        'original_model': {
            'time_per_iter_ms': original_metrics['elapsed_time_per_iteration'] * 1000,
            'memory_delta_mb': original_metrics['memory_delta_mb'],
            'size_mb': orig_size['total_size_mb']
        },
        'quantized_model': {
            'time_per_iter_ms': quantized_metrics['elapsed_time_per_iteration'] * 1000,
            'memory_delta_mb': quantized_metrics['memory_delta_mb'],
            'size_mb': quant_size['total_size_mb']
        },
        'improvements': {
            'speedup_factor': speedup,
            'time_reduction_percent': (1-1/speedup)*100,
            'memory_savings_percent': memory_saving,
            'size_reduction_percent': (1 - quant_size['total_size_mb']/orig_size['total_size_mb'])*100,
            'compression_ratio': compression_ratio
        }
    }
    
    print("\nMetrics saved. Ready for deployment!")
    print("="*80 + "\n")
    
    return metrics_summary


def main():
    parser = argparse.ArgumentParser(description='iTransformer Quantization Demo')
    
    # Data
    parser.add_argument('--data', type=str, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/ETT-small/', help='root path of data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M', help='forecasting task')
    parser.add_argument('--target', type=str, default='OT', help='target feature')
    parser.add_argument('--freq', type=str, default='h', help='frequency for time features')
    
    # Model
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='label length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction length')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--num_workers', type=int, default=0, help='num workers')
    parser.add_argument('--use_gpu', type=bool, default=False, help='use GPU')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index')
    parser.add_argument('--use_norm', type=int, default=True, help='use normalization')
    parser.add_argument('--use_amp', action='store_true', help='use AMP')
    parser.add_argument('--output_attention', action='store_true', help='output attention')
    parser.add_argument('--factor', type=int, default=1, help='attention factor')
    parser.add_argument('--distil', action='store_false', help='distilling', default=True)
    parser.add_argument('--class_strategy', type=str, default='projection', help='class strategy')
    
    args = parser.parse_args()
    
    # Set fbgemm backend by default
    torch.backends.quantized.engine = 'fbgemm'
    
    # Run demo
    metrics = run_quantization_demo(args)
    
    return metrics


if __name__ == '__main__':
    main()
