"""
Enhanced experiment class with quantization and profiling support.
Extends the base experiment class with optimization capabilities.
"""

import torch
import torch.nn as nn
from utils.quantization import (
    apply_dynamic_quantization,
    apply_static_quantization,
    get_model_size,
    compare_models,
    list_quantizable_layers,
    QuantizationConfig
)
from utils.profiling import PerformanceProfiler, BatchProfiler, compare_performance
import os
import numpy as np


class Exp_Quantized_Long_Term_Forecast:
    """
    Extended experiment class with quantization and profiling.
    Can be mixed with existing experiment classes.
    """
    
    def __init__(self):
        self.original_model = None
        self.quantized_model = None
        self.profiler = None
        self.batch_profiler = BatchProfiler()
        self.quantization_config = QuantizationConfig()
        
    def enable_profiling(self, name: str = "Model Inference"):
        """Enable performance profiling"""
        self.profiler = PerformanceProfiler(name=name)
        
    def quantize_model_dynamic(self, model: nn.Module) -> nn.Module:
        """
        Apply dynamic quantization to model.
        
        Args:
            model: Model to quantize
            
        Returns:
            Quantized model
        """
        print("\n" + "="*60)
        print("APPLYING DYNAMIC QUANTIZATION")
        print("="*60)
        
        self.original_model = model
        
        # Get original model size
        original_size = get_model_size(model)
        print(f"\nOriginal Model Size: {original_size['total_size_mb']:.4f} MB")
        
        # Quantize
        self.quantized_model = apply_dynamic_quantization(
            model, 
            self.quantization_config
        )
        
        # Get quantized model size
        quantized_size = get_model_size(self.quantized_model)
        print(f"Quantized Model Size: {quantized_size['total_size_mb']:.4f} MB")
        
        # Compare
        comparison = compare_models(model, self.quantized_model)
        print(f"\nCompression Ratio: {comparison['compression_ratio']:.2f}x")
        print(f"Size Reduction: {comparison['size_reduction_percent']:.2f}%")
        print("="*60 + "\n")
        
        return self.quantized_model
    
    def quantize_model_static(self, model: nn.Module, 
                             calibration_loader) -> nn.Module:
        """
        Apply static quantization with calibration data.
        
        Args:
            model: Model to quantize
            calibration_loader: DataLoader for calibration
            
        Returns:
            Quantized model
        """
        print("\n" + "="*60)
        print("APPLYING STATIC QUANTIZATION")
        print("="*60)
        
        self.original_model = model
        
        # Get original model size
        original_size = get_model_size(model)
        print(f"\nOriginal Model Size: {original_size['total_size_mb']:.4f} MB")
        print("Calibrating with sample data...")
        
        # Prepare calibration data
        calibration_data = []
        for batch_idx, batch in enumerate(calibration_loader):
            if batch_idx >= 10:  # Use first 10 batches
                break
            if isinstance(batch, (list, tuple)):
                calibration_data.append(batch)
            else:
                calibration_data.append((batch,))
        
        # Quantize
        self.quantized_model = apply_static_quantization(
            model,
            calibration_data,
            self.quantization_config
        )
        
        # Get quantized model size
        quantized_size = get_model_size(self.quantized_model)
        print(f"Quantized Model Size: {quantized_size['total_size_mb']:.4f} MB")
        
        # Compare
        comparison = compare_models(model, self.quantized_model)
        print(f"\nCompression Ratio: {comparison['compression_ratio']:.2f}x")
        print(f"Size Reduction: {comparison['size_reduction_percent']:.2f}%")
        print("="*60 + "\n")
        
        return self.quantized_model
    
    def profile_inference(self, model: nn.Module,
                         batch_x: torch.Tensor,
                         batch_x_mark: torch.Tensor,
                         dec_inp: torch.Tensor,
                         batch_y_mark: torch.Tensor,
                         device: str = 'cpu',
                         num_iterations: int = 1) -> dict:
        """
        Profile model inference.
        
        Args:
            model: Model to profile
            batch_x: Input tensor
            batch_x_mark: Input marks
            dec_inp: Decoder input
            batch_y_mark: Output marks
            device: Device to run on
            num_iterations: Number of iterations for profiling
            
        Returns:
            Profiling metrics
        """
        model = model.to(device).eval()
        batch_x = batch_x.to(device).float()
        batch_x_mark = batch_x_mark.to(device).float() if batch_x_mark is not None else None
        dec_inp = dec_inp.to(device).float()
        batch_y_mark = batch_y_mark.to(device).float() if batch_y_mark is not None else None
        
        profiler = PerformanceProfiler(name="Model Inference")
        
        with torch.no_grad():
            # Warmup
            for _ in range(2):
                try:
                    _ = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                except:
                    _ = model(batch_x)
            
            # Profile
            profiler.start()
            for _ in range(num_iterations):
                try:
                    _ = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                except:
                    _ = model(batch_x)
            metrics = profiler.stop()
        
        # Normalize to per-iteration
        metrics['elapsed_time_per_iteration'] = metrics['elapsed_time_seconds'] / num_iterations
        
        return metrics
    
    def compare_quantized_vs_original(self, 
                                     test_loader,
                                     device: str = 'cpu') -> dict:
        """
        Compare inference performance: original vs quantized.
        
        Args:
            test_loader: Test DataLoader
            device: Device to run on
            
        Returns:
            Comparison metrics
        """
        if self.original_model is None or self.quantized_model is None:
            print("Error: Both original and quantized models must be set.")
            return {}
        
        print("\n" + "="*60)
        print("COMPARING ORIGINAL vs QUANTIZED MODEL")
        print("="*60)
        
        # Get a batch for testing
        batch_x, batch_y, batch_x_mark, batch_y_mark = next(iter(test_loader))
        
        original_metrics = self.profile_inference(
            self.original_model,
            batch_x, batch_x_mark,
            torch.zeros_like(batch_y[:, -96:, :]),  # Assuming pred_len=96
            batch_y_mark,
            device=device,
            num_iterations=10
        )
        
        quantized_metrics = self.profile_inference(
            self.quantized_model,
            batch_x, batch_x_mark,
            torch.zeros_like(batch_y[:, -96:, :]),
            batch_y_mark,
            device=device,
            num_iterations=10
        )
        
        speedup = original_metrics['elapsed_time_per_iteration'] / quantized_metrics['elapsed_time_per_iteration']
        
        print(f"\nOriginal Model:")
        print(f"  - Time per iteration: {original_metrics['elapsed_time_per_iteration']:.6f}s")
        print(f"  - Memory delta: {original_metrics['memory_delta_mb']:+.2f} MB")
        
        print(f"\nQuantized Model:")
        print(f"  - Time per iteration: {quantized_metrics['elapsed_time_per_iteration']:.6f}s")
        print(f"  - Memory delta: {quantized_metrics['memory_delta_mb']:+.2f} MB")
        
        print(f"\nSpeedup: {speedup:.2f}x")
        print(f"Memory Improvement: {((original_metrics['memory_delta_mb'] - quantized_metrics['memory_delta_mb']) / abs(original_metrics['memory_delta_mb']) * 100) if original_metrics['memory_delta_mb'] != 0 else 0:.2f}%")
        print("="*60 + "\n")
        
        return {
            'original_metrics': original_metrics,
            'quantized_metrics': quantized_metrics,
            'speedup': speedup
        }
    
    def print_model_analysis(self, model: nn.Module):
        """Print detailed model analysis"""
        print("\n" + "="*60)
        print("MODEL ANALYSIS")
        print("="*60)
        
        # Model size
        size_info = get_model_size(model)
        print(f"\nModel Size:")
        print(f"  - Parameters: {size_info['param_size_mb']:.4f} MB")
        print(f"  - Buffers: {size_info['buffer_size_mb']:.4f} MB")
        print(f"  - Total: {size_info['total_size_mb']:.4f} MB")
        
        # Quantizable layers
        quantizable = list_quantizable_layers(model)
        print(f"\nQuantizable Layers:")
        print(f"  - Linear: {len(quantizable['linear'])}")
        print(f"  - Conv: {len(quantizable['conv'])}")
        print(f"  - LSTM/GRU: {len(quantizable['lstm'])}")
        
        # Total parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\nTotal Parameters: {total_params:,}")
        
        print("="*60 + "\n")
    
    def save_quantized_model(self, filepath: str):
        """Save quantized model"""
        if self.quantized_model is None:
            print("Error: No quantized model to save.")
            return
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(self.quantized_model, filepath)
        print(f"Quantized model saved to: {filepath}")
    
    def load_quantized_model(self, filepath: str, device: str = 'cpu') -> nn.Module:
        """Load quantized model"""
        self.quantized_model = torch.load(filepath, map_location=device)
        print(f"Quantized model loaded from: {filepath}")
        return self.quantized_model
