"""
Performance profiling utilities to track time and memory usage.
Provides detailed metrics for model inference and training.
"""

import time
import psutil
import torch
import os
from typing import Dict, Any, Optional, Callable
from datetime import datetime
import json


class PerformanceProfiler:
    """Track performance metrics including memory and execution time"""
    
    def __init__(self, name: str = "Model"):
        self.name = name
        self.metrics = {}
        self.start_time = None
        self.start_memory = None
        self.process = psutil.Process(os.getpid())
        
    def start(self):
        """Start profiling"""
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        self.start_time = time.time()
        self.start_memory = self._get_memory_usage()
        
    def stop(self) -> Dict[str, Any]:
        """Stop profiling and return metrics"""
        if self.start_time is None:
            raise RuntimeError("Profiler not started. Call start() first.")
        
        elapsed_time = time.time() - self.start_time
        end_memory = self._get_memory_usage()
        
        metrics = {
            'name': self.name,
            'elapsed_time_seconds': elapsed_time,
            'start_memory_mb': self.start_memory['rss'],
            'end_memory_mb': end_memory['rss'],
            'memory_delta_mb': end_memory['rss'] - self.start_memory['rss'],
            'peak_memory_mb': end_memory['peak'] if 'peak' in end_memory else end_memory['rss'],
            'timestamp': datetime.now().isoformat()
        }
        
        if torch.cuda.is_available():
            metrics['gpu_memory_allocated_mb'] = torch.cuda.memory_allocated() / (1024**2)
            metrics['gpu_memory_reserved_mb'] = torch.cuda.memory_reserved() / (1024**2)
            metrics['gpu_peak_memory_mb'] = torch.cuda.max_memory_allocated() / (1024**2)
        
        self.metrics = metrics
        return metrics
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage in MB"""
        mem_info = self.process.memory_info()
        memory = {
            'rss': mem_info.rss / (1024 * 1024),  # Resident Set Size
            'vms': mem_info.vms / (1024 * 1024),  # Virtual Memory Size
        }
        
        if torch.cuda.is_available():
            memory['peak'] = torch.cuda.max_memory_allocated() / (1024**2)
        
        return memory
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get collected metrics"""
        return self.metrics
    
    def print_report(self):
        """Print formatted performance report"""
        if not self.metrics:
            print("No metrics collected. Run profiler first.")
            return
        
        print("\n" + "="*60)
        print(f"PERFORMANCE REPORT: {self.metrics['name']}")
        print("="*60)
        print(f"Execution Time: {self.metrics['elapsed_time_seconds']:.4f} seconds")
        print(f"Memory Usage:")
        print(f"  - Start: {self.metrics['start_memory_mb']:.2f} MB")
        print(f"  - End: {self.metrics['end_memory_mb']:.2f} MB")
        print(f"  - Delta: {self.metrics['memory_delta_mb']:+.2f} MB")
        
        if 'gpu_memory_allocated_mb' in self.metrics:
            print(f"GPU Memory:")
            print(f"  - Allocated: {self.metrics['gpu_memory_allocated_mb']:.2f} MB")
            print(f"  - Reserved: {self.metrics['gpu_memory_reserved_mb']:.2f} MB")
            print(f"  - Peak: {self.metrics['gpu_peak_memory_mb']:.2f} MB")
        
        print(f"Timestamp: {self.metrics['timestamp']}")
        print("="*60 + "\n")


class BatchProfiler:
    """Track metrics across multiple operations"""
    
    def __init__(self):
        self.profilers = []
        self.summaries = []
        
    def add_profiler(self, profiler: PerformanceProfiler):
        """Add a profiler to the batch"""
        self.profilers.append(profiler)
        
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all profilers"""
        if not self.profilers:
            return {}
        
        metrics_list = [p.get_metrics() for p in self.profilers]
        
        total_time = sum(m.get('elapsed_time_seconds', 0) for m in metrics_list)
        avg_time = total_time / len(metrics_list) if metrics_list else 0
        
        max_memory = max((m.get('end_memory_mb', 0) for m in metrics_list), default=0)
        avg_memory = sum(m.get('memory_delta_mb', 0) for m in metrics_list) / len(metrics_list) if metrics_list else 0
        
        return {
            'num_operations': len(metrics_list),
            'total_time_seconds': total_time,
            'avg_time_seconds': avg_time,
            'max_memory_mb': max_memory,
            'avg_memory_delta_mb': avg_memory,
            'details': metrics_list
        }
    
    def print_summary(self):
        """Print batch summary"""
        summary = self.get_summary()
        
        print("\n" + "="*60)
        print("BATCH PROFILING SUMMARY")
        print("="*60)
        print(f"Total Operations: {summary['num_operations']}")
        print(f"Total Time: {summary['total_time_seconds']:.4f} seconds")
        print(f"Average Time per Operation: {summary['avg_time_seconds']:.4f} seconds")
        print(f"Max Memory Usage: {summary['max_memory_mb']:.2f} MB")
        print(f"Average Memory Delta: {summary['avg_memory_delta_mb']:+.2f} MB")
        print("="*60 + "\n")


def profile_function(func: Callable, *args, **kwargs) -> tuple:
    """
    Profile a function execution.
    
    Args:
        func: Function to profile
        *args: Function arguments
        **kwargs: Function keyword arguments
        
    Returns:
        Tuple of (result, metrics)
    """
    profiler = PerformanceProfiler(name=func.__name__)
    profiler.start()
    
    try:
        result = func(*args, **kwargs)
    finally:
        metrics = profiler.stop()
    
    return result, metrics


def compare_performance(model1: torch.nn.Module,
                       model2: torch.nn.Module,
                       input_data: torch.Tensor,
                       device: str = 'cpu') -> Dict[str, Any]:
    """
    Compare performance of two models.
    
    Args:
        model1: First model
        model2: Second model
        input_data: Input tensor for inference
        device: Device to run on ('cpu' or 'cuda')
        
    Returns:
        Comparison dictionary
    """
    model1 = model1.to(device).eval()
    model2 = model2.to(device).eval()
    input_data = input_data.to(device)
    
    # Profile model1
    profiler1 = PerformanceProfiler(name="Model 1")
    profiler1.start()
    with torch.no_grad():
        output1 = model1(input_data)
    metrics1 = profiler1.stop()
    
    # Profile model2
    profiler2 = PerformanceProfiler(name="Model 2")
    profiler2.start()
    with torch.no_grad():
        output2 = model2(input_data)
    metrics2 = profiler2.stop()
    
    speedup = metrics1['elapsed_time_seconds'] / metrics2['elapsed_time_seconds']
    memory_improvement = (metrics1['memory_delta_mb'] - metrics2['memory_delta_mb']) / metrics1['memory_delta_mb'] * 100 if metrics1['memory_delta_mb'] != 0 else 0
    
    return {
        'model1_time_seconds': metrics1['elapsed_time_seconds'],
        'model2_time_seconds': metrics2['elapsed_time_seconds'],
        'speedup_factor': speedup,
        'model1_memory_delta_mb': metrics1['memory_delta_mb'],
        'model2_memory_delta_mb': metrics2['memory_delta_mb'],
        'memory_improvement_percent': memory_improvement,
        'model1_metrics': metrics1,
        'model2_metrics': metrics2
    }


class MemoryMonitor:
    """Monitor memory usage over time"""
    
    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.memory_log = []
        self.monitoring = False
        
    def get_current_memory(self) -> Dict[str, float]:
        """Get current memory usage"""
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        
        memory = {
            'timestamp': time.time(),
            'rss_mb': mem_info.rss / (1024 * 1024),
            'vms_mb': mem_info.vms / (1024 * 1024),
        }
        
        if torch.cuda.is_available():
            memory['gpu_allocated_mb'] = torch.cuda.memory_allocated() / (1024**2)
            memory['gpu_reserved_mb'] = torch.cuda.memory_reserved() / (1024**2)
        
        return memory
    
    def get_memory_log(self) -> list:
        """Get memory log"""
        return self.memory_log
    
    def print_memory_stats(self):
        """Print memory statistics"""
        if not self.memory_log:
            print("No memory data collected.")
            return
        
        rss_values = [m['rss_mb'] for m in self.memory_log]
        
        print("\n" + "="*60)
        print("MEMORY STATISTICS")
        print("="*60)
        print(f"Peak Memory: {max(rss_values):.2f} MB")
        print(f"Min Memory: {min(rss_values):.2f} MB")
        print(f"Average Memory: {sum(rss_values)/len(rss_values):.2f} MB")
        print("="*60 + "\n")
