"""
Advanced Usage Examples for Model Quantization and Profiling
Shows practical implementations for production deployments
"""

# ============================================================================
# Example 1: Basic Dynamic Quantization
# ============================================================================

def example_1_basic_quantization():
    """Simple dynamic quantization in 5 lines"""
    from experiments.exp_quantized import Exp_Quantized_Long_Term_Forecast
    from model import iTransformer
    
    quant = Exp_Quantized_Long_Term_Forecast()
    model = iTransformer.Model(args)
    quantized_model = quant.quantize_model_dynamic(model)
    quant.save_quantized_model('./checkpoints/quantized_model.pth')


# ============================================================================
# Example 2: Profiling Inference Loop
# ============================================================================

def example_2_profile_inference():
    """Profile model inference with detailed metrics"""
    from utils.profiling import PerformanceProfiler
    import torch
    
    profiler = PerformanceProfiler(name="ETTh1 Inference")
    model = load_model()
    
    profiler.start()
    with torch.no_grad():
        for batch in test_loader:
            output = model(batch)
    metrics = profiler.stop()
    
    profiler.print_report()
    # Output includes: time, memory delta, GPU memory, timestamp


# ============================================================================
# Example 3: Compare Original vs Quantized
# ============================================================================

def example_3_compare_models():
    """Direct performance comparison"""
    from utils.profiling import compare_performance
    
    comparison = compare_performance(
        original_model,
        quantized_model,
        sample_input,
        device='cpu'
    )
    
    print(f"Speedup: {comparison['speedup_factor']:.2f}x")
    print(f"Memory improvement: {comparison['memory_improvement_percent']:.2f}%")
    print(f"Original time: {comparison['model1_time_seconds']*1000:.2f}ms")
    print(f"Quantized time: {comparison['model2_time_seconds']*1000:.2f}ms")


# ============================================================================
# Example 4: Static Quantization with Calibration
# ============================================================================

def example_4_static_quantization():
    """Use calibration data for better quantization"""
    from experiments.exp_quantized import Exp_Quantized_Long_Term_Forecast
    
    quant = Exp_Quantized_Long_Term_Forecast()
    model = load_model()
    
    # Use first 50 batches as calibration data
    quantized = quant.quantize_model_static(model, train_loader)
    
    # Should show better compression than dynamic
    quant.print_model_analysis(quantized)


# ============================================================================
# Example 5: Batch Profiling Multiple Experiments
# ============================================================================

def example_5_batch_profiling():
    """Profile multiple models at once"""
    from utils.profiling import BatchProfiler, PerformanceProfiler
    
    batch_profiler = BatchProfiler()
    
    # Profile model 1
    profiler1 = PerformanceProfiler("Model v1")
    profiler1.start()
    _ = model_v1(x)
    profiler1.stop()
    batch_profiler.add_profiler(profiler1)
    
    # Profile model 2
    profiler2 = PerformanceProfiler("Model v2")
    profiler2.start()
    _ = model_v2(x)
    profiler2.stop()
    batch_profiler.add_profiler(profiler2)
    
    # Summary
    batch_profiler.print_summary()


# ============================================================================
# Example 6: Profiling Function Decorator Style
# ============================================================================

def example_6_function_profiling():
    """Profile a function without explicit start/stop"""
    from utils.profiling import profile_function
    
    def inference_step(model, data):
        with torch.no_grad():
            return model(data)
    
    result, metrics = profile_function(inference_step, model, batch)
    
    print(f"Time: {metrics['elapsed_time_seconds']*1000:.2f}ms")
    print(f"Memory: {metrics['memory_delta_mb']:.2f}MB")


# ============================================================================
# Example 7: Custom Quantization Configuration
# ============================================================================

def example_7_custom_config():
    """Use custom quantization settings"""
    from utils.quantization import QuantizationConfig, apply_dynamic_quantization
    
    # Create custom config (fbgemm is default, works on all platforms)
    config = QuantizationConfig(
        quantization_type='dynamic',
        backend='fbgemm',  # or 'qnnpack' on Linux/Mac
        dtype=torch.qint8   # 8-bit integer quantization
    )
    
    quantized = apply_dynamic_quantization(model, config)


# ============================================================================
# Example 8: Integration with Training Loop
# ============================================================================

def example_8_training_integration():
    """Add quantization to your existing training pipeline"""
    from experiments.exp_long_term_forecasting import Exp_Long_Term_Forecast
    from experiments.exp_quantized import Exp_Quantized_Long_Term_Forecast
    
    class Exp_With_Quantization(Exp_Long_Term_Forecast):
        def __init__(self, args):
            super().__init__(args)
            self.quant = Exp_Quantized_Long_Term_Forecast()
            
        def train(self, setting):
            # ... existing training code ...
            print("Training complete. Applying quantization...")
            
            # Quantize after training
            self.model = self.quant.quantize_model_dynamic(self.model)
            print(f"Model compressed: {self.quant.quantization_config}")
            
            # Run test with quantized model
            self.test(setting)
            
        def test(self, setting, test=0):
            # Profile inference
            metrics = self.quant.profile_inference(
                self.model,
                batch_x, batch_x_mark, dec_inp, batch_y_mark,
                device=str(self.device),
                num_iterations=5
            )
            
            # Print profiling report
            profiler = self.quant.profiler
            if profiler:
                profiler.print_report()
            
            # Continue with regular testing
            super().test(setting, test)


# ============================================================================
# Example 9: Memory Monitoring During Inference
# ============================================================================

def example_9_memory_monitoring():
    """Monitor memory usage during inference"""
    from utils.profiling import MemoryMonitor
    
    monitor = MemoryMonitor(interval=0.1)
    
    with torch.no_grad():
        for batch in dataloader:
            output = model(batch)
            mem = monitor.get_current_memory()
            print(f"Current RSS Memory: {mem['rss_mb']:.2f} MB")
    
    monitor.print_memory_stats()


# ============================================================================
# Example 10: Production Deployment Pattern
# ============================================================================

def example_10_production_deployment():
    """Complete production-ready pattern"""
    from experiments.exp_quantized import Exp_Quantized_Long_Term_Forecast
    import json
    
    # Step 1: Train and quantize
    print("Step 1: Loading model...")
    model = load_trained_model()
    
    # Step 2: Apply quantization
    print("Step 2: Quantizing model...")
    quant = Exp_Quantized_Long_Term_Forecast()
    quantized_model = quant.quantize_model_dynamic(model)
    
    # Step 3: Save quantized model
    print("Step 3: Saving quantized model...")
    quant.save_quantized_model('./models/quantized_itransformer.pth')
    
    # Step 4: Profile for documentation
    print("Step 4: Profiling performance...")
    metrics = quant.profile_inference(
        quantized_model,
        test_batch_x, test_batch_x_mark,
        test_dec_inp, test_batch_y_mark,
        device='cpu',
        num_iterations=100
    )
    
    # Step 5: Save metrics
    performance_report = {
        'model': 'iTransformer_quantized',
        'version': '1.0',
        'metrics': metrics,
        'timestamp': datetime.now().isoformat(),
    }
    
    with open('./models/performance_report.json', 'w') as f:
        json.dump(performance_report, f, indent=2)
    
    # Step 6: Print summary
    print("\n" + "="*60)
    print("DEPLOYMENT READY")
    print("="*60)
    print(f"Quantized model: ./models/quantized_itransformer.pth")
    print(f"Performance report: ./models/performance_report.json")
    print(f"Inference time: {metrics['elapsed_time_per_iteration']*1000:.2f}ms")
    print(f"Model size: {get_model_size(quantized_model)['total_size_mb']:.2f}MB")
    print("="*60)


# ============================================================================
# Example 11: Comparing Multiple Quantization Strategies
# ============================================================================

def example_11_compare_strategies():
    """Compare dynamic vs static quantization"""
    from experiments.exp_quantized import Exp_Quantized_Long_Term_Forecast
    
    quant = Exp_Quantized_Long_Term_Forecast()
    model = load_model()
    
    print("Strategy 1: Dynamic Quantization")
    dynamic_model = quant.quantize_model_dynamic(model)
    dynamic_metrics = quant.profile_inference(dynamic_model, *test_data)
    
    print("\nStrategy 2: Static Quantization")
    static_model = quant.quantize_model_static(model, train_loader)
    static_metrics = quant.profile_inference(static_model, *test_data)
    
    print("\nComparison:")
    print(f"Dynamic - Time: {dynamic_metrics['elapsed_time_per_iteration']*1000:.2f}ms")
    print(f"Static - Time: {static_metrics['elapsed_time_per_iteration']*1000:.2f}ms")
    print(f"Speedup (Static vs Dynamic): {dynamic_metrics['elapsed_time_per_iteration']/static_metrics['elapsed_time_per_iteration']:.2f}x")


# ============================================================================
# Example 12: GPU vs CPU Inference Comparison
# ============================================================================

def example_12_gpu_cpu_comparison():
    """Compare GPU and CPU inference performance"""
    from utils.profiling import compare_performance
    
    # CPU inference
    model_cpu = model.to('cpu')
    input_cpu = input_data.to('cpu')
    
    # GPU inference  
    model_gpu = model.to('cuda')
    input_gpu = input_data.to('cuda')
    
    # Create a temporary wrapper for fair comparison
    cpu_comparison = compare_performance(
        dummy_model,
        model_cpu,
        input_cpu,
        device='cpu'
    )
    
    gpu_comparison = compare_performance(
        dummy_model,
        model_gpu,
        input_gpu,
        device='cuda'
    )
    
    print(f"CPU Time: {cpu_comparison['model2_time_seconds']*1000:.2f}ms")
    print(f"GPU Time: {gpu_comparison['model2_time_seconds']*1000:.2f}ms")


# ============================================================================
# Example 13: Model Size Analysis
# ============================================================================

def example_13_model_analysis():
    """Detailed model size and structure analysis"""
    from utils.quantization import (
        get_model_size,
        list_quantizable_layers,
        compare_models
    )
    
    model = load_model()
    quantized = quantize_model(model)
    
    # Analyze original
    print("Original Model:")
    print_model_size(get_model_size(model))
    print_quantizable_layers(list_quantizable_layers(model))
    
    # Analyze quantized
    print("\nQuantized Model:")
    print_model_size(get_model_size(quantized))
    
    # Compare
    comparison = compare_models(model, quantized)
    print(f"\nCompression: {comparison['compression_ratio']:.2f}x")
    print(f"Size reduction: {comparison['size_reduction_percent']:.2f}%")


# ============================================================================
# Helper function for printing
# ============================================================================

def print_model_size(size_dict):
    """Pretty print model size info"""
    print(f"  Parameters: {size_dict['param_size_mb']:.4f} MB")
    print(f"  Buffers: {size_dict['buffer_size_mb']:.4f} MB")
    print(f"  Total: {size_dict['total_size_mb']:.4f} MB")


def print_quantizable_layers(layers_dict):
    """Pretty print quantizable layers"""
    print(f"  Linear layers: {len(layers_dict['linear'])}")
    print(f"  Conv layers: {len(layers_dict['conv'])}")
    print(f"  LSTM/GRU: {len(layers_dict['lstm'])}")
    print(f"  Total quantizable: {sum(len(v) for v in layers_dict.values())}")


# ============================================================================
# Running the examples
# ============================================================================

if __name__ == '__main__':
    print("Advanced Usage Examples")
    print("=====================")
    print("\nTo use these examples, copy the code and adapt to your needs.")
    print("Each example is self-contained and demonstrates a specific pattern.")
    print("\nKey modules:")
    print("  - experiments/exp_quantized.py : Main quantization helper")
    print("  - utils/quantization.py : Core quantization functions")
    print("  - utils/profiling.py : Performance profiling tools")
    print("\nDocumentation: See QUANTIZATION_GUIDE.md")
