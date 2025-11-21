"""
Model Quantization utilities for optimizing inference speed and memory usage.
Supports both dynamic and static quantization techniques.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import copy


class QuantizationConfig:
    """Configuration for model quantization"""
    
    def __init__(self, 
                 quantization_type: str = 'dynamic',
                 backend: str = 'fbgemm',
                 dtype: torch.dtype = torch.qint8):
        """
        Args:
            quantization_type: 'dynamic' or 'static'
            backend: 'fbgemm' (default, works on most platforms)
            dtype: torch.qint8 or torch.qint32
        """
        self.quantization_type = quantization_type
        self.backend = backend
        self.dtype = dtype


def apply_dynamic_quantization(model: nn.Module, 
                               qconfig: Optional[QuantizationConfig] = None) -> nn.Module:
    """
    Apply dynamic quantization to reduce model size and improve inference speed.
    
    Args:
        model: PyTorch model to quantize
        qconfig: Quantization configuration
        
    Returns:
        Quantized model
    """
    if qconfig is None:
        qconfig = QuantizationConfig()
    
    # Set the backend
    torch.backends.quantized.engine = qconfig.backend
    
    # Create a copy to avoid modifying the original
    quantized_model = copy.deepcopy(model)
    quantized_model = quantized_model.to('cpu')  # Quantization requires CPU
    
    # Apply dynamic quantization
    quantized_model = torch.quantization.quantize_dynamic(
        quantized_model,
        {nn.Linear},  # Quantize linear layers
        dtype=qconfig.dtype
    )
    
    return quantized_model


def apply_static_quantization(model: nn.Module,
                             calibration_data: torch.Tensor,
                             qconfig: Optional[QuantizationConfig] = None) -> nn.Module:
    """
    Apply static quantization with calibration data.
    
    Args:
        model: PyTorch model to quantize
        calibration_data: Calibration dataset for static quantization
        qconfig: Quantization configuration
        
    Returns:
        Quantized model
    """
    if qconfig is None:
        qconfig = QuantizationConfig()
    
    torch.backends.quantized.engine = qconfig.backend
    
    quantized_model = copy.deepcopy(model)
    quantized_model = quantized_model.to('cpu')
    
    # Set quantization config
    quantized_model.qconfig = torch.quantization.get_default_qconfig(qconfig.backend)
    torch.quantization.prepare(quantized_model, inplace=True)
    
    # Calibrate with sample data
    quantized_model.eval()
    with torch.no_grad():
        for data in calibration_data:
            if isinstance(data, (list, tuple)):
                quantized_model(*[d.cpu() if isinstance(d, torch.Tensor) else d for d in data])
            else:
                quantized_model(data.cpu())
    
    # Convert to quantized model
    torch.quantization.convert(quantized_model, inplace=True)
    
    return quantized_model


def get_model_size(model: nn.Module) -> Dict[str, float]:
    """
    Get model size in MB.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with size information
    """
    param_size = sum(p.numel() for p in model.parameters()) * 4 / (1024 * 1024)  # Assuming float32
    buffer_size = sum(b.numel() for b in model.buffers()) * 4 / (1024 * 1024)
    total_size = param_size + buffer_size
    
    return {
        'param_size_mb': param_size,
        'buffer_size_mb': buffer_size,
        'total_size_mb': total_size
    }


def compare_models(original_model: nn.Module, 
                   quantized_model: nn.Module) -> Dict[str, Any]:
    """
    Compare original and quantized models.
    
    Args:
        original_model: Original model
        quantized_model: Quantized model
        
    Returns:
        Comparison dictionary
    """
    original_size = get_model_size(original_model)
    quantized_size = get_model_size(quantized_model)
    
    compression_ratio = original_size['total_size_mb'] / quantized_size['total_size_mb']
    size_reduction = (1 - quantized_size['total_size_mb'] / original_size['total_size_mb']) * 100
    
    return {
        'original_size_mb': original_size['total_size_mb'],
        'quantized_size_mb': quantized_size['total_size_mb'],
        'compression_ratio': compression_ratio,
        'size_reduction_percent': size_reduction
    }


def list_quantizable_layers(model: nn.Module) -> Dict[str, list]:
    """
    List layers that can be quantized.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with quantizable layer info
    """
    quantizable = {'linear': [], 'conv': [], 'lstm': []}
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            quantizable['linear'].append(name)
        elif isinstance(module, (nn.Conv1d, nn.Conv2d)):
            quantizable['conv'].append(name)
        elif isinstance(module, (nn.LSTM, nn.GRU)):
            quantizable['lstm'].append(name)
    
    return quantizable
