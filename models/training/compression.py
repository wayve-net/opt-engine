# compression.py
from torch.quantization import quantize_dynamic
import torch.nn as nn

def compress_model(model_path):
    """Compress trained model to under 100MB"""
    
    # Load trained model
    model = AutoModelForCausalLM.from_pretrained(model_path)
    
    # Dynamic quantization (INT8)
    quantized_model = quantize_dynamic(
        model, 
        {nn.Linear}, 
        dtype=torch.qint8
    )
    
    # Save compressed model
    quantized_model.save_pretrained("./compressed_network_model")
    
    # Verify size
    model_size = sum(p.numel() * p.element_size() for p in quantized_model.parameters())
    print(f"Compressed model size: {model_size / (1024**2):.1f}MB")
