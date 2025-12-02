# memory_management.py
import gc
import torch

def optimize_cpu_memory():
    """Optimize memory usage for CPU training"""
    
    # Clear cache regularly
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
    
    # Garbage collection
    gc.collect()
    
    # Use gradient checkpointing
    model.gradient_checkpointing_enable()
    
    # Reduce precision where possible
    model.half()  # FP16 if supported
