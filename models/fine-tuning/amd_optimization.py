# amd_optimizations.py
import torch
import os

def setup_amd_optimizations():
    """Configure optimal settings for AMD CPU training"""
    
    # AMD-specific environment variables
    os.environ["OMP_NUM_THREADS"] = "4"  # Your CPU core count
    os.environ["MKL_NUM_THREADS"] = "4"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    
    # Use AMD's optimized BLAS if available
    torch.backends.mkl.enabled = True if torch.backends.mkl.is_available() else False
    
    # Enable CPU optimizations
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    
    print(f"PyTorch using {torch.get_num_threads()} threads")
    print(f"MKL available: {torch.backends.mkl.is_available()}")
