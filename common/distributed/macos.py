# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# // macOS-compatible distributed functions
# // Licensed under the Apache License, Version 2.0

"""
macOS-compatible distributed functions for single-device operation
"""

import os
import torch
from datetime import timedelta


def get_global_rank() -> int:
    """Get the global rank (always 0 for single device)"""
    return 0


def get_local_rank() -> int:
    """Get the local rank (always 0 for single device)"""
    return 0


def get_world_size() -> int:
    """Get the world size (always 1 for single device)"""
    return 1


def get_device():
    """Get the appropriate device for macOS"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def init_torch(cudnn_benchmark=False, timeout=None):
    """Initialize PyTorch for macOS"""
    if torch.backends.mps.is_available():
        print("✅ Using MPS (Metal Performance Shaders)")
        # MPS-specific optimizations
        torch.backends.mps.allow_tf32 = True
    else:
        print("⚠️  MPS not available, using CPU")
        
    # Set benchmark for CPU/MPS (not same as CUDA cudnn_benchmark)
    torch.backends.cudnn.benchmark = False  # Not applicable but keep for compatibility
    
    print(f"🔧 PyTorch initialized with device: {get_device()}")


def init_distributed(backend="nccl", timeout=timedelta(seconds=3600)):
    """No-op for single device"""
    print("📱 Single device mode - distributed training disabled")
    return


def barrier():
    """No-op barrier for single device"""
    pass


def broadcast(tensor, src=0):
    """No-op broadcast for single device"""
    return tensor


def all_reduce(tensor, op=None):
    """No-op all_reduce for single device"""
    return tensor


def all_gather(tensor):
    """No-op all_gather for single device"""
    return [tensor]


def reduce_scatter(tensor, tensor_list, op=None):
    """No-op reduce_scatter for single device"""
    return tensor


def is_distributed():
    """Always False for single device"""
    return False


def cleanup():
    """No-op cleanup for single device"""
    pass