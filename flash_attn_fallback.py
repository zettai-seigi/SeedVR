"""
Flash attention fallback for macOS compatibility
This module provides fallback implementations when flash_attn is not available
"""

import torch
import torch.nn.functional as F
import warnings

# Flag to indicate flash_attn is not available
FLASH_ATTN_AVAILABLE = False

def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, 
                          dropout_p=0.0, softmax_scale=None, causal=False, window_size=(-1, -1), 
                          alibi_slopes=None, deterministic=False, return_attn_probs=False):
    """
    Fallback implementation of flash_attn_varlen_func using native PyTorch
    This is a simplified version that may not handle all edge cases
    """
    warnings.warn("Using fallback flash_attn_varlen_func - performance may be reduced")
    
    # Convert variable length format to regular batched format
    # This is a simplified conversion - real implementation would be more complex
    batch_size = len(cu_seqlens_q) - 1
    
    # For now, raise an error to indicate this specific pattern needs implementation
    raise NotImplementedError(
        "flash_attn_varlen_func fallback not fully implemented. "
        "Consider modifying the model to use regular attention patterns."
    )

def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, 
                   window_size=(-1, -1), alibi_slopes=None, deterministic=False, 
                   return_attn_probs=False):
    """
    Fallback implementation of flash_attn_func using native PyTorch attention
    """
    # Use PyTorch's native scaled_dot_product_attention
    return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=causal)

# Create a mock flash_attn module
class FlashAttnModule:
    def __getattr__(self, name):
        if name == "flash_attn_varlen_func":
            return flash_attn_varlen_func
        elif name == "flash_attn_func":
            return flash_attn_func
        else:
            raise AttributeError(f"module 'flash_attn' has no attribute '{name}'")