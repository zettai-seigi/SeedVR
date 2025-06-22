# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# // macOS-compatible version without flash_attn dependency
# // Licensed under the Apache License, Version 2.0

import torch
import torch.nn.functional as F
import warnings

# macOS compatibility: Replace flash_attn with native PyTorch attention
try:
    from flash_attn import flash_attn_varlen_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    warnings.warn("flash_attn not available, using native PyTorch attention")
    
    def flash_attn_varlen_func(*args, **kwargs):
        """Fallback implementation using native PyTorch attention"""
        raise NotImplementedError("flash_attn_varlen_func not available on macOS. Use TorchAttention instead.")

from torch import nn

class TorchAttention(nn.Module):
    def tflops(self, args, kwargs, output) -> float:
        assert len(args) == 0 or len(args) > 2, "query, key should both provided by args / kwargs"
        q = kwargs.get("query") or args[0]
        k = kwargs.get("key") or args[1]
        b, h, sq, d = q.shape
        b, h, sk, d = k.shape
        return b * h * (4 * d * (sq / 1e6) * (sk / 1e6))

    def forward(self, *args, **kwargs):
        return F.scaled_dot_product_attention(*args, **kwargs)


class FlashAttentionVarlen(nn.Module):
    """macOS-compatible fallback that uses native PyTorch attention"""
    
    def __init__(self):
        super().__init__()
        if not FLASH_ATTN_AVAILABLE:
            warnings.warn("FlashAttentionVarlen falling back to native PyTorch attention on macOS")
    
    def tflops(self, args, kwargs, output) -> float:
        if "cu_seqlens_q" in kwargs and "cu_seqlens_k" in kwargs:
            cu_seqlens_q = kwargs["cu_seqlens_q"]
            cu_seqlens_k = kwargs["cu_seqlens_k"]
            _, h, d = output.shape
            seqlens_q = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]) / 1e6
            seqlens_k = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]) / 1e6
            return h * (4 * d * (seqlens_q * seqlens_k).sum())
        else:
            # Fallback calculation for non-varlen case
            return 0.0

    def forward(self, *args, **kwargs):
        if FLASH_ATTN_AVAILABLE:
            kwargs["deterministic"] = torch.are_deterministic_algorithms_enabled()
            return flash_attn_varlen_func(*args, **kwargs)
        else:
            # macOS fallback: convert varlen format to regular attention
            return self._fallback_attention(*args, **kwargs)
    
    def _fallback_attention(self, q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, **kwargs):
        """
        Fallback implementation that converts varlen format to regular attention
        This is a simplified version - may not handle all edge cases
        """
        # Convert from varlen format to regular batch format
        batch_size = len(cu_seqlens_q) - 1
        
        # This is a simplified fallback - in practice you'd need more sophisticated conversion
        # For now, we'll raise an error to indicate this path needs implementation
        raise NotImplementedError(
            "FlashAttentionVarlen fallback not fully implemented for macOS. "
            "Consider using TorchAttention or implementing varlen to batch conversion."
        )


# macOS compatibility alias - prefer TorchAttention over FlashAttentionVarlen
if not FLASH_ATTN_AVAILABLE:
    # Override the original FlashAttentionVarlen to use TorchAttention when possible
    class FlashAttentionVarlenMacOS(TorchAttention):
        """macOS-specific fallback that uses TorchAttention for most cases"""
        
        def forward(self, *args, **kwargs):
            # If this is a simple attention case, use TorchAttention
            if len(args) >= 3 and all(isinstance(arg, torch.Tensor) for arg in args[:3]):
                q, k, v = args[:3]
                if q.dim() == 4 and k.dim() == 4 and v.dim() == 4:  # Regular format
                    return F.scaled_dot_product_attention(q, k, v)
            
            # For varlen format, we need proper conversion - not implemented yet
            raise NotImplementedError(
                "Complex FlashAttention patterns not supported on macOS. "
                "Use regular attention patterns or implement varlen conversion."
            )
    
    # Replace the class for macOS compatibility
    FlashAttentionVarlen = FlashAttentionVarlenMacOS