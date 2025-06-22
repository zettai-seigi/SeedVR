# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# // macOS-compatible version with MPS support
# // Modified for Apple Silicon compatibility

import os
import sys

# macOS compatibility: Enable MPS fallback for unsupported operations
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch
import mediapy
from einops import rearrange
from omegaconf import OmegaConf
print(os.getcwd())
import datetime
from tqdm import tqdm
import gc
import warnings

# macOS compatibility: Mock flash_attn module with working fallbacks
class MockFlashAttn:
    @staticmethod
    def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, 
                              dropout_p=0.0, softmax_scale=None, causal=False, **kwargs):
        """Fallback implementation using native PyTorch attention"""
        # Convert varlen format to regular batched format
        batch_size = len(cu_seqlens_q) - 1
        
        # Simple implementation: process each sequence in the batch
        outputs = []
        for i in range(batch_size):
            start_q = cu_seqlens_q[i]
            end_q = cu_seqlens_q[i + 1]
            start_k = cu_seqlens_k[i]
            end_k = cu_seqlens_k[i + 1]
            
            q_seq = q[start_q:end_q].unsqueeze(0)  # Add batch dim
            k_seq = k[start_k:end_k].unsqueeze(0)
            v_seq = v[start_k:end_k].unsqueeze(0)
            
            # Reshape for attention: (batch, seq, heads, dim) -> (batch, heads, seq, dim)
            if q_seq.dim() == 3:  # (seq, heads, dim)
                q_seq = q_seq.transpose(0, 1).unsqueeze(0)  # (1, heads, seq, dim)
                k_seq = k_seq.transpose(0, 1).unsqueeze(0)
                v_seq = v_seq.transpose(0, 1).unsqueeze(0)
            elif q_seq.dim() == 4 and q_seq.size(0) == 1:  # (1, seq, heads, dim)
                q_seq = q_seq.transpose(1, 2)  # (1, heads, seq, dim)
                k_seq = k_seq.transpose(1, 2)
                v_seq = v_seq.transpose(1, 2)
            
            # Use native scaled_dot_product_attention
            out_seq = torch.nn.functional.scaled_dot_product_attention(
                q_seq, k_seq, v_seq, dropout_p=dropout_p, is_causal=causal
            )
            
            # Reshape back to match expected output format
            if out_seq.dim() == 4:  # (1, heads, seq, dim)
                out_seq = out_seq.transpose(1, 2).squeeze(0)  # (seq, heads, dim)
            
            outputs.append(out_seq)
        
        # Concatenate all outputs
        return torch.cat(outputs, dim=0)
    
    @staticmethod 
    def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, **kwargs):
        """Simple fallback for regular attention"""
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=dropout_p, is_causal=causal
        )

# macOS compatibility: Mock apex module with proper spec
import importlib.util
import types

class MockApexNormalization:
    @staticmethod
    def FusedRMSNorm(normalized_shape, eps=1e-5, elementwise_affine=True):
        """Fallback to PyTorch RMSNorm"""
        return torch.nn.RMSNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)
    
    @staticmethod
    def FusedLayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True):
        """Fallback to PyTorch LayerNorm"""
        return torch.nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)

# Create proper module objects with specs
def create_mock_module(name):
    spec = importlib.util.spec_from_loader(name, loader=None)
    module = importlib.util.module_from_spec(spec)
    module.__spec__ = spec
    return module

# Create mock apex module
mock_apex = create_mock_module('apex')
mock_apex.normalization = MockApexNormalization()

# Create mock apex.normalization module
mock_apex_norm = create_mock_module('apex.normalization')
mock_apex_norm.FusedRMSNorm = MockApexNormalization.FusedRMSNorm
mock_apex_norm.FusedLayerNorm = MockApexNormalization.FusedLayerNorm

# Create mock flash_attn module
mock_flash_attn = create_mock_module('flash_attn')
mock_flash_attn.flash_attn_varlen_func = MockFlashAttn.flash_attn_varlen_func
mock_flash_attn.flash_attn_func = MockFlashAttn.flash_attn_func

# Inject mock modules into sys.modules before any model imports
sys.modules['flash_attn'] = mock_flash_attn
sys.modules['apex'] = mock_apex
sys.modules['apex.normalization'] = mock_apex_norm

# macOS compatibility: Monkey patch torch.distributed to avoid initialization issues
import torch.distributed as dist
original_barrier = dist.barrier
original_init_process_group = dist.init_process_group

def noop_barrier(*args, **kwargs):
    """No-op barrier for single device"""
    pass

def noop_init_process_group(*args, **kwargs):
    """No-op init_process_group for single device"""
    pass

# Only patch if not already initialized
if not dist.is_initialized():
    dist.barrier = noop_barrier
    dist.init_process_group = noop_init_process_group
    
    # Patch additional distributed functions for single-device mode
    def noop_get_rank(*args, **kwargs):
        return 0
    
    def noop_get_world_size(*args, **kwargs):
        return 1
    
    def noop_is_initialized(*args, **kwargs):
        return True
    
    dist.get_rank = noop_get_rank
    dist.get_world_size = noop_get_world_size
    # Override is_initialized to return True after our patches
    dist.is_initialized = noop_is_initialized
    
    # Set a flag to indicate we're in single-device mode
    dist._is_single_device_mode = True

# macOS compatibility: Monkey patch dtype conversions 
original_to_cuda = torch.Tensor.cuda
original_to = torch.Tensor.to

def safe_cuda_conversion(self, device=None, **kwargs):
    """Redirect CUDA calls to appropriate device"""
    target_device = get_device()
    if target_device.type == "mps":
        return self.to(target_device, **kwargs)
    elif target_device.type == "cpu":
        return self.to(target_device, **kwargs)
    else:
        return original_to_cuda(self, device, **kwargs)

torch.Tensor.cuda = safe_cuda_conversion
# Will patch .to() method after get_device is available

# Store original arange for later patching
original_arange = torch.arange

# Store original bfloat16 for later replacement
_original_bfloat16 = torch.bfloat16

from data.image.transforms.divisible_crop import DivisibleCrop
from data.image.transforms.na_resize import NaResize
from data.video.transforms.rearrange import Rearrange

# Color fix is optional on macOS
if os.path.exists("./projects/video_diffusion_sr/color_fix.py"):
    from projects.video_diffusion_sr.color_fix import wavelet_reconstruction
    use_colorfix = True
else:
    use_colorfix = False
    print('Note!!! Color fix is not available!')

from torchvision.transforms import Compose, Lambda, Normalize
from torchvision.io.video import read_video
from torchvision.io import read_image
import argparse

# macOS compatibility: use our macOS-specific distributed module
from common.distributed.macos import get_device, init_torch

# macOS compatibility: Monkey patch torch.arange to redirect CUDA device calls
def safe_arange(*args, device=None, **kwargs):
    """Redirect CUDA device calls in torch.arange to appropriate device"""
    if device is not None and isinstance(device, (str, torch.device)):
        device_str = str(device)
        if 'cuda' in device_str:
            target_device = get_device()
            device = target_device
    return original_arange(*args, device=device, **kwargs)

torch.arange = safe_arange

# macOS compatibility: Patch .to() method to handle bfloat16 conversions
def safe_to(self, *args, **kwargs):
    """Patch .to() method to handle problematic dtype conversions"""
    # Handle the case where dtype might cause CUDA initialization
    new_args = []
    new_kwargs = {}
    
    for arg in args:
        # Handle device arguments
        if isinstance(arg, torch.device) and arg.type == "cuda":
            arg = get_device()
        elif isinstance(arg, str) and "cuda" in arg:
            arg = get_device()
        # Handle dtype arguments
        elif arg is torch.bfloat16:
            device = get_device()
            if device.type == "mps":
                arg = torch.float32  # Use float32 for MPS to avoid dtype mixing
            elif device.type == "cpu":
                arg = torch.float32
        new_args.append(arg)
    
    for key, value in kwargs.items():
        # Handle device keyword arguments
        if key == "device":
            if isinstance(value, torch.device) and value.type == "cuda":
                value = get_device()
            elif isinstance(value, str) and "cuda" in value:
                value = get_device()
        # Handle dtype keyword arguments
        elif key == "dtype" and value is torch.bfloat16:
            device = get_device()
            if device.type == "mps":
                value = torch.float32  # Use float32 for MPS to avoid dtype mixing
            elif device.type == "cpu":
                value = torch.float32
        new_kwargs[key] = value
    
    return original_to(self, *new_args, **new_kwargs)

torch.Tensor.to = safe_to

# Don't replace torch.bfloat16 globally as it breaks serialization

# Simplified distributed functions for single-device macOS
def get_data_parallel_rank():
    return 0

def get_data_parallel_world_size():
    return 1

def get_sequence_parallel_rank():
    return 0

def get_sequence_parallel_world_size():
    return 1

def sync_data(data, rank):
    """No-op for single device"""
    return data

from projects.video_diffusion_sr.infer import VideoDiffusionInfer
from common.config import load_config
from common.seed import set_seed
from common.partition import partition_by_groups, partition_by_size

def configure_sequence_parallel(sp_size):
    """Disabled for macOS single-device setup"""
    if sp_size > 1:
        print(f"Warning: Sequence parallel (sp_size={sp_size}) not supported on macOS. Using sp_size=1")
        return 1
    return sp_size

def is_image_file(filename):
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    return os.path.splitext(filename.lower())[1] in image_exts

def configure_runner(sp_size):
    config_path = os.path.join('./configs_3b', 'main.yaml')
    config = load_config(config_path)
    runner = VideoDiffusionInfer(config)
    OmegaConf.set_readonly(runner.config, False)
    
    # macOS: For numerical stability, force everything to run on CPU
    init_torch(cudnn_benchmark=False, timeout=datetime.timedelta(seconds=3600))
    sp_size = configure_sequence_parallel(sp_size)
    
    # Update checkpoint path and force CPU device for stability
    device = get_device()
    if device.type == "mps":
        print("🔧 Forcing CPU mode for numerical stability (MPS has issues with this model)")
        # Override device to CPU for the entire runner configuration - make it permanent
        import common.distributed.macos as macos_module
        macos_module.get_device = lambda: torch.device("cpu")
        device = torch.device("cpu")
        # Also override the global get_device in common.distributed
        import common.distributed
        common.distributed.get_device = lambda: torch.device("cpu")
    
    checkpoint_path = './ckpts/SeedVR2-3B/seedvr2_ema_3b.pth'
    if not os.path.exists(checkpoint_path):
        checkpoint_path = './ckpts/seedvr2_ema_3b.pth'  # fallback
    
    runner.configure_dit_model(device="cpu", checkpoint=checkpoint_path)
    
    # Force DiT model to float32 for stability
    print(f"🔧 Converting DiT model to float32 for numerical stability")
    runner.dit.to(device="cpu", dtype=torch.float32)
    
    # Monkey patch the embedding module to force CPU device
    if hasattr(runner.dit, 'emb_in'):
        original_emb_in_forward = runner.dit.emb_in.forward
        def patched_emb_in_forward(self, timestep, device=None, dtype=None):
            # Force both input tensor and device to CPU
            if hasattr(timestep, 'to'):
                timestep = timestep.to(device="cpu", dtype=torch.float32)
            print(f"🔧 Patching emb_in - timestep device: {timestep.device}, forcing CPU")
            result = original_emb_in_forward(timestep, device=torch.device("cpu"), dtype=torch.float32)
            # Ensure result is also on CPU
            if hasattr(result, 'to'):
                result = result.to(device="cpu", dtype=torch.float32)
            return result
        
        import types
        runner.dit.emb_in.forward = types.MethodType(patched_emb_in_forward, runner.dit.emb_in)
        print("🔧 Applied emb_in device patch")
    
    # macOS: Update VAE checkpoint path 
    vae_checkpoint_path = './ckpts/SeedVR2-3B/ema_vae.pth'
    if not os.path.exists(vae_checkpoint_path):
        vae_checkpoint_path = './ckpts/ema_vae.pth'  # fallback
    runner.config.vae.checkpoint = vae_checkpoint_path
    print(f"🔧 Using VAE checkpoint: {vae_checkpoint_path}")
    
    # macOS: Override VAE dtype for compatibility - must be done before configure_vae_model
    if device.type == "mps":
        # Use float32 for MPS to avoid dtype mismatch issues
        runner.config.vae.dtype = "float32"
        print(f"🔧 Overriding VAE dtype to float32 for MPS compatibility (avoids f16/f32 mixing)")
    elif device.type == "cpu":
        # CPU works best with float32  
        runner.config.vae.dtype = "float32"
        print(f"🔧 Overriding VAE dtype to float32 for CPU compatibility")
    
    # Monkey patch the specific getattr call in configure_vae_model
    import projects.video_diffusion_sr.infer as infer_module
    from common.config import create_object
    original_configure_vae = infer_module.VideoDiffusionInfer.configure_vae_model
    
    def patched_configure_vae_model(self):
        # Create vae model.
        # Force float32 and CPU for numerical stability
        dtype = torch.float32
        target_device = torch.device("cpu")
        print(f"🔧 Forcing VAE to use float32 on CPU for stability")
            
        self.vae = create_object(self.config.vae.model)
        self.vae.requires_grad_(False).eval()
        self.vae.to(device=target_device, dtype=dtype)

        # Load vae checkpoint.
        state = torch.load(
            self.config.vae.checkpoint, map_location=target_device, mmap=True
        )
        self.vae.load_state_dict(state)

        # Set causal slicing.
        if hasattr(self.vae, "set_causal_slicing") and hasattr(self.config.vae, "slicing"):
            self.vae.set_causal_slicing(**self.config.vae.slicing)
    
    # Temporarily replace the method
    infer_module.VideoDiffusionInfer.configure_vae_model = patched_configure_vae_model
    
    try:
        runner.configure_vae_model()
    finally:
        # Restore original method
        infer_module.VideoDiffusionInfer.configure_vae_model = original_configure_vae
    
    # Set memory limit (adjust for Apple Silicon)
    if hasattr(runner.vae, "set_memory_limit"):
        # Set reasonable memory limits for Apple Silicon
        memory_config = runner.config.vae.get('memory_limit', {})
        if memory_config:
            # Ensure memory limits are reasonable (not too small which causes overflow)
            for key in memory_config:
                if isinstance(memory_config[key], (int, float)):
                    # Set minimum memory limit to avoid division by zero/infinity
                    original_value = memory_config[key]
                    if original_value <= 0 or original_value < 0.1:
                        memory_config[key] = 2.0  # 2GB minimum
                        print(f"🔧 Adjusted {key} memory limit from {original_value} to {memory_config[key]} GB")
                    else:
                        # Keep original or scale reasonably
                        memory_config[key] = max(1.0, original_value)  # At least 1GB
        else:
            # Set default memory limits if none exist
            memory_config = {
                'conv_max_mem': 2.0,  # 2GB for conv operations
                'norm_max_mem': 1.0,  # 1GB for normalization
            }
            print(f"🔧 Setting default memory limits: {memory_config}")
        
        runner.vae.set_memory_limit(**memory_config)
    
    # Configure diffusion components for CPU
    print("🔧 Configuring diffusion components for CPU...")
    runner.configure_diffusion()
    
    # Ensure scheduler is on CPU and patch schedule operations
    if hasattr(runner, 'schedule'):
        if hasattr(runner.schedule, 'to'):
            runner.schedule.to("cpu")
        
        # Patch the schedule's convert_from_pred method to ensure CPU tensors
        if hasattr(runner.schedule, 'convert_from_pred'):
            original_convert_from_pred = runner.schedule.convert_from_pred
            def patched_convert_from_pred(self, pred, prediction_type, x_t, t):
                # Force all inputs to CPU
                if hasattr(pred, 'to'):
                    pred = pred.to("cpu")
                if hasattr(x_t, 'to'):
                    x_t = x_t.to("cpu")
                if hasattr(t, 'to'):
                    t = t.to("cpu")
                print(f"🔧 Schedule convert_from_pred - pred: {pred.device}, x_t: {x_t.device}, t: {t.device}")
                result = original_convert_from_pred(pred, prediction_type, x_t, t)
                # Ensure outputs are on CPU
                if isinstance(result, tuple):
                    result = tuple(r.to("cpu") if hasattr(r, 'to') else r for r in result)
                elif hasattr(result, 'to'):
                    result = result.to("cpu")
                return result
            
            import types
            runner.schedule.convert_from_pred = types.MethodType(patched_convert_from_pred, runner.schedule)
            print("🔧 Applied schedule convert_from_pred patch")
        
        # Also patch the schedule's forward method
        if hasattr(runner.schedule, 'forward'):
            original_schedule_forward = runner.schedule.forward
            def patched_schedule_forward(self, x_0, x_T, t):
                # Force all inputs to CPU
                if hasattr(x_0, 'to'):
                    x_0 = x_0.to("cpu")
                if hasattr(x_T, 'to'):
                    x_T = x_T.to("cpu")
                if hasattr(t, 'to'):
                    t = t.to("cpu")
                print(f"🔧 Schedule forward - x_0: {x_0.device}, x_T: {x_T.device}, t: {t.device}")
                result = original_schedule_forward(x_0, x_T, t)
                # Ensure output is on CPU
                if hasattr(result, 'to'):
                    result = result.to("cpu")
                return result
            
            runner.schedule.forward = types.MethodType(patched_schedule_forward, runner.schedule)
            print("🔧 Applied schedule forward patch")
    
    # Also patch the sampler's step_to method
    if hasattr(runner, 'sampler') and hasattr(runner.sampler, 'step_to'):
        original_step_to = runner.sampler.step_to
        def patched_step_to(self, pred, x_t, t, s):
            # Force all inputs to CPU
            if hasattr(pred, 'to'):
                pred = pred.to("cpu")
            if hasattr(x_t, 'to'):
                x_t = x_t.to("cpu")
            if hasattr(t, 'to'):
                t = t.to("cpu")
            if hasattr(s, 'to'):
                s = s.to("cpu")
            print(f"🔧 Sampler step_to - pred: {pred.device}, x_t: {x_t.device}, t: {t.device}, s: {s.device}")
            result = original_step_to(pred, x_t, t, s)
            # Ensure result is on CPU
            if hasattr(result, 'to'):
                result = result.to("cpu")
            return result
        
        import types
        runner.sampler.step_to = types.MethodType(patched_step_to, runner.sampler)
        print("🔧 Applied sampler step_to patch")
    
    return runner

def generation_step(runner, text_embeds_dict, cond_latents):
    # Everything runs on CPU for stability
    target_device = torch.device("cpu")
    
    def _move_to_device(x):
        return [i.to(target_device) for i in x]

    # Ensure all tensors are created on CPU
    noises = [torch.randn_like(latent).to(target_device) for latent in cond_latents]
    aug_noises = [torch.randn_like(latent).to(target_device) for latent in cond_latents]
    print(f"Generating with noise shape: {noises[0].size()}.")
    print(f"🔧 Noise device: {noises[0].device}")
    
    # No distributed sync needed for single device - everything should already be on CPU
    noises, aug_noises, cond_latents = list(
        map(lambda x: _move_to_device(x), (noises, aug_noises, cond_latents))
    )
    
    # Double-check all tensors are on CPU
    print(f"🔧 After move_to_device - noise device: {noises[0].device}")
    print(f"🔧 After move_to_device - cond_latents device: {cond_latents[0].device}")
    cond_noise_scale = 0.0

    def _add_noise(x, aug_noise):
        # Use x.device to ensure we're on the same device as the input tensor
        target_device = x.device
        t = (
            torch.tensor([1000.0], device=target_device)
            * cond_noise_scale
        )
        shape = torch.tensor(x.shape[1:], device=target_device)[None]
        t = runner.timestep_transform(t, shape)
        print(
            f"Timestep shifting from"
            f" {1000.0 * cond_noise_scale} to {t}."
        )
        x = runner.schedule.forward(x, aug_noise, t)
        return x

    conditions = [
        runner.get_condition(
            noise,
            task="sr",
            latent_blur=_add_noise(latent_blur, aug_noise),
        )
        for noise, aug_noise, latent_blur in zip(noises, aug_noises, cond_latents)
    ]

    with torch.no_grad():
        # Everything is already on CPU with float32, just run inference
        print("🔧 Running CPU inference with float32...")
        
        # Convert tensors to float32 for computation
        cpu_noises = [noise.to(device="cpu", dtype=torch.float32) for noise in noises]
        cpu_conditions = [cond.to(device="cpu", dtype=torch.float32) for cond in conditions]
        cpu_text_embeds = {
            "texts_pos": [emb.to(device="cpu", dtype=torch.float32) for emb in text_embeds_dict["texts_pos"]],
            "texts_neg": [emb.to(device="cpu", dtype=torch.float32) for emb in text_embeds_dict["texts_neg"]]
        }
        
        # Debug tensor devices before inference
        print(f"🔧 Before inference - cpu_noises device: {cpu_noises[0].device}")
        print(f"🔧 Before inference - cpu_conditions device: {cpu_conditions[0].device}")
        print(f"🔧 Before inference - text_pos device: {cpu_text_embeds['texts_pos'][0].device}")
        print(f"🔧 Model dit device: {next(runner.dit.parameters()).device}")
        print(f"🔧 Model vae device: {next(runner.vae.parameters()).device}")
        
        video_tensors = runner.inference(
            noises=cpu_noises,
            conditions=cpu_conditions,
            dit_offload=False,  # Keep on CPU
            **cpu_text_embeds,
        )

    print(f"🔧 Video tensors from inference: {[v.shape for v in video_tensors]}")
    if len(video_tensors) > 0:
        print(f"🔧 Video tensor values - min: {video_tensors[0].min():.4f}, max: {video_tensors[0].max():.4f}, mean: {video_tensors[0].mean():.4f}")
    
    # video_tensors from inference are already decoded and in the correct format
    # Just return them directly - no need for additional rearranging
    samples = video_tensors
    
    print(f"🔧 Final samples: {[s.shape for s in samples]}")
    if len(samples) > 0:
        print(f"🔧 Sample values - min: {samples[0].min():.4f}, max: {samples[0].max():.4f}, mean: {samples[0].mean():.4f}")

    return samples

def generation_loop(runner, video_path='./test_videos', output_dir='./results', batch_size=1, cfg_scale=1.0, cfg_rescale=0.0, sample_steps=1, seed=666, res_h=None, res_w=None, sp_size=1, out_fps=None):

    def _build_pos_and_neg_prompt():
        positive_text = "Cinematic, High Contrast, highly detailed, taken using a Canon EOS R camera, \
        hyper detailed photo - realistic maximum detail, 32k, Color Grading, ultra HD, extreme meticulous detailing, \
        skin pore detailing, hyper sharpness, perfect without deformations."
        negative_text = "painting, oil painting, illustration, drawing, art, sketch, oil painting, cartoon, \
        CG Style, 3D render, unreal engine, blurring, dirty, messy, worst quality, low quality, frames, watermark, \
        signature, jpeg artifacts, deformed, lowres, over-smooth"
        return positive_text, negative_text

    def _build_test_prompts(video_path):
        positive_text, negative_text = _build_pos_and_neg_prompt()
        original_videos = []
        prompts = {}
        video_list = os.listdir(video_path)
        for f in video_list:
            original_videos.append(f)
            prompts[f] = positive_text
        print(f"Total prompts to be generated: {len(original_videos)}")
        return original_videos, prompts, negative_text

    def _extract_text_embeds():
        positive_prompts_embeds = []
        for texts_pos in tqdm(original_videos_local):
            text_pos_embeds = torch.load('pos_emb.pt', map_location='cpu')
            text_neg_embeds = torch.load('neg_emb.pt', map_location='cpu')

            positive_prompts_embeds.append(
                {"texts_pos": [text_pos_embeds], "texts_neg": [text_neg_embeds]}
            )
        gc.collect()
        # macOS: Clear MPS cache instead of CUDA
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        return positive_prompts_embeds

    def cut_videos(videos, sp_size):
        # Force sp_size=1 for macOS
        sp_size = 1
        t = videos.size(1)
        if t == 1:
            return videos
        if t <= 4 * sp_size:
            print(f"Cut input video size: {videos.size()}")
            padding = [videos[:, -1].unsqueeze(1)] * (4 * sp_size - t + 1)
            padding = torch.cat(padding, dim=1)
            videos = torch.cat([videos, padding], dim=1)
            return videos
        if (t - 1) % (4 * sp_size) == 0:
            return videos
        else:
            padding = [videos[:, -1].unsqueeze(1)] * (
                4 * sp_size - ((t - 1) % (4 * sp_size))
            )
            padding = torch.cat(padding, dim=1)
            videos = torch.cat([videos, padding], dim=1)
            assert (videos.size(1) - 1) % (4 * sp_size) == 0
            return videos

    # Force sp_size=1 for macOS compatibility
    sp_size = 1
    
    # classifier-free guidance
    runner.config.diffusion.cfg.scale = cfg_scale
    runner.config.diffusion.cfg.rescale = cfg_rescale
    # sampling steps
    runner.config.diffusion.timesteps.sampling.steps = sample_steps
    # Diffusion already configured in configure_runner, just update config

    # set random seed
    set_seed(seed, same_across_ranks=True)
    os.makedirs(output_dir, exist_ok=True)
    tgt_path = output_dir

    # get test prompts
    original_videos, _, _ = _build_test_prompts(video_path)

    # Simplified for single device
    original_videos_local = partition_by_size(original_videos, batch_size)

    # pre-extract the text embeddings
    positive_prompts_embeds = _extract_text_embeds()

    # generation loop
    for videos, text_embeds in tqdm(zip(original_videos_local, positive_prompts_embeds)):
        raw_videos = []
        fps_lists = []
        # Get input properties for intelligent resolution handling
        input_heights = []
        input_widths = []
        
        for video in videos:
            if is_image_file(video):
                video = read_image(
                    os.path.join(video_path, video)
                ).unsqueeze(0) / 255.0
                fps_lists.append(1.0)  # Default fps for images (not used but needed for zip)
            else:
                video, _, info = read_video(
                    os.path.join(video_path, video), output_format="TCHW"
                    )
                video = video / 255.0
                fps_lists.append(info["video_fps"] if out_fps is None else out_fps)
            
            print(f"📐 Input size: {video.size()}")
            input_heights.append(video.size(2))
            input_widths.append(video.size(3))
            
            # macOS compatibility: Handle channel mismatch (4 channels to 3 channels)
            if video.size(1) == 4:  # RGBA to RGB conversion
                video = video[:, :3, :, :]  # Remove alpha channel
                print(f"🔧 Converted RGBA to RGB, new size: {video.size()}")
            
            # Store videos on CPU for consistency
            raw_videos.append(video.to(torch.device("cpu")))
        
        # Use input resolution if not specified
        if res_h is None or res_w is None:
            avg_h = sum(input_heights) // len(input_heights)
            avg_w = sum(input_widths) // len(input_widths)
            # Round to nearest multiple of 16 for model compatibility
            actual_res_h = ((avg_h + 15) // 16) * 16
            actual_res_w = ((avg_w + 15) // 16) * 16
            print(f"🎯 Auto-detected resolution: {actual_res_h}x{actual_res_w} (from input {avg_h}x{avg_w})")
        else:
            actual_res_h = res_h
            actual_res_w = res_w
            print(f"🎯 Using specified resolution: {actual_res_h}x{actual_res_w}")

        # Create video transform with determined resolution
        video_transform = Compose([
            NaResize(
                resolution=(actual_res_h * actual_res_w) ** 0.5,
                mode="area",
                downsample_only=False,
            ),
            Lambda(lambda x: torch.clamp(x, 0.0, 1.0)),
            DivisibleCrop((16, 16)),
            Normalize(0.5, 0.5),
            Rearrange("t c h w -> c t h w"),
        ])

        # Apply transform to all videos on CPU
        cond_latents = []
        for video in raw_videos:
            # Ensure video is on CPU before transform
            video_cpu = video.to(torch.device("cpu"))
            transformed = video_transform(video_cpu)
            # Ensure result is on CPU
            cond_latents.append(transformed.to(torch.device("cpu")))

        ori_lengths = [video.size(1) for video in cond_latents]
        input_videos = cond_latents
        cond_latents = [cut_videos(video, sp_size) for video in cond_latents]

        runner.dit.to("cpu")
        print(f"Encoding videos: {list(map(lambda x: x.size(), cond_latents))}")
        # Move cond_latents to CPU before VAE encoding
        print("🔧 Moving video tensors to CPU before VAE encoding...")
        cond_latents = [latent.to("cpu") for latent in cond_latents]
        print("🔧 VAE encoding on CPU...")
        cond_latents = runner.vae_encode(cond_latents)
        # Ensure encoded latents are on CPU
        cond_latents = [latent.to("cpu") for latent in cond_latents]
        print(f"🔧 Encoded latents shapes: {[latent.shape for latent in cond_latents]}")
        # Models stay on CPU

        # Everything runs on CPU, so move text embeddings to CPU too
        target_device = torch.device("cpu")
        for i, emb in enumerate(text_embeds["texts_pos"]):
            text_embeds["texts_pos"][i] = emb.to(target_device)
        for i, emb in enumerate(text_embeds["texts_neg"]):
            text_embeds["texts_neg"][i] = emb.to(target_device)

        samples = generation_step(runner, text_embeds, cond_latents=cond_latents)
        runner.dit.to("cpu")
        del cond_latents
        
        # The samples from generation_step are already decoded by the VAE in the inference method
        # Check if samples contain NaN values and handle them
        print(f"🔧 Checking samples for NaN values...")
        for i, sample in enumerate(samples):
            if torch.isnan(sample).any():
                print(f"⚠️  Sample {i} contains NaN values, this indicates a numerical issue in the diffusion process")
                print(f"🔧 Generating a neutral gray output instead of zeros...")
                # Create a neutral gray image instead of zeros
                # Assume the sample is in range [-1, 1] for diffusion models
                samples[i] = torch.zeros_like(sample)  # Start with zeros then set to neutral gray
            else:
                print(f"✅ Sample {i} is valid - min: {sample.min():.4f}, max: {sample.max():.4f}, mean: {sample.mean():.4f}")

        # dump samples to the output directory (always run on single device)
        print(f"🔧 Starting output saving for {len(videos)} files...")
        print(f"🔧 Samples shape: {[s.shape for s in samples]}")
        
        for i, (path, input, sample, ori_length, save_fps) in enumerate(zip(
            videos, input_videos, samples, ori_lengths, fps_lists
        )):
            print(f"🔧 Processing file {i+1}/{len(videos)}: {path}")
            print(f"🔧 Sample shape before processing: {sample.shape}")
            
            if ori_length < sample.shape[0]:
                sample = sample[:ori_length]
            filename = os.path.join(tgt_path, os.path.basename(path))
            print(f"🔧 Output filename: {filename}")
            
            # Ensure input is in the correct format for comparison
            if input.ndim == 3:
                input = rearrange(input[:, None], "c t h w -> t c h w")  
            else:
                input = rearrange(input, "c t h w -> t c h w")
            
            # Ensure sample is in the correct format 
            print(f"🔧 Sample shape before processing: {sample.shape}")
            
            # Handle the sample format - it should be c h w for single images
            if sample.ndim == 3:
                # Format: c h w (single image)
                sample = sample.unsqueeze(0)  # Add time dimension -> t c h w: 1 x c x h x w
            elif sample.ndim == 4:
                # Could be t c h w or c t h w - check which
                if sample.shape[0] == 3 and sample.shape[1] > 3:
                    # Likely c t h w format -> need to transpose to t c h w
                    sample = rearrange(sample, "c t h w -> t c h w")
                # else: already t c h w format
            
            print(f"🔧 Sample shape after dimension handling: {sample.shape}")
            
            if use_colorfix:
                # Ensure both sample and input have 3 channels for color fix
                sample_for_colorfix = sample.to("cpu")
                input_for_colorfix = input[: sample.size(0)].to("cpu")
                
                # Handle channel dimension mismatch
                if sample_for_colorfix.size(1) == 1 and input_for_colorfix.size(1) == 3:
                    # Convert single channel to 3 channels by repeating
                    sample_for_colorfix = sample_for_colorfix.repeat(1, 3, 1, 1)
                    print(f"🔧 Expanded sample channels from 1 to 3 for color fix")
                elif sample_for_colorfix.size(1) == 3 and input_for_colorfix.size(1) == 1:
                    # Convert input single channel to 3 channels
                    input_for_colorfix = input_for_colorfix.repeat(1, 3, 1, 1)
                    print(f"🔧 Expanded input channels from 1 to 3 for color fix")
                
                sample = wavelet_reconstruction(sample_for_colorfix, input_for_colorfix)
            else:
                sample = sample.to("cpu")
                
            # Convert to final output format: t c h w -> t h w c
            sample = rearrange(sample, "t c h w -> t h w c")
            print(f"🔧 Sample shape after final rearrange: {sample.shape}")
            print(f"🔧 Sample before clipping - min: {sample.min():.4f}, max: {sample.max():.4f}, mean: {sample.mean():.4f}")
            sample = sample.clip(-1, 1).mul_(0.5).add_(0.5).mul_(255).round()
            print(f"🔧 Sample after processing - min: {sample.min():.4f}, max: {sample.max():.4f}, mean: {sample.mean():.4f}")
            sample = sample.to(torch.uint8).numpy()
            
            print(f"🔧 Final sample shape: {sample.shape}")
            print(f"🔧 Sample dtype: {sample.dtype}")
            print(f"🔧 Final numpy - min: {sample.min()}, max: {sample.max()}, mean: {sample.mean():.2f}")

            try:
                if sample.shape[0] == 1:
                    print(f"🔧 Writing image to: {filename}")
                    # Handle single image: shape should be (h, w, c)
                    img_array = sample.squeeze(0)  # Remove time dimension: (1, h, w, c) -> (h, w, c) 
                    print(f"🔧 Final image shape: {img_array.shape}")
                    
                    # Ensure we have 3 channels for RGB
                    if img_array.shape[2] == 1:
                        # Grayscale -> RGB
                        img_array = img_array.repeat(3, axis=2)
                        print(f"🔧 Converted grayscale to RGB: {img_array.shape}")
                    
                    mediapy.write_image(filename, img_array)
                    print(f"✅ Image saved successfully!")
                else:
                    print(f"🔧 Writing video to: {filename} with fps: {save_fps}")
                    mediapy.write_video(filename, sample, fps=save_fps)
                    print(f"✅ Video saved successfully!")
            except Exception as e:
                print(f"❌ Error saving {filename}: {e}")
                import traceback
                traceback.print_exc()
        gc.collect()
        # macOS: Clear MPS cache
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("--video_path", type=str, default="./test_videos")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--res_h", type=int, default=None, help="Output height (auto-detected from input if not specified)")
    parser.add_argument("--res_w", type=int, default=None, help="Output width (auto-detected from input if not specified)")
    parser.add_argument("--sp_size", type=int, default=1, help="Forced to 1 on macOS")
    parser.add_argument("--out_fps", type=float, default=None)
    args = parser.parse_args()

    # Force sp_size=1 for macOS
    args.sp_size = 1
    
    print("🍎 Running SeedVR2-3B on macOS with MPS support")
    print(f"Device: {get_device()}")
    
    runner = configure_runner(args.sp_size)
    generation_loop(runner, **vars(args))