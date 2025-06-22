# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SeedVR is a video restoration project using diffusion transformers. It includes two versions:
- **SeedVR**: Multi-step diffusion-based video restoration 
- **SeedVR2**: One-step video restoration via diffusion adversarial post-training

The codebase implements large-scale diffusion transformer models (3B and 7B parameters) for generic video restoration tasks like super-resolution, denoising, and artifact removal.

## Environment Setup

```bash
# Create conda environment
conda create -n seedvr python=3.10 -y
conda activate seedvr

# Install dependencies
pip install -r requirements.txt
pip install flash_attn==2.5.9.post1 --no-build-isolation

# Install apex (required for training/inference optimizations)
# Use pre-built wheels if compilation fails:
# For python=3.10, torch=2.4.0, cuda=12.1:
pip install apex-0.1-cp310-cp310-linux_x86_64.whl
```

## Model Architecture

### Core Components
- **DiT (Diffusion Transformer)**: Located in `models/dit/` and `models/dit_v2/`
  - Attention mechanisms with window-based and multimodal support
  - Positional embeddings (RoPE) and modulation layers
  - Support for variable resolution inputs

- **Video VAE**: Located in `models/video_vae_v3/`
  - Handles video encoding/decoding with temporal consistency
  - Context parallel processing for long videos
  - Causal inflation for temporal modeling

- **Data Processing**: Located in `data/`
  - Image transforms: area resize, divisible crop, side resize
  - Video transforms: frame rearrangement and temporal processing

## Inference Commands

### SeedVR2 (One-step restoration)
```bash
# 3B model
torchrun --nproc-per-node=NUM_GPUS projects/inference_seedvr2_3b.py \
  --video_path INPUT_FOLDER \
  --output_dir OUTPUT_FOLDER \
  --seed SEED_NUM \
  --res_h OUTPUT_HEIGHT \
  --res_w OUTPUT_WIDTH \
  --sp_size NUM_SP

# 7B model  
torchrun --nproc-per-node=NUM_GPUS projects/inference_seedvr2_7b.py \
  --video_path INPUT_FOLDER \
  --output_dir OUTPUT_FOLDER \
  --seed SEED_NUM \
  --res_h OUTPUT_HEIGHT \
  --res_w OUTPUT_WIDTH \
  --sp_size NUM_SP
```

### SeedVR (Multi-step restoration)
```bash
# 3B model
torchrun --nproc-per-node=NUM_GPUS projects/inference_seedvr_3b.py \
  --video_path INPUT_FOLDER \
  --output_dir OUTPUT_FOLDER \
  --seed SEED_NUM \
  --res_h OUTPUT_HEIGHT \
  --res_w OUTPUT_WIDTH \
  --sp_size NUM_SP

# 7B model
torchrun --nproc-per-node=NUM_GPUS projects/inference_seedvr_7b.py \
  --video_path INPUT_FOLDER \
  --output_dir OUTPUT_FOLDER \
  --seed SEED_NUM \
  --res_h OUTPUT_HEIGHT \
  --res_w OUTPUT_WIDTH \
  --sp_size NUM_SP
```

## GPU Requirements
- **1 H100-80G**: Videos up to 100x720x1280
- **4 H100-80G**: 1080p and 2K videos (use sp_size=4 for sequence parallelism)
- Multi-GPU inference uses sequence parallel for memory efficiency

## Configuration Files
- `configs_3b/main.yaml`: Configuration for 3B parameter models
- `configs_7b/main.yaml`: Configuration for 7B parameter models
- Model configurations include architecture details, training hyperparameters, and data processing settings

## Development Tools
The project includes standard Python development tools:
- **black**: Code formatting (`black >= 24, < 25`)
- **flake8**: Code style linting (`flake8 >= 7, < 8`) 
- **isort**: Import sorting (`isort >= 5, < 6`)
- **pre-commit**: Pre-commit hooks (`pre-commit==3.7.0`)

Run formatting and linting:
```bash
black .
flake8 .
isort .
```

## Key Dependencies
- **torch==2.3.0**: Main deep learning framework
- **diffusers==0.29.1**: Hugging Face diffusion models
- **transformers==4.38.2**: Transformer architectures
- **flash_attn==2.5.9.post1**: Optimized attention implementation
- **apex**: NVIDIA optimizations (install separately)
- **einops==0.7.0**: Tensor operations
- **omegaconf==2.3.0**: Configuration management

## Optional Features
- **Color Fix**: Place `color_fix.py` in `./projects/video_diffusion_sr/` for wavelet-based color correction
- **Distributed Training**: Uses custom distributed utilities in `common/distributed/`
- **Advanced Metrics**: Includes LPIPS, FID, and other perceptual metrics via `torchmetrics`

## Model Download
Models are hosted on Hugging Face Hub under `ByteDance-Seed/SeedVR*` repositories. Use `huggingface_hub.snapshot_download()` to download checkpoints to `ckpts/` directory.