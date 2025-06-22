#!/usr/bin/env python3
"""Download SeedVR model checkpoints for macOS"""

import os
from huggingface_hub import snapshot_download

# Create ckpts directory if it doesn't exist
os.makedirs("ckpts", exist_ok=True)

# Download SeedVR2-3B (smaller model, good for testing)
print("Downloading SeedVR2-3B model...")
save_dir = "ckpts/SeedVR2-3B"
repo_id = "ByteDance-Seed/SeedVR2-3B"

try:
    snapshot_download(
        repo_id=repo_id,
        local_dir=save_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
        allow_patterns=["*.json", "*.safetensors", "*.pth", "*.bin", "*.py", "*.md", "*.txt"],
    )
    print(f"✅ Model downloaded to: {save_dir}")
    
    # List downloaded files
    print("\nDownloaded files:")
    for root, dirs, files in os.walk(save_dir):
        for file in files:
            print(f"  {os.path.relpath(os.path.join(root, file), save_dir)}")
            
except Exception as e:
    print(f"❌ Download failed: {e}")