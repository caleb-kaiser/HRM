#!/usr/bin/env python3
"""
Download pretrained HRM models from HuggingFace.

This script downloads the three available pretrained HRM models:
- ARC-AGI-2
- Sudoku 9x9 Extreme 
- Maze 30x30 Hard

Models are saved to ./checkpoints/ with the necessary all_config.yaml files.
"""

import os
from pathlib import Path
from huggingface_hub import snapshot_download
import sys

def download_model(repo_id: str, local_dir: str, model_name: str):
    """Download a model from HuggingFace Hub to local directory."""
    print(f"📦 Downloading {model_name}...")
    print(f"   Repository: {repo_id}")
    print(f"   Local path: {local_dir}")
    
    try:
        # Create the local directory
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        
        # Download the model
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,  # Copy files instead of symlinks
            resume_download=True,  # Resume if partially downloaded
        )
        
        # Check if essential files exist
        config_path = Path(local_dir) / "all_config.yaml"
        if config_path.exists():
            print(f"✅ {model_name} downloaded successfully!")
        else:
            print(f"⚠️  {model_name} downloaded, but all_config.yaml not found")
            print(f"   This might be expected - check the repository contents")
        
        # List downloaded files for verification
        files = list(Path(local_dir).glob("*"))
        print(f"   Downloaded {len(files)} files")
        
    except Exception as e:
        print(f"❌ Failed to download {model_name}: {str(e)}")
        return False
    
    return True

def main():
    """Download all pretrained HRM models."""
    print("🚀 HRM Model Downloader")
    print("=" * 50)
    
    # Model configurations: (repo_id, local_dir, display_name)
    models = [
        ("sapientinc/HRM-checkpoint-ARC-2", 
         "./checkpoints/arc-agi-2", 
         "ARC-AGI-2"),
        
        ("sapientinc/HRM-checkpoint-sudoku-extreme", 
         "./checkpoints/sudoku-extreme", 
         "Sudoku 9x9 Extreme"),
        
        ("sapientinc/HRM-checkpoint-maze-30x30-hard", 
         "./checkpoints/maze-30x30-hard", 
         "Maze 30x30 Hard"),
    ]
    
    # Create base checkpoints directory
    Path("./checkpoints").mkdir(exist_ok=True)
    
    success_count = 0
    total_count = len(models)
    
    for repo_id, local_dir, model_name in models:
        print()
        if download_model(repo_id, local_dir, model_name):
            success_count += 1
        print("-" * 50)
    
    print()
    print("📊 Download Summary:")
    print(f"   ✅ Successfully downloaded: {success_count}/{total_count} models")
    
    if success_count == total_count:
        print("🎉 All models downloaded successfully!")
        print()
        print("💡 Usage:")
        print("   To evaluate a model, use:")
        print("   python evaluate.py checkpoint=./checkpoints/MODEL_NAME/CHECKPOINT_FILE")
        print()
        print("📁 Available models:")
        for _, local_dir, model_name in models:
            print(f"   • {model_name}: {local_dir}")
    else:
        print("⚠️  Some downloads failed. Check the error messages above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 