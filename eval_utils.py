#!/usr/bin/env python3
"""
Evaluation utilities for HRM models.

This module provides lightweight model initialization for evaluation
that doesn't create unnecessary optimizers or training components.
"""

import torch
import torch.nn as nn
from pathlib import Path
import yaml
from typing import Dict, Any
import os

from obj_types import PretrainConfig
from utils.functions import load_model_class
from dataset.common import PuzzleDatasetMetadata


def create_eval_model(config: PretrainConfig, metadata: PuzzleDatasetMetadata):
    """Create model for evaluation without optimizers or training components."""
    
    model_cfg = dict(
        **config.arch.__pydantic_extra__,  # type: ignore
        batch_size=config.global_batch_size,  # Use full batch size for eval
        vocab_size=metadata.vocab_size,
        seq_len=metadata.seq_len,
        num_puzzle_identifiers=metadata.num_puzzle_identifiers,
        causal=False  # Non-autoregressive
    )

    # Instantiate model with loss head
    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)

    with torch.device("cuda"):
        model: nn.Module = model_cls(model_cfg)
        model = loss_head_cls(model, **config.arch.loss.__pydantic_extra__)  # type: ignore
        
        # Don't compile for evaluation to avoid overhead
        if "FORCE_COMPILE" in os.environ:
            model = torch.compile(model, dynamic=False)  # type: ignore

    return model


def load_eval_model(checkpoint_path, device: str = "cuda"):
    """Load a trained model for evaluation."""
    
    checkpoint_path = Path(checkpoint_path)
    checkpoint_dir = checkpoint_path.parent
    config_file = checkpoint_dir / "all_config.yaml"
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    # Load configuration
    with open(config_file, "r") as f:
        config_dict = yaml.safe_load(f)
    config = PretrainConfig(**config_dict)
    
    # Try multiple approaches to get metadata
    metadata = None
    
    # Approach 1: Load from dataset metadata.json
    try:
        import json
        metadata_file = Path(config.data_path) / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                metadata_dict = json.load(f)
                metadata = PuzzleDatasetMetadata(**metadata_dict)
                print(f"✅ Loaded metadata from {metadata_file}")
        else:
            print(f"⚠️  Metadata file not found: {metadata_file}")
    except Exception as e:
        print(f"⚠️  Could not load metadata from dataset: {e}")
    
    # Approach 2: Infer from checkpoint if metadata loading failed
    if metadata is None:
        try:
            print("🔍 Inferring model dimensions from checkpoint...")
            state_dict = torch.load(str(checkpoint_path), map_location="cpu")
            
            # Handle torch.compile naming
            if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
                state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
            
            # Infer dimensions from checkpoint tensors
            vocab_size = None
            num_puzzle_identifiers = None
            seq_len = None
            
            for key, tensor in state_dict.items():
                if "embed_tokens.embedding_weight" in key:
                    vocab_size = tensor.shape[0]
                elif "lm_head.weight" in key:
                    if vocab_size is None:
                        vocab_size = tensor.shape[0]
                elif "puzzle_emb.weights" in key:
                    num_puzzle_identifiers = tensor.shape[0]
                elif "embed_pos.embedding_weight" in key:
                    seq_len = tensor.shape[0]
            
            # Use inferred values or reasonable defaults
            metadata = PuzzleDatasetMetadata(
                # Required token IDs (use standard values)
                pad_id=0,
                ignore_label_id=-100,
                blank_identifier_id=1,
                
                # Inferred or default model dimensions
                vocab_size=vocab_size or 11,  # Common for sudoku (0-9 + special)
                seq_len=seq_len or 200,
                num_puzzle_identifiers=num_puzzle_identifiers or 1,
                
                # Dataset structure defaults
                total_groups=1,
                mean_puzzle_examples=1.0,
                sets=["test"]
            )
            
            print(f"   Inferred vocab_size: {metadata.vocab_size}")
            print(f"   Inferred num_puzzle_identifiers: {metadata.num_puzzle_identifiers}")
            print(f"   Inferred seq_len: {metadata.seq_len}")
            
        except Exception as e:
            print(f"❌ Could not infer metadata from checkpoint: {e}")
            raise
    
    # Create model with correct metadata
    model = create_eval_model(config, metadata)
    
    # Load checkpoint weights
    try:
        state_dict = torch.load(str(checkpoint_path), map_location=device)
        model.load_state_dict(state_dict, assign=True)
    except:
        # Handle torch.compile naming differences
        state_dict = torch.load(str(checkpoint_path), map_location=device)
        cleaned_state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
        model.load_state_dict(cleaned_state_dict, assign=True)
    
    # Set to evaluation mode
    model.eval()
    
    return model, config, metadata


class EvalModelWrapper:
    """Lightweight wrapper for evaluation-only models."""
    
    def __init__(self, model: nn.Module, config: PretrainConfig, metadata: PuzzleDatasetMetadata):
        self.model = model
        self.config = config  
        self.metadata = metadata
        
    def __call__(self, *args, **kwargs):
        """Forward pass."""
        return self.model(*args, **kwargs)
        
    def __getattr__(self, name):
        """Delegate to model."""
        return getattr(self.model, name)


def load_eval_wrapper(checkpoint_path: str, device: str = "cuda") -> EvalModelWrapper:
    """Load model as a convenient wrapper for evaluation."""
    model, config, metadata = load_eval_model(checkpoint_path, device)
    return EvalModelWrapper(model, config, metadata) 