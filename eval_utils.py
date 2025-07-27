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
    
    # Try to load metadata
    try:
        import json
        with open(Path(config.data_path) / "metadata.json", "r") as f:
            metadata_dict = json.load(f)
            metadata = PuzzleDatasetMetadata(**metadata_dict)
    except:
        # Create minimal metadata for testing with all required fields
        print("⚠️  Could not load metadata, using defaults")
        metadata = PuzzleDatasetMetadata(
            # Required token IDs
            pad_id=0,
            ignore_label_id=-100,  # Standard ignore label
            blank_identifier_id=1,
            
            # Model dimensions  
            vocab_size=100,
            seq_len=200,
            num_puzzle_identifiers=1000,
            
            # Dataset structure
            total_groups=1,
            mean_puzzle_examples=1.0,
            sets=["test"]
        )
    
    # Create model (no optimizers!)
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