#!/usr/bin/env python3
"""
Debug HRM Model Outputs

This script investigates what data is actually available in the HRM model
outputs and carry state to understand where q_halt_logits should come from.
"""

import torch
from pathlib import Path
from eval_utils import load_eval_model
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig

def inspect_model_outputs():
    """Inspect HRM model outputs to find Q-head data."""
    print("🔍 Debugging HRM Model Outputs")
    print("=" * 50)
    
    # Load model
    checkpoint_path = "./checkpoints/sudoku-extreme/checkpoint"
    print(f"Loading model from {checkpoint_path}")
    model, config, metadata = load_eval_model(checkpoint_path)
    
    # Create a small dataset
    dataset = PuzzleDataset(PuzzleDatasetConfig(
        seed=42,
        dataset_path=config.data_path,
        rank=0,
        num_replicas=1,
        test_set_mode=False,
        epochs_per_iter=1,
        global_batch_size=1,
    ), split="train")
    
    # Get one batch
    for batch_data in dataset:
        set_name, batch, global_batch_size = batch_data
        break
    
    # Move to GPU
    batch_gpu = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    
    print("\n📊 Batch structure:")
    for key, value in batch_gpu.items():
        if isinstance(value, torch.Tensor):
            print(f"   {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"   {key}: {type(value)} = {value}")
    
    # Initialize model state
    with torch.device("cuda"):
        carry = model.initial_carry(batch_gpu)
    
    print(f"\n🔄 Initial carry structure:")
    print(f"   carry type: {type(carry)}")
    for attr in dir(carry):
        if not attr.startswith('_'):
            try:
                value = getattr(carry, attr)
                if isinstance(value, torch.Tensor):
                    print(f"   {attr}: shape={value.shape}, dtype={value.dtype}")
                elif hasattr(value, '__dict__'):
                    print(f"   {attr}: {type(value)} (nested object)")
                    # Look inside nested objects
                    for sub_attr in dir(value):
                        if not sub_attr.startswith('_'):
                            try:
                                sub_value = getattr(value, sub_attr)
                                if isinstance(sub_value, torch.Tensor):
                                    print(f"      {attr}.{sub_attr}: shape={sub_value.shape}, dtype={sub_value.dtype}")
                            except:
                                pass
                else:
                    print(f"   {attr}: {type(value)} = {value}")
            except:
                pass
    
    # Run one step
    print(f"\n⚡ Running model forward pass...")
    with torch.inference_mode():
        new_carry, loss, metrics, preds, all_finish = model(carry=carry, batch=batch_gpu, return_keys=[])
    
    print(f"\n📤 Model outputs:")
    print(f"   loss: {type(loss)} = {loss}")
    print(f"   metrics: {type(metrics)}")
    if isinstance(metrics, dict):
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                print(f"      {key}: shape={value.shape}, dtype={value.dtype}, value={value}")
            else:
                print(f"      {key}: {type(value)} = {value}")
    
    print(f"   preds: {type(preds)}")
    if isinstance(preds, dict):
        for key, value in preds.items():
            if isinstance(value, torch.Tensor):
                print(f"      {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"      {key}: {type(value)} = {value}")
    else:
        print(f"      preds value: {preds}")
    
    print(f"   all_finish: {type(all_finish)} = {all_finish}")
    
    print(f"\n🔄 New carry structure:")
    print(f"   new_carry type: {type(new_carry)}")
    for attr in dir(new_carry):
        if not attr.startswith('_'):
            try:
                value = getattr(new_carry, attr)
                if isinstance(value, torch.Tensor):
                    print(f"   {attr}: shape={value.shape}, dtype={value.dtype}")
                    if 'halt' in attr.lower() or 'q_' in attr.lower():
                        print(f"      *** POTENTIAL Q-HEAD DATA: {attr} = {value}")
                elif hasattr(value, '__dict__'):
                    print(f"   {attr}: {type(value)} (nested object)")
                    # Look inside nested objects for Q-head data
                    for sub_attr in dir(value):
                        if not sub_attr.startswith('_'):
                            try:
                                sub_value = getattr(value, sub_attr)
                                if isinstance(sub_value, torch.Tensor):
                                    print(f"      {attr}.{sub_attr}: shape={sub_value.shape}, dtype={sub_value.dtype}")
                                    if 'halt' in sub_attr.lower() or 'q_' in sub_attr.lower():
                                        print(f"         *** POTENTIAL Q-HEAD DATA: {attr}.{sub_attr} = {sub_value}")
                            except:
                                pass
                else:
                    print(f"   {attr}: {type(value)} = {value}")
            except:
                pass
    
    # Check if there are any model components we can access directly
    print(f"\n🧠 Model structure:")
    print(f"   model type: {type(model)}")
    for attr in dir(model):
        if not attr.startswith('_') and not callable(getattr(model, attr)):
            try:
                value = getattr(model, attr)
                print(f"   {attr}: {type(value)}")
                if hasattr(value, '__dict__'):
                    # Look for Q-head components
                    for sub_attr in dir(value):
                        if 'halt' in sub_attr.lower() or 'q_' in sub_attr.lower():
                            sub_value = getattr(value, sub_attr)
                            print(f"      *** POTENTIAL Q-HEAD COMPONENT: {attr}.{sub_attr}: {type(sub_value)}")
            except:
                pass

if __name__ == "__main__":
    inspect_model_outputs() 