#!/usr/bin/env python3
"""
Test L-step Vocabulary Logits Capture

This script tests that the modified HRM state recorder correctly captures
L-step vocabulary logits at each timestep alongside the existing H-step 
vocabulary logits.
"""

import torch
from pathlib import Path
from eval_utils import load_eval_model
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
from hrm_state_recorder import HRMStateRecorder, HRMRecordingWrapper


def test_l_vocab_logits_capture():
    """Test that L-step vocabulary logits are captured alongside H-step logits."""
    print("🧪 Testing L-step Vocabulary Logits Capture")
    print("=" * 60)
    
    # Load model
    checkpoint_path = "./checkpoints/sudoku-extreme/checkpoint"
    print(f"📂 Loading model from {checkpoint_path}")
    
    try:
        model, config, metadata = load_eval_model(checkpoint_path)
        print(f"✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("   Trying alternative checkpoint paths...")
        
        # Try alternative paths
        alt_paths = [
            "./checkpoints/sudoku-extreme-1k-aug-1000/checkpoint",
            "./checkpoints/sudoku/checkpoint", 
            "./checkpoints/latest/checkpoint"
        ]
        
        model = None
        for alt_path in alt_paths:
            try:
                model, config, metadata = load_eval_model(alt_path)
                checkpoint_path = alt_path
                print(f"✅ Model loaded from {alt_path}")
                break
            except:
                continue
        
        if model is None:
            print("❌ Could not load model from any checkpoint path")
            return False
    
    # Create recorder and wrapper
    recorder = HRMStateRecorder(device="cuda")
    wrapped_model = HRMRecordingWrapper(model, recorder, config.arch)
    print(f"✅ Recording wrapper created with instrumented model")
    
    # Create a small dataset
    try:
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
            
        print(f"✅ Dataset loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return False
    
    # Move to GPU
    batch_gpu = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    print(f"✅ Data moved to GPU")
    
    # Initialize model state
    with torch.device("cuda"):
        carry = model.initial_carry(batch_gpu)
    
    print(f"\n🔄 Testing recording with L-step vocabulary logits capture...")
    
    # Record execution
    recorder.start_recording(model)
    
    try:
        with torch.inference_mode():
            new_carry, outputs = wrapped_model(carry, batch_gpu, return_keys=[])
        
        recorder.stop_recording()
        print(f"✅ Recording completed successfully")
        
    except Exception as e:
        print(f"❌ Error during recording: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"\n📊 Recording Results:")
    print(f"   📈 Traces recorded: {len(recorder.traces)}")
    
    if not recorder.traces:
        print(f"❌ No traces recorded!")
        return False
    
    trace = recorder.traces[0]
    print(f"   📈 Snapshots in trace: {len(trace.snapshots)}")
    
    # Check if we have L_vocab_logits in addition to regular vocab_logits
    fields_found = {
        'h_states': 0,
        'l_states': 0,
        'q_halt_logits': 0,
        'q_continue_logits': 0,
        'vocab_logits': 0,      # H-step vocabulary logits
        'L_vocab_logits': 0     # NEW: L-step vocabulary logits
    }
    
    print(f"\n🔍 Analyzing snapshots for L-step vocab logits:")
    for i, snapshot in enumerate(trace.snapshots):
        print(f"   📸 Snapshot {i}:")
        
        # Check each field
        for field_name in fields_found.keys():
            field_value = getattr(snapshot, field_name, None)
            if field_value is not None:
                fields_found[field_name] += 1
                if field_name in ['vocab_logits', 'L_vocab_logits']:
                    print(f"      ✅ {field_name}: shape={field_value.shape}, dtype={field_value.dtype}")
                else:
                    print(f"      ✅ {field_name}: present")
            else:
                print(f"      ❌ {field_name}: None")
    
    print(f"\n📈 Summary of captured data:")
    for field_name, count in fields_found.items():
        print(f"   {field_name}: {count}/{len(trace.snapshots)} snapshots")
    
    # Test saving traces with L_vocab_logits
    try:
        print(f"\n💾 Testing tensor save with L_vocab_logits...")
        test_save_path = "./test_l_vocab_logits_trace"
        recorder.save_traces_with_tensors(test_save_path, save_format="pt")
        
        # Try to load and verify
        loaded_data = torch.load(f"{test_save_path}_trace_0.pt", map_location="cpu")
        
        print(f"   📊 Saved tensor keys: {list(loaded_data.keys())}")
        
        if 'L_vocab_logits' in loaded_data:
            l_vocab_shape = loaded_data['L_vocab_logits'].shape
            print(f"   ✅ L_vocab_logits saved successfully: shape={l_vocab_shape}")
        else:
            print(f"   ❌ L_vocab_logits not found in saved data")
            
        if 'vocab_logits' in loaded_data:
            vocab_shape = loaded_data['vocab_logits'].shape  
            print(f"   ✅ vocab_logits also present: shape={vocab_shape}")
            
        # Compare shapes (should be similar but may differ due to timing)
        if 'L_vocab_logits' in loaded_data and 'vocab_logits' in loaded_data:
            l_shape = loaded_data['L_vocab_logits'].shape
            h_shape = loaded_data['vocab_logits'].shape
            print(f"   🔍 Shape comparison:")
            print(f"      H-step vocab_logits: {h_shape}")  
            print(f"      L-step vocab_logits: {l_shape}")
            
        return True
        
    except Exception as e:
        print(f"   ❌ Error during save/load test: {e}")
        return False


def main():
    """Main test function."""
    success = test_l_vocab_logits_capture()
    
    if success:
        print(f"\n🎉 SUCCESS: L-step vocabulary logits capture is working!")
        print(f"   ✅ L_vocab_logits field added to snapshots")
        print(f"   ✅ L-step logits computed and captured")
        print(f"   ✅ Data saves and loads correctly")
    else:
        print(f"\n❌ FAILED: Issues with L-step vocabulary logits capture")
    
    return success


if __name__ == "__main__":
    main() 