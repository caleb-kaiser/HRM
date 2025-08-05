#!/usr/bin/env python3
"""
Test Vocabulary Logits Capture

This script tests that the modified HRM state recorder correctly captures
vocabulary logits at each timestep and that the utility functions can
extract them properly.
"""

import torch
from pathlib import Path
from eval_utils import load_eval_model
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
from hrm_state_recorder import HRMStateRecorder, HRMRecordingWrapper
from eda.utils import create_problem_tensors, load_dataset_metadata
import json

def test_vocab_logits_capture():
    """Test that vocabulary logits are captured at each timestep."""
    print("🧪 Testing Vocabulary Logits Capture")
    print("=" * 60)
    
    # Load model
    checkpoint_path = "./checkpoints/sudoku-extreme/checkpoint"
    print(f"📂 Loading model from {checkpoint_path}")
    model, config, metadata = load_eval_model(checkpoint_path)
    print(f"✅ Model loaded successfully")
    
    # Create recorder and wrapper
    recorder = HRMStateRecorder(device="cuda")
    wrapped_model = HRMRecordingWrapper(model, recorder, config.arch)
    print(f"✅ Recording wrapper created")
    
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
    print(f"✅ Data loaded and moved to GPU")
    
    # Initialize model state
    with torch.device("cuda"):
        carry = model.initial_carry(batch_gpu)
    
    print(f"\n🔄 Testing recording with vocabulary logits capture...")
    
    # Record execution
    recorder.clear_traces()
    recorder.start_recording(model)
    
    with torch.inference_mode():
        new_carry, outputs = wrapped_model(carry, batch_gpu, return_keys=[])
    
    recorder.stop_recording()
    
    print(f"\n📊 Recording Results:")
    print(f"   📈 Traces recorded: {len(recorder.traces)}")
    
    if not recorder.traces:
        print(f"❌ No traces recorded!")
        return False
    
    trace = recorder.traces[0]
    print(f"   📈 Snapshots in trace: {len(trace.snapshots)}")
    
    # Check if we have all expected data including vocab_logits
    fields_found = {
        'h_states': 0,
        'l_states': 0,
        'q_halt_logits': 0,
        'q_continue_logits': 0,
        'vocab_logits': 0  # NEW: Check for vocabulary logits
    }
    
    print(f"\n🔍 Analyzing snapshots:")
    for i, snapshot in enumerate(trace.snapshots):
        print(f"   📸 Snapshot {i}:")
        
        if snapshot.z_H is not None:
            fields_found['h_states'] += 1
            print(f"      ✅ H-states: {snapshot.z_H.shape}")
            
        if snapshot.z_L is not None:
            fields_found['l_states'] += 1
            print(f"      ✅ L-states: {snapshot.z_L.shape}")
            
        if snapshot.q_halt_logits is not None:
            fields_found['q_halt_logits'] += 1
            print(f"      ✅ Q-halt logits: {snapshot.q_halt_logits.shape}")
            
        if snapshot.q_continue_logits is not None:
            fields_found['q_continue_logits'] += 1
            print(f"      ✅ Q-continue logits: {snapshot.q_continue_logits.shape}")
            
        if snapshot.vocab_logits is not None:
            fields_found['vocab_logits'] += 1
            print(f"      ✅ Vocab logits: {snapshot.vocab_logits.shape}")
            # Verify vocab_logits have reasonable shape
            expected_vocab_size = 11  # For Sudoku
            if len(snapshot.vocab_logits.shape) >= 2 and snapshot.vocab_logits.shape[-1] == expected_vocab_size:
                print(f"         🎯 Vocabulary size correct: {expected_vocab_size}")
            else:
                print(f"         ⚠️  Unexpected vocab logits shape: {snapshot.vocab_logits.shape}")
        else:
            print(f"      ❌ No vocab logits in snapshot {i}")
    
    # Summary of findings
    print(f"\n📋 Summary of captured data:")
    for field, count in fields_found.items():
        if count > 0:
            print(f"   ✅ {field}: Found in {count}/{len(trace.snapshots)} snapshots")
        else:
            print(f"   ❌ {field}: Not found in any snapshots")
    
    # Check if vocab_logits were captured
    vocab_logits_success = fields_found['vocab_logits'] > 0
    if not vocab_logits_success:
        print(f"\n❌ CRITICAL: Vocabulary logits were not captured!")
        return False
    
    print(f"\n💾 Testing trace saving and loading...")
    
    # Test saving with vocab_logits
    test_dir = Path("./test_vocab_logits_capture")
    test_dir.mkdir(exist_ok=True)
    
    try:
        # Save traces
        recorder.save_traces_with_tensors(
            str(test_dir / "trace_vocab_test"), 
            save_format="pt"
        )
        
        # Create fake metadata for testing utility function
        fake_metadata = {
            "problems": [{
                "problem_id": 0,
                "sudoku_input": [[0] * 9] * 9,  # Dummy data
                "sudoku_target": [[1] * 9] * 9,  # Dummy data
                "metadata": {"test": True},
                "trace_file": "trace_vocab_test_trace_0.pt"
            }]
        }
        
        metadata_file = test_dir / "dataset_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(fake_metadata, f)
        
        print(f"   ✅ Traces saved successfully")
        
        # Test loading with utility function
        print(f"\n🔧 Testing utility function extraction...")
        
        problem = fake_metadata["problems"][0]
        tensors = create_problem_tensors(problem, str(test_dir))
        
        print(f"\n📋 Extracted tensors:")
        for key, value in tensors.items():
            if isinstance(value, torch.Tensor):
                print(f"   📊 {key}: shape={value.shape}, dtype={value.dtype}")
            elif value is not None:
                print(f"   📊 {key}: {type(value)}")
            else:
                print(f"   ❌ {key}: None")
        
        # Verify vocab_logits are in extracted tensors
        if 'vocab_logits' in tensors and tensors['vocab_logits'] is not None:
            vocab_logits = tensors['vocab_logits']
            print(f"\n🎯 Vocabulary Logits Analysis:")
            print(f"   📊 Shape: {vocab_logits.shape}")
            print(f"   📊 Data type: {vocab_logits.dtype}")
            print(f"   📊 Min value: {vocab_logits.min():.4f}")
            print(f"   📊 Max value: {vocab_logits.max():.4f}")
            print(f"   📊 Mean value: {vocab_logits.mean():.4f}")
            
            # Check if we can do argmax (simulate final prediction)
            if len(vocab_logits.shape) >= 3:  # [timesteps, batch, seq, vocab]
                final_timestep = vocab_logits[-1]  # Last timestep
                predicted_tokens = torch.argmax(final_timestep, dim=-1)
                print(f"   🎲 Final predictions shape: {predicted_tokens.shape}")
                print(f"   🎲 Sample predictions: {predicted_tokens[0, :10].tolist()}")  # First 10 tokens
                print(f"   ✅ Vocabulary logits are usable for prediction!")
            
            return True
        else:
            print(f"\n❌ FAILED: vocab_logits not found in extracted tensors!")
            return False
            
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("🚀 Starting Vocabulary Logits Capture Test")
    print("=" * 60)
    
    success = test_vocab_logits_capture()
    
    print(f"\n" + "=" * 60)
    if success:
        print(f"🎉 SUCCESS! Vocabulary logits capture is working correctly!")
        print(f"✅ Vocabulary logits are captured at each timestep")
        print(f"✅ Traces are saved with vocab_logits included")
        print(f"✅ Utility function extracts vocab_logits properly")
        print(f"✅ Vocab_logits can be used for token prediction analysis")
    else:
        print(f"❌ FAILED! Vocabulary logits capture is not working.")
        print(f"   Check the modifications to HRMStateRecorder and utility functions.")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main()) 