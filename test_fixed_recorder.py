#!/usr/bin/env python3
"""
Test Fixed Recorder

Quick test to verify the fixed HRMRecordingWrapper now captures Q-head logits.
"""

import torch
from pathlib import Path
from eval_utils import load_eval_model
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
from hrm_state_recorder import HRMStateRecorder, HRMRecordingWrapper

def test_fixed_recorder():
    """Test that the fixed recorder captures Q-head logits."""
    print("🧪 Testing Fixed HRM Recorder")
    print("=" * 50)
    
    # Load model
    checkpoint_path = "./checkpoints/sudoku-extreme/checkpoint"
    print(f"Loading model from {checkpoint_path}")
    model, config, metadata = load_eval_model(checkpoint_path)
    
    # Create recorder and wrapper
    recorder = HRMStateRecorder(device="cuda")
    wrapped_model = HRMRecordingWrapper(model, recorder, config.arch)
    
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
    
    # Initialize model state
    with torch.device("cuda"):
        carry = model.initial_carry(batch_gpu)
    
    print(f"\n🔄 Testing recording with fixed wrapper...")
    
    # Record execution
    recorder.clear_traces()
    recorder.start_recording(model)
    
    with torch.inference_mode():
        new_carry, outputs = wrapped_model(carry, batch_gpu, return_keys=[])
    
    recorder.stop_recording()
    
    print(f"\n📊 Recording results:")
    print(f"   Traces recorded: {len(recorder.traces)}")
    
    if recorder.traces:
        trace = recorder.traces[0]
        print(f"   Snapshots in trace: {len(trace.snapshots)}")
        
        # Check if we have Q-head logits
        q_halt_found = False
        q_continue_found = False
        
        for i, snapshot in enumerate(trace.snapshots):
            if snapshot.q_halt_logits is not None:
                q_halt_found = True
                print(f"   ✅ Snapshot {i}: q_halt_logits shape={snapshot.q_halt_logits.shape}")
            if snapshot.q_continue_logits is not None:
                q_continue_found = True
                print(f"   ✅ Snapshot {i}: q_continue_logits shape={snapshot.q_continue_logits.shape}")
        
        if q_halt_found and q_continue_found:
            print(f"\n🎉 SUCCESS: Q-head logits are now being captured!")
        else:
            print(f"\n❌ FAILED: Q-head logits still missing")
            print(f"   q_halt_logits found: {q_halt_found}")
            print(f"   q_continue_logits found: {q_continue_found}")
        
        # Test saving
        print(f"\n💾 Testing trace saving...")
        test_dir = Path("./test_fixed_traces_quick")
        test_dir.mkdir(exist_ok=True)
        
        try:
            recorder.save_traces_with_tensors(
                str(test_dir / "trace_test"), 
                save_format="pt"
            )
            
            # Load and verify
            trace_file = test_dir / "trace_test_trace_0.pt"
            if trace_file.exists():
                trace_data = torch.load(trace_file, map_location='cpu')
                print(f"\n📋 Saved trace contains:")
                for key, value in trace_data.items():
                    if isinstance(value, torch.Tensor):
                        print(f"   {key}: shape={value.shape}, dtype={value.dtype}")
                    else:
                        print(f"   {key}: {type(value)}")
                
                # Check for critical fields
                critical_fields = ['h_states', 'l_states', 'q_halt_logits', 'metadata']
                missing_fields = []
                for field in critical_fields:
                    if field not in trace_data:
                        missing_fields.append(field)
                
                if missing_fields:
                    print(f"\n❌ Missing critical fields: {missing_fields}")
                else:
                    print(f"\n✅ All critical fields present!")
                
            else:
                print(f"\n❌ Trace file not saved: {trace_file}")
                
        except Exception as e:
            print(f"\n❌ Error saving trace: {e}")
    
    else:
        print(f"\n❌ No traces recorded!")

if __name__ == "__main__":
    test_fixed_recorder() 