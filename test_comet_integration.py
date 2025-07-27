#!/usr/bin/env python3
"""
Test Comet ML Integration with HRM State Recorder

This script demonstrates how to:
1. Record HRM states during inference
2. Upload the traces to Comet ML as versioned artifacts
3. Download and analyze traces from Comet ML

Prerequisites:
- Set your Comet API key: export COMET_API_KEY="your-api-key"
- Or use: comet_ml.login() interactively
"""

import torch
from pathlib import Path
import argparse

from hrm_state_recorder import HRMStateRecorder, HRMRecordingWrapper
from eval_utils import load_eval_model
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig


def test_comet_integration(checkpoint_path: str, 
                          project_name: str = "hrm-traces",
                          experiment_name: str = "sudoku-state-recording",
                          num_samples: int = 2, 
                          max_steps: int = 5):
    """Test the full Comet ML integration workflow."""
    
    print("🧪 Testing HRM State Recorder + Comet ML Integration")
    print("=" * 60)
    
    # Step 1: Record HRM states (same as before)
    print("\n📊 Step 1: Recording HRM States")
    print("-" * 30)
    
    # Load model
    model, config, metadata = load_eval_model(checkpoint_path)
    print(f"✅ Loaded model from {checkpoint_path}")
    
    # Create test data
    test_samples = create_test_dataloader(config, num_samples)
    print(f"✅ Created test data ({num_samples} samples)")
    
    # Record states
    recorder = HRMStateRecorder(device="cuda")
    wrapped_model = HRMRecordingWrapper(model, recorder, config.arch)
    
    # Run inference with recording
    set_name, batch, global_batch_size = test_samples[0]
    batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    
    with torch.device("cuda"):
        carry = model.initial_carry(batch)
    
    print(f"🔍 Running inference ({max_steps} max steps)...")
    with torch.inference_mode():
        for step in range(max_steps):
            carry, outputs = wrapped_model(carry, batch, return_keys=[])
            if hasattr(carry, 'halted') and carry.halted.all():
                break
    
    recorder.stop_recording()
    print(f"✅ Recorded {len(recorder.traces)} execution traces")
    
    # Step 2: Upload to Comet ML
    print("\n🚀 Step 2: Uploading to Comet ML")
    print("-" * 30)
    
    # Create Comet experiment
    experiment = HRMStateRecorder.create_comet_experiment(
        project_name=project_name,
        experiment_name=experiment_name,
        tags=["hrm", "state-recording", "sudoku", "inference"]
    )
    
    # Log some experiment metadata
    experiment.log_parameter("model_checkpoint", checkpoint_path)
    experiment.log_parameter("num_samples", num_samples)
    experiment.log_parameter("max_steps", max_steps)
    experiment.log_parameter("model_architecture", config.arch.name)
    
    # Upload traces as artifact
    artifact_version = recorder.upload_traces_to_comet(
        experiment=experiment,
        artifact_name="hrm-sudoku-traces",
        aliases=["latest", "sudoku-experiment", f"samples-{num_samples}"],
        description=f"HRM execution traces from Sudoku model with {num_samples} samples",
        model_checkpoint=checkpoint_path,
        include_tensors=True,
        tensor_format="both"  # Save both PyTorch and NumPy formats
    )
    
    print(f"🎯 Artifact uploaded with version: {artifact_version}")
    print(f"🌐 View experiment: {experiment.url}")
    
    # Step 3: Simulate downloading from Comet ML
    print("\n📥 Step 3: Downloading from Comet ML")
    print("-" * 30)
    
    # Create a new recorder to simulate fresh download
    new_recorder = HRMStateRecorder()
    
    # Download the artifact we just uploaded
    download_result = new_recorder.download_traces_from_comet(
        experiment=experiment,
        artifact_name="hrm-sudoku-traces",
        version_or_alias="latest",
        download_tensors=True,
        local_path="./downloaded_comet_traces"
    )
    
    print(f"✅ Downloaded artifact version: {download_result['version']}")
    print(f"📁 Files downloaded to: {download_result['download_path']}")
    print(f"🧠 Loaded {len(new_recorder.traces)} traces from artifact")
    
    # Step 4: Verify the downloaded data
    print("\n🔍 Step 4: Verifying Downloaded Data")
    print("-" * 30)
    
    original_trace = recorder.get_latest_trace()
    downloaded_trace = new_recorder.get_latest_trace()
    
    if original_trace and downloaded_trace:
        orig_summary = original_trace.get_execution_summary()
        down_summary = downloaded_trace.get_execution_summary()
        
        print(f"Original trace steps: {orig_summary['total_steps']}")
        print(f"Downloaded trace steps: {down_summary['total_steps']}")
        print(f"Metadata matches: {orig_summary == down_summary}")
    
    # Step 5: Demonstrate tensor loading
    print("\n🧠 Step 5: Loading Full Tensor Data")
    print("-" * 30)
    
    # Load full tensor data from downloaded files
    tensor_files = download_result["files"].get("tensors", [])
    if tensor_files:
        # Load PyTorch tensor file
        pt_files = [f for f in tensor_files if f.endswith('.pt')]
        if pt_files:
            data = torch.load(pt_files[0], map_location="cpu")
            h_states = data.get('h_states')
            l_states = data.get('l_states')
            
            if h_states is not None:
                print(f"✅ H-states shape: {h_states.shape}")
                print(f"   H-state magnitude: {h_states.abs().mean():.4f}")
            
            if l_states is not None:
                print(f"✅ L-states shape: {l_states.shape}")
                print(f"   L-state magnitude: {l_states.abs().mean():.4f}")
    
    # Clean up
    experiment.end()
    
    print("\n🎉 Comet ML Integration Test Completed Successfully!")
    print(f"   🔗 Experiment URL: {experiment.url}")
    print(f"   📦 Artifact: hrm-sudoku-traces v{artifact_version}")
    print(f"   💾 Local traces: ./downloaded_comet_traces")
    
    return experiment, download_result


def create_test_dataloader(config, num_samples: int = 5):
    """Create a small test dataloader."""
    dataset = PuzzleDataset(PuzzleDatasetConfig(
        seed=config.seed,
        dataset_path=config.data_path,
        rank=0,
        num_replicas=1,
        test_set_mode=True,
        epochs_per_iter=1,
        global_batch_size=num_samples,
    ), split="test")
    
    samples = []
    for i, (set_name, batch, global_batch_size) in enumerate(dataset):
        if i >= 1:
            break
        samples.append((set_name, batch, global_batch_size))
    
    return samples


def demo_artifact_versioning():
    """Demonstrate artifact versioning and aliases with multiple uploads."""
    
    print("\n🔄 Demonstrating Artifact Versioning")
    print("=" * 50)
    
    # This would show how to create multiple versions of the same artifact
    # with different aliases and track the evolution over time
    
    experiment = HRMStateRecorder.create_comet_experiment(
        project_name="hrm-versioning-demo",
        experiment_name="artifact-versioning-test"
    )
    
    # Simulate different experimental runs
    scenarios = [
        {"alias": "baseline", "description": "Initial baseline traces"},
        {"alias": "optimized", "description": "Traces with optimized hyperparameters"},
        {"alias": "production", "description": "Production-ready traces"}
    ]
    
    print("📦 This demo would create multiple artifact versions:")
    for i, scenario in enumerate(scenarios):
        print(f"   v{i+1}.0.0: {scenario['description']} (alias: {scenario['alias']})")
    
    print("💡 Each version would be accessible by:")
    print("   - Exact version: experiment.get_artifact('hrm-traces', version='1.0.0')")
    print("   - Alias: experiment.get_artifact('hrm-traces', version_or_alias='baseline')")
    print("   - Latest: experiment.get_artifact('hrm-traces')")
    
    experiment.end()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test Comet ML integration with HRM state recorder")
    parser.add_argument("--checkpoint", 
                       default="./checkpoints/sudoku-extreme/checkpoint",
                       help="Path to checkpoint file")
    parser.add_argument("--project", 
                       default="hrm-traces",
                       help="Comet project name")
    parser.add_argument("--experiment", 
                       default="sudoku-state-recording",
                       help="Comet experiment name")
    parser.add_argument("--samples", type=int, default=2,
                       help="Number of samples to test")
    parser.add_argument("--max-steps", type=int, default=5,
                       help="Maximum inference steps")
    parser.add_argument("--demo-versioning", action="store_true",
                       help="Show artifact versioning demo")
    
    args = parser.parse_args()
    
    try:
        # Check if Comet ML is available
        import comet_ml
        print("✅ Comet ML is available")
        
        # Check for API key
        if not comet_ml.config.get_api_key():
            print("⚠️  No Comet API key found. Please set COMET_API_KEY environment variable")
            print("   or run: comet_ml.login()")
            return 1
        
    except ImportError:
        print("❌ Comet ML not installed. Run: pip install comet_ml")
        return 1
    
    if args.demo_versioning:
        demo_artifact_versioning()
        return 0
    
    if not Path(args.checkpoint).exists():
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        print("   Please run download_models.py first or specify correct checkpoint path")
        return 1
    
    try:
        experiment, download_result = test_comet_integration(
            checkpoint_path=args.checkpoint,
            project_name=args.project,
            experiment_name=args.experiment,
            num_samples=args.samples,
            max_steps=args.max_steps
        )
        
        print(f"\n🎯 Next Steps:")
        print(f"   1. Visit your experiment: {experiment.url}")
        print(f"   2. Explore the Artifacts tab to see your uploaded traces")
        print(f"   3. Use the downloaded traces: {download_result['download_path']}")
        print(f"   4. Try different analysis scripts on the tensor data")
        
        return 0
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main()) 