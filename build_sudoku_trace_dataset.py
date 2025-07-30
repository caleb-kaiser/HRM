#!/usr/bin/env python3
"""
Build Sudoku Trace Dataset

This script generates a comprehensive dataset of HRM execution traces
from the entire Sudoku training dataset. Each trace is associated with
the specific Sudoku problem it came from for advanced analysis.

Features:
- Processes entire training dataset
- Records hidden states for each problem
- Associates traces with original problems
- Efficient batch processing
- Uploads to Comet ML as artifact
- Memory-efficient processing
"""

import torch
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass, asdict

from hrm_state_recorder import HRMStateRecorder, HRMRecordingWrapper
from eval_utils import load_eval_model
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig


@dataclass
class ProblemTrace:
    """Associates a trace with its original Sudoku problem."""
    problem_id: int
    sudoku_input: List[List[int]]  # 9x9 Sudoku grid
    sudoku_target: List[List[int]]  # 9x9 target solution
    metadata: Dict[str, Any]
    trace_file: str  # Path to saved trace tensors


@dataclass
class TraceDatasetConfig:
    """Configuration for trace dataset generation."""
    checkpoint_path: str
    output_dir: str = "./sudoku_trace_dataset"
    batch_size: int = 8
    max_problems: Optional[int] = None  # None = process all
    max_steps_per_problem: int = 50
    tensor_format: str = "pt"  # "pt", "npz", or "both"
    include_failed: bool = False  # Include traces where model failed
    comet_project: str = "hrm-trace-datasets"
    comet_artifact_name: str = "sudoku-training-traces"


def extract_sudoku_from_batch(batch: Dict[str, torch.Tensor], idx: int) -> Tuple[List[List[int]], List[List[int]]]:
    """Extract a single Sudoku problem and solution from a batch."""
    try:
        # Debug: Print available keys for troubleshooting
        if idx == 0:  # Only print for first sample to avoid spam
            print(f"   Available batch keys: {list(batch.keys())}")
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    print(f"   {key}: shape {value.shape}, dtype {value.dtype}")
        
        # Get the input for this sample
        inputs = batch['inputs'][idx].cpu().numpy()  # [seq_len]
        
        # Try to get targets - handle different possible key names
        targets = None
        target_keys = ['targets', 'labels', 'target', 'solutions', 'outputs']
        
        for target_key in target_keys:
            if target_key in batch:
                targets = batch[target_key][idx].cpu().numpy()
                break
        
        # Convert from flattened representation back to 9x9 grid
        # Extract input grid (first 81 positions, assuming 0-10 vocabulary)
        # Filter out special tokens (padding, etc.) and keep only 0-9
        valid_inputs = inputs[:81]  # Take first 81 tokens
        valid_inputs = np.clip(valid_inputs, 0, 9)  # Ensure values are 0-9
        input_grid = valid_inputs.reshape(9, 9).tolist()
        
        # Extract target grid
        if targets is not None and len(targets) >= 81:
            valid_targets = targets[:81]
            valid_targets = np.clip(valid_targets, 0, 9)  # Ensure values are 0-9
            target_grid = valid_targets.reshape(9, 9).tolist()
        else:
            # Fallback: try to extract solution from input sequence
            # Some datasets encode the solution after the input
            if len(inputs) >= 162:  # Input + solution
                solution_part = inputs[81:162]
                solution_part = np.clip(solution_part, 0, 9)
                target_grid = solution_part.reshape(9, 9).tolist()
            else:
                # Last resort: use input as target (for now)
                target_grid = input_grid.copy()
                if idx == 0:  # Only warn once
                    print(f"   Warning: No targets found, using input as fallback")
        
        return input_grid, target_grid
        
    except Exception as e:
        # More detailed error info
        print(f"   Error in extract_sudoku_from_batch for sample {idx}: {e}")
        print(f"   Batch keys: {list(batch.keys()) if batch else 'None'}")
        if 'inputs' in batch:
            print(f"   Input shape: {batch['inputs'].shape}")
        raise


def extract_prediction_from_outputs(outputs: Dict[str, torch.Tensor], seq_len: int = 81) -> Optional[List[List[int]]]:
    """Extract the final Sudoku prediction from model outputs."""
    try:
        # The outputs typically contain 'preds' or 'logits'
        if 'preds' in outputs:
            predictions = outputs['preds']
        elif 'logits' in outputs:
            # Convert logits to predictions
            predictions = torch.argmax(outputs['logits'], dim=-1)
        else:
            # Fallback: look for any tensor that could be predictions
            for key, value in outputs.items():
                if isinstance(value, torch.Tensor) and value.numel() > 0:
                    if len(value.shape) >= 2:  # Has batch and sequence dimensions
                        predictions = value
                        if predictions.dtype == torch.float:
                            predictions = torch.argmax(predictions, dim=-1)
                        break
            else:
                return None
        
        # Extract first batch item and first seq_len tokens
        if predictions.dim() >= 2:
            pred_sequence = predictions[0, :seq_len].cpu().numpy()
        else:
            pred_sequence = predictions[:seq_len].cpu().numpy()
        
        # Ensure values are in valid range (0-9)
        pred_sequence = np.clip(pred_sequence, 0, 9)
        
        # Reshape to 9x9 grid
        if len(pred_sequence) >= 81:
            prediction_grid = pred_sequence[:81].reshape(9, 9).tolist()
            return prediction_grid
        else:
            return None
            
    except Exception as e:
        print(f"   Warning: Could not extract prediction: {e}")
        return None


def compare_sudoku_solutions(prediction: Optional[List[List[int]]], target: List[List[int]]) -> Dict[str, Any]:
    """Compare predicted Sudoku solution with target."""
    if prediction is None:
        return {
            "is_exact_match": False,
            "accuracy": 0.0,
            "correct_cells": 0,
            "total_cells": 81,
            "valid_sudoku": False
        }
    
    # Convert to numpy for easier comparison
    pred_array = np.array(prediction)
    target_array = np.array(target)
    
    # Calculate cell-wise accuracy
    correct_mask = (pred_array == target_array)
    correct_cells = np.sum(correct_mask)
    total_cells = pred_array.size
    accuracy = correct_cells / total_cells
    
    # Check for exact match
    is_exact_match = (correct_cells == total_cells)
    
    # Basic Sudoku validity check
    valid_sudoku = is_valid_sudoku(prediction)
    
    return {
        "is_exact_match": is_exact_match,
        "accuracy": float(accuracy),
        "correct_cells": int(correct_cells),
        "total_cells": int(total_cells),
        "valid_sudoku": valid_sudoku
    }


def is_valid_sudoku(grid: List[List[int]]) -> bool:
    """Check if a Sudoku grid is valid (no duplicates in rows/cols/boxes)."""
    try:
        grid_array = np.array(grid)
        
        # Check rows
        for row in grid_array:
            non_zero = row[row != 0]
            if len(non_zero) != len(np.unique(non_zero)):
                return False
        
        # Check columns  
        for col in range(9):
            column = grid_array[:, col]
            non_zero = column[column != 0]
            if len(non_zero) != len(np.unique(non_zero)):
                return False
        
        # Check 3x3 boxes
        for box_row in range(3):
            for box_col in range(3):
                box = grid_array[box_row*3:(box_row+1)*3, box_col*3:(box_col+1)*3].flatten()
                non_zero = box[box != 0]
                if len(non_zero) != len(np.unique(non_zero)):
                    return False
        
        return True
        
    except Exception:
        return False


def create_sudoku_dataloader(config, dataset_config: TraceDatasetConfig):
    """Create dataloader for the training dataset."""
    try:
        dataset = PuzzleDataset(PuzzleDatasetConfig(
            seed=config.seed,
            dataset_path=config.data_path,
            rank=0,
            num_replicas=1,
            test_set_mode=False,  # Use training set
            epochs_per_iter=1,
            global_batch_size=dataset_config.batch_size,
        ), split="train")
        
        return dataset
    except Exception as e:
        print(f"❌ Failed to create dataloader: {e}")
        raise


def process_batch(model, wrapped_model, batch: Dict[str, torch.Tensor], 
                 dataset_config: TraceDatasetConfig, 
                 problem_offset: int) -> List[ProblemTrace]:
    """Process a single batch and return problem traces."""
    batch_size = batch['inputs'].shape[0]
    problem_traces = []
    
    # Move batch to GPU
    batch_gpu = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    
    for sample_idx in range(batch_size):
        problem_id = problem_offset + sample_idx
        
        try:
            # Extract single sample
            single_batch = {
                k: v[sample_idx:sample_idx+1] if isinstance(v, torch.Tensor) else v 
                for k, v in batch_gpu.items()
            }
            
            # Extract Sudoku problem
            try:
                sudoku_input, sudoku_target = extract_sudoku_from_batch(batch, sample_idx)
            except Exception as e:
                print(f"⚠️  Error extracting Sudoku from problem {problem_id}: {e}")
                continue
            
            # Initialize model state
            with torch.device("cuda"):
                carry = model.initial_carry(single_batch)
            
            # Run inference with recording
            wrapped_model.recorder.clear_traces()  # Clear previous traces
            wrapped_model.recorder.start_recording(model)
            
            final_outputs = None
            with torch.inference_mode():
                step = 0
                halted = False
                
                while step < dataset_config.max_steps_per_problem and not halted:
                    carry, outputs = wrapped_model(carry, single_batch, return_keys=[])
                    final_outputs = outputs  # Keep the last outputs for prediction extraction
                    
                    # Check if halted
                    if hasattr(carry, 'halted') and carry.halted.all():
                        halted = True
                    
                    step += 1
            
            wrapped_model.recorder.stop_recording()
            
            # Extract final prediction and check correctness
            final_prediction = extract_prediction_from_outputs(final_outputs) if final_outputs else None
            correctness_info = compare_sudoku_solutions(final_prediction, sudoku_target)
            
            # Check if we should include this trace
            if not dataset_config.include_failed and not halted:
                # Model didn't halt - might be a failed solve
                continue
            
            # Save trace tensors
            trace_filename = f"trace_{problem_id:06d}"
            trace_file_path = Path(dataset_config.output_dir) / "traces" / f"{trace_filename}.{dataset_config.tensor_format}"
            
            # Create problem trace metadata
            problem_trace = ProblemTrace(
                problem_id=problem_id,
                sudoku_input=sudoku_input,
                sudoku_target=sudoku_target,
                metadata={
                    "steps_taken": step,
                    "halted": halted,
                    "batch_id": problem_offset // dataset_config.batch_size,
                    "sample_in_batch": sample_idx,
                    # Correctness tracking
                    "is_correct": correctness_info["is_exact_match"],
                    "accuracy": correctness_info["accuracy"],
                    "correct_cells": correctness_info["correct_cells"],
                    "valid_sudoku": correctness_info["valid_sudoku"],
                    "final_prediction": final_prediction,
                },
                trace_file=str(trace_file_path)
            )
            
            problem_traces.append(problem_trace)
            
        except Exception as e:
            print(f"⚠️  Error processing problem {problem_id}: {e}")
            continue
    
    return problem_traces


def save_batch_traces(recorder: HRMStateRecorder, problem_traces: List[ProblemTrace], 
                     dataset_config: TraceDatasetConfig):
    """Save traces for a batch of problems."""
    if not problem_traces:
        return
    
    output_dir = Path(dataset_config.output_dir)
    traces_dir = output_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    
    # Save individual trace files
    for i, problem_trace in enumerate(problem_traces):
        if i < len(recorder.traces):
            trace = recorder.traces[i]
            
            # Create a temporary recorder with just this trace
            temp_recorder = HRMStateRecorder()
            temp_recorder.traces = [trace]
            
            # Save trace tensors
            trace_base = Path(problem_trace.trace_file).with_suffix('')
            temp_recorder.save_traces_with_tensors(
                str(trace_base), 
                save_format=dataset_config.tensor_format
            )


def build_trace_dataset(dataset_config: TraceDatasetConfig) -> Dict[str, Any]:
    """Build the complete trace dataset."""
    print("🏗️  Building Sudoku Trace Dataset")
    print("=" * 50)
    print(f"📁 Output directory: {dataset_config.output_dir}")
    print(f"🔢 Batch size: {dataset_config.batch_size}")
    print(f"📊 Max problems: {dataset_config.max_problems or 'All'}")
    print(f"⚡ Max steps per problem: {dataset_config.max_steps_per_problem}")
    print(f"💾 Tensor format: {dataset_config.tensor_format}")
    
    # Create output directory
    output_dir = Path(dataset_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"\n🤖 Loading model from {dataset_config.checkpoint_path}")
    model, config, metadata = load_eval_model(dataset_config.checkpoint_path)
    print(f"✅ Model loaded successfully")
    
    # Create dataloader
    print(f"\n📊 Creating dataloader...")
    dataloader = create_sudoku_dataloader(config, dataset_config)
    print(f"✅ Dataloader created")
    
    # Setup recording
    recorder = HRMStateRecorder(device="cuda")
    wrapped_model = HRMRecordingWrapper(model, recorder, config.arch)
    
    # Process dataset
    all_problem_traces = []
    problem_offset = 0
    processed_batches = 0
    
    print(f"\n🔄 Processing dataset...")
    
    try:
        for batch_data in tqdm(dataloader, desc="Processing batches"):
            set_name, batch, global_batch_size = batch_data
            
            # Process this batch
            problem_traces = process_batch(
                model, wrapped_model, batch, dataset_config, problem_offset
            )
            
            # Save traces for this batch
            save_batch_traces(recorder, problem_traces, dataset_config)
            
            # Update tracking
            all_problem_traces.extend(problem_traces)
            problem_offset += batch['inputs'].shape[0]
            processed_batches += 1
            
            # Check if we've hit our limit
            if (dataset_config.max_problems and 
                len(all_problem_traces) >= dataset_config.max_problems):
                print(f"🎯 Reached max problems limit: {dataset_config.max_problems}")
                break
                
            # Periodic updates
            if processed_batches % 10 == 0:
                print(f"📈 Processed {len(all_problem_traces)} problems so far...")
    
    except KeyboardInterrupt:
        print(f"\n⚠️  Interrupted by user. Saving progress...")
    
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        raise
    
    # Save dataset metadata
    dataset_metadata = {
        "total_problems": len(all_problem_traces),
        "successful_traces": sum(1 for pt in all_problem_traces if pt.metadata.get("halted", False)),
        "config": asdict(dataset_config),
        "model_checkpoint": dataset_config.checkpoint_path,
        "created_at": torch.utils.data.get_worker_info(),  # Timestamp would be better
        "problems": [asdict(pt) for pt in all_problem_traces]
    }
    
    metadata_file = output_dir / "dataset_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(dataset_metadata, f, indent=2)
    
    print(f"\n✅ Dataset building completed!")
    print(f"   📊 Total problems processed: {len(all_problem_traces)}")
    print(f"   ✅ Successful traces: {dataset_metadata['successful_traces']}")
    print(f"   💾 Metadata saved to: {metadata_file}")
    
    return dataset_metadata


def upload_to_comet(dataset_config: TraceDatasetConfig, dataset_metadata: Dict[str, Any]):
    """Upload the dataset to Comet ML."""
    print(f"\n🚀 Uploading dataset to Comet ML...")
    
    try:
        import comet_ml
        
        # Create experiment
        experiment = comet_ml.Experiment(project_name=dataset_config.comet_project)
        experiment.set_name(f"sudoku-trace-dataset-{len(dataset_metadata['problems'])}-problems")
        experiment.add_tag("trace-dataset")
        experiment.add_tag("sudoku")
        experiment.add_tag("training-data")
        
        # Log dataset statistics
        experiment.log_parameter("total_problems", dataset_metadata["total_problems"])
        experiment.log_parameter("successful_traces", dataset_metadata["successful_traces"])
        experiment.log_parameter("batch_size", dataset_config.batch_size)
        experiment.log_parameter("max_steps_per_problem", dataset_config.max_steps_per_problem)
        experiment.log_parameter("tensor_format", dataset_config.tensor_format)
        experiment.log_parameter("model_checkpoint", dataset_config.checkpoint_path)
        
        # Create artifact
        artifact = comet_ml.Artifact(
            name=dataset_config.comet_artifact_name,
            artifact_type="dataset",
            aliases=["latest", "training-set", f"{len(dataset_metadata['problems'])}-problems"],
            metadata={
                "dataset_type": "hrm_trace_dataset",
                "task": "sudoku",
                "total_problems": dataset_metadata["total_problems"],
                "tensor_format": dataset_config.tensor_format,
                "created_from": dataset_config.checkpoint_path
            }
        )
        
        # Add dataset files
        output_dir = Path(dataset_config.output_dir)
        
        # Add metadata
        artifact.add(
            str(output_dir / "dataset_metadata.json"),
            logical_path="dataset_metadata.json",
            metadata={"file_type": "metadata", "format": "json"}
        )
        
        # Add trace files
        traces_dir = output_dir / "traces"
        if traces_dir.exists():
            for trace_file in traces_dir.glob(f"*.{dataset_config.tensor_format}"):
                artifact.add(
                    str(trace_file),
                    logical_path=f"traces/{trace_file.name}",
                    metadata={"file_type": "trace_tensors", "format": dataset_config.tensor_format}
                )
        
        # Upload artifact
        print(f"📤 Uploading artifact with {len(list(traces_dir.glob('*')))} trace files...")
        experiment.log_artifact(artifact)
        
        print(f"✅ Successfully uploaded to Comet ML!")
        print(f"   🌐 Experiment: {experiment.url}")
        print(f"   📦 Artifact: {dataset_config.comet_artifact_name}")
        
        experiment.end()
        return experiment.url
        
    except ImportError:
        print("⚠️  Comet ML not available. Skipping upload.")
        print("   Install with: pip install comet_ml")
        return None
    except Exception as e:
        print(f"❌ Failed to upload to Comet: {e}")
        return None


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Build comprehensive Sudoku trace dataset")
    parser.add_argument("--checkpoint", 
                       default="./checkpoints/sudoku-extreme/checkpoint",
                       help="Path to model checkpoint")
    parser.add_argument("--output-dir", 
                       default="./sudoku_trace_dataset",
                       help="Output directory for dataset")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Batch size for processing")
    parser.add_argument("--max-problems", type=int, default=None,
                       help="Maximum number of problems to process")
    parser.add_argument("--max-steps", type=int, default=50,
                       help="Maximum steps per problem")
    parser.add_argument("--tensor-format", choices=["pt", "npz", "both"], default="pt",
                       help="Format for saving tensors")
    parser.add_argument("--include-failed", action="store_true",
                       help="Include traces where model failed to halt")
    parser.add_argument("--no-upload", action="store_true",
                       help="Skip uploading to Comet ML")
    parser.add_argument("--comet-project", default="hrm-trace-datasets",
                       help="Comet ML project name")
    parser.add_argument("--comet-artifact", default="sudoku-training-traces",
                       help="Comet ML artifact name")
    
    args = parser.parse_args()
    
    # Validate checkpoint
    if not Path(args.checkpoint).exists():
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        print("   Please run download_models.py first or specify correct checkpoint path")
        return 1
    
    # Create configuration
    config = TraceDatasetConfig(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_problems=args.max_problems,
        max_steps_per_problem=args.max_steps,
        tensor_format=args.tensor_format,
        include_failed=args.include_failed,
        comet_project=args.comet_project,
        comet_artifact_name=args.comet_artifact
    )
    
    try:
        # Build dataset
        dataset_metadata = build_trace_dataset(config)
        
        # Upload to Comet ML
        if not args.no_upload:
            comet_url = upload_to_comet(config, dataset_metadata)
            if comet_url:
                print(f"\n🎯 Next Steps:")
                print(f"   1. Explore dataset: {args.output_dir}")
                print(f"   2. View on Comet: {comet_url}")
                print(f"   3. Use for analysis: load traces from {args.output_dir}/traces/")
        else:
            print(f"\n💾 Dataset saved locally to: {args.output_dir}")
        
        print(f"\n🎉 Sudoku trace dataset building completed!")
        return 0
        
    except Exception as e:
        print(f"❌ Dataset building failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main()) 