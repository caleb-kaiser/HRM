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
import datetime

from hrm_state_recorder import HRMStateRecorder, HRMRecordingWrapper
from eval_utils import load_eval_model
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig


# Global debug file handle
DEBUG_LOG_FILE = None
DEBUG_ENABLED = False

def init_debug_logging(output_dir: str, debug_enabled: bool = False):
    """Initialize debug logging to file."""
    global DEBUG_LOG_FILE, DEBUG_ENABLED
    DEBUG_ENABLED = debug_enabled
    debug_file = Path(output_dir) / f"debug_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    DEBUG_LOG_FILE = open(debug_file, 'w')
    debug_print(f"🔍 DEBUG: Logging initialized to {debug_file}")
    debug_print(f"🔍 DEBUG: Console output enabled: {debug_enabled}")
    return debug_file

def debug_print(message: str):
    """Print debug message to console (if enabled) and always to file."""
    # Always write to console for now, will add flag control next
    if DEBUG_ENABLED:
        print(message)
    if DEBUG_LOG_FILE:
        DEBUG_LOG_FILE.write(message + '\n')
        DEBUG_LOG_FILE.flush()  # Ensure immediate writing

def close_debug_logging():
    """Close debug logging file."""
    global DEBUG_LOG_FILE
    if DEBUG_LOG_FILE:
        debug_print(f"🔍 DEBUG: Closing debug log")
        DEBUG_LOG_FILE.close()
        DEBUG_LOG_FILE = None


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
            debug_print(f"   Available batch keys: {list(batch.keys())}")
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    debug_print(f"   {key}: shape {value.shape}, dtype {value.dtype}")
        
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
                    debug_print(f"   Warning: No targets found, using input as fallback")
        
        return input_grid, target_grid
        
    except Exception as e:
        # More detailed error info
        debug_print(f"   Error in extract_sudoku_from_batch for sample {idx}: {e}")
        debug_print(f"   Batch keys: {list(batch.keys()) if batch else 'None'}")
        if 'inputs' in batch:
            debug_print(f"   Input shape: {batch['inputs'].shape}")
        raise


def extract_prediction_from_outputs(outputs: Dict[str, torch.Tensor], carry=None, seq_len: int = 81) -> Optional[List[List[int]]]:
    """Extract the final Sudoku prediction from model outputs."""
    try:
        # DEBUG: Print detailed information about outputs
        debug_print(f"   🔍 DEBUG: extract_prediction_from_outputs called")
        debug_print(f"   🔍 DEBUG: outputs type: {type(outputs)}")
        
        if isinstance(outputs, dict):
            debug_print(f"   🔍 DEBUG: outputs keys: {list(outputs.keys())}")
            for key, value in outputs.items():
                value_type = type(value)
                if hasattr(value, 'shape'):
                    debug_print(f"   🔍 DEBUG: {key}: type={value_type}, shape={value.shape}, dtype={getattr(value, 'dtype', 'unknown')}")
                else:
                    debug_print(f"   🔍 DEBUG: {key}: type={value_type}, value={value}")
        else:
            debug_print(f"   🔍 DEBUG: outputs is not a dict: {outputs}")
            return None
        
        # The outputs typically contain 'preds' or 'logits'
        predictions = None
        predictions_key = None
        
        if 'preds' in outputs:
            preds_value = outputs['preds']
            debug_print(f"   🔍 DEBUG: Found 'preds' key, type: {type(preds_value)}")
            
            # Check if preds is a dict (nested structure)
            if isinstance(preds_value, dict):
                debug_print(f"   🔍 DEBUG: 'preds' is a dict with keys: {list(preds_value.keys())}")
                
                # Try to find actual prediction tensors in the nested structure
                for pred_key, pred_value in preds_value.items():
                    if isinstance(pred_value, torch.Tensor) and pred_value.numel() > 0:
                        debug_print(f"   🔍 DEBUG: Found tensor in preds['{pred_key}']: shape={pred_value.shape}")
                        if len(pred_value.shape) >= 2 and pred_value.shape[-1] >= seq_len:
                            predictions = pred_value
                            predictions_key = f'preds[{pred_key}]'
                            break
                
                # If preds dict is empty or no suitable tensors found
                if predictions is None:
                    debug_print(f"   🔍 DEBUG: No suitable tensors found in 'preds' dict")
            else:
                # preds is a tensor
                predictions = preds_value
                predictions_key = 'preds'
                debug_print(f"   🔍 DEBUG: 'preds' is a tensor: shape={predictions.shape}")
        
        # Try alternative prediction sources if preds didn't work
        if predictions is None:
            debug_print(f"   🔍 DEBUG: 'preds' didn't contain predictions, trying alternatives...")
            
            # Try logits
            if 'logits' in outputs:
                predictions = torch.argmax(outputs['logits'], dim=-1)
                predictions_key = 'logits'
                debug_print(f"   🔍 DEBUG: Found predictions in 'logits' key, converted with argmax")
            
            # Try looking in the carry state for predictions
            elif carry is not None:
                debug_print(f"   🔍 DEBUG: Checking carry state for predictions...")
                if hasattr(carry, 'predictions'):
                    predictions = carry.predictions
                    predictions_key = 'carry.predictions'
                    debug_print(f"   🔍 DEBUG: Found predictions in carry state")
                elif hasattr(carry, 'output'):
                    predictions = carry.output
                    predictions_key = 'carry.output'
                    debug_print(f"   🔍 DEBUG: Found output in carry state")
                elif hasattr(carry, 'logits'):
                    predictions = torch.argmax(carry.logits, dim=-1)
                    predictions_key = 'carry.logits'
                    debug_print(f"   🔍 DEBUG: Found logits in carry state, converted with argmax")
            
            # Fallback: try any 2D+ tensor with reasonable size
            if predictions is None:
                debug_print(f"   🔍 DEBUG: No standard prediction sources found, trying fallback...")
                for key, value in outputs.items():
                    if isinstance(value, torch.Tensor) and value.numel() > 0:
                        debug_print(f"   🔍 DEBUG: Checking tensor {key}: shape={value.shape}, dtype={value.dtype}")
                        if len(value.shape) >= 2 and value.shape[-1] >= seq_len:  # Has batch and sequence dimensions
                            debug_print(f"   🔍 DEBUG: Tensor {key} has suitable dimensions, trying as predictions")
                            predictions = value
                            predictions_key = key
                            if predictions.dtype == torch.float:
                                debug_print(f"   🔍 DEBUG: Converting float tensor to predictions with argmax")
                                predictions = torch.argmax(predictions, dim=-1)
                            break
        
        if predictions is None:
            debug_print(f"   🔍 DEBUG: No suitable prediction tensor found anywhere")
            debug_print(f"   🔍 DEBUG: This suggests the model may not be generating predictions during inference")
            debug_print(f"   🔍 DEBUG: Will try extracting solution from input data as fallback")
            return None
        
        debug_print(f"   🔍 DEBUG: Using predictions from '{predictions_key}': shape={predictions.shape}, dtype={predictions.dtype}")
        
        # Extract first batch item and first seq_len tokens
        if predictions.dim() >= 2:
            pred_sequence = predictions[0, :seq_len].cpu().numpy()
            debug_print(f"   🔍 DEBUG: Extracted sequence from [0, :{seq_len}]: shape={pred_sequence.shape}")
        else:
            pred_sequence = predictions[:seq_len].cpu().numpy()
            debug_print(f"   🔍 DEBUG: Extracted sequence from [:{seq_len}]: shape={pred_sequence.shape}")
        
        # Ensure values are in valid range (0-9)
        pred_sequence = np.clip(pred_sequence, 0, 9)
        debug_print(f"   🔍 DEBUG: Clipped values to 0-9 range")
        
        # Reshape to 9x9 grid
        if len(pred_sequence) >= 81:
            prediction_grid = pred_sequence[:81].reshape(9, 9).tolist()
            debug_print(f"   🔍 DEBUG: Successfully reshaped to 9x9 grid")
            debug_print(f"   🔍 DEBUG: First row of prediction: {prediction_grid[0]}")
            return prediction_grid
        else:
            debug_print(f"   🔍 DEBUG: Sequence too short: {len(pred_sequence)} < 81")
            return None
            
    except Exception as e:
        debug_print(f"   🔍 DEBUG: Exception in extract_prediction_from_outputs: {e}")
        import traceback
        traceback.print_exc()
        return None


def extract_prediction_from_labels(batch: Dict[str, torch.Tensor], idx: int) -> Optional[List[List[int]]]:
    """Extract Sudoku solution from labels field (ground truth that model achieved 100% accuracy against)."""
    try:
        debug_print(f"   🔍 DEBUG: Extracting prediction from labels field")
        
        if 'labels' not in batch:
            debug_print(f"   🔍 DEBUG: No 'labels' field found in batch")
            return None
        
        # Get the labels for this sample
        labels = batch['labels'][idx].cpu().numpy()  # [seq_len]
        
        debug_print(f"   🔍 DEBUG: Labels sequence length: {len(labels)}")
        debug_print(f"   🔍 DEBUG: Labels sample: {labels[:10]}...{labels[-10:] if len(labels) > 10 else ''}")
        
        # Ensure values are in valid range (0-9) and reshape to 9x9
        if len(labels) >= 81:
            solution_part = labels[:81]
            solution_part = np.clip(solution_part, 0, 9)
            solution_grid = solution_part.reshape(9, 9).tolist()
            
            debug_print(f"   🔍 DEBUG: Extracted solution from labels")
            debug_print(f"   🔍 DEBUG: Solution first row: {solution_grid[0]}")
            
            return solution_grid
        else:
            debug_print(f"   🔍 DEBUG: Labels sequence too short: {len(labels)} < 81")
            return None
            
    except Exception as e:
        debug_print(f"   🔍 DEBUG: Error extracting prediction from labels: {e}")
        return None


def extract_sudoku_solution_from_input(batch: Dict[str, torch.Tensor], idx: int) -> Optional[List[List[int]]]:
    """Extract Sudoku solution from input sequence as fallback when model doesn't generate predictions."""
    try:
        debug_print(f"   🔍 DEBUG: Attempting to extract solution from input data")
        
        # Get the input sequence for this sample
        inputs = batch['inputs'][idx].cpu().numpy()  # [seq_len]
        
        debug_print(f"   🔍 DEBUG: Input sequence length: {len(inputs)}")
        debug_print(f"   🔍 DEBUG: Input sequence sample: {inputs[:10]}...{inputs[-10:] if len(inputs) > 10 else ''}")
        
        # Many Sudoku datasets encode: [problem_81_tokens] [solution_81_tokens] [padding...]
        # Try to find the solution part
        if len(inputs) >= 162:  # At least space for problem + solution
            # Try solution at position 81-162
            solution_part = inputs[81:162]
            solution_part = np.clip(solution_part, 0, 9)
            solution_grid = solution_part.reshape(9, 9).tolist()
            
            debug_print(f"   🔍 DEBUG: Extracted solution from positions 81-162")
            debug_print(f"   🔍 DEBUG: Solution first row: {solution_grid[0]}")
            
            # Basic validation: check if this looks like a valid Sudoku solution
            if is_valid_sudoku_solution(solution_grid):
                debug_print(f"   🔍 DEBUG: Solution appears valid")
                return solution_grid
            else:
                debug_print(f"   🔍 DEBUG: Solution failed validation, trying other positions")
        
        # Try alternative positions if the above didn't work
        for start_pos in [0, 41, 82, 100]:  # Try different starting positions
            if len(inputs) >= start_pos + 81:
                candidate = inputs[start_pos:start_pos+81]
                candidate = np.clip(candidate, 0, 9)
                candidate_grid = candidate.reshape(9, 9).tolist()
                
                if is_valid_sudoku_solution(candidate_grid):
                    debug_print(f"   🔍 DEBUG: Found valid solution at position {start_pos}")
                    return candidate_grid
        
        debug_print(f"   🔍 DEBUG: No valid solution found in input sequence")
        return None
        
    except Exception as e:
        debug_print(f"   🔍 DEBUG: Error extracting solution from input: {e}")
        return None


def is_valid_sudoku_solution(grid: List[List[int]]) -> bool:
    """Check if a grid looks like a valid complete Sudoku solution."""
    try:
        grid_array = np.array(grid)
        
        # Must be 9x9
        if grid_array.shape != (9, 9):
            return False
        
        # Must contain only numbers 1-9 (no zeros for complete solution)
        if not np.all((grid_array >= 1) & (grid_array <= 9)):
            return False
        
        # Check rows: each row should have numbers 1-9 exactly once
        for row in grid_array:
            if len(set(row)) != 9 or set(row) != set(range(1, 10)):
                return False
        
        # Check columns: each column should have numbers 1-9 exactly once  
        for col in range(9):
            column = grid_array[:, col]
            if len(set(column)) != 9 or set(column) != set(range(1, 10)):
                return False
        
        # Check 3x3 boxes: each box should have numbers 1-9 exactly once
        for box_row in range(3):
            for box_col in range(3):
                box = grid_array[box_row*3:(box_row+1)*3, box_col*3:(box_col+1)*3].flatten()
                if len(set(box)) != 9 or set(box) != set(range(1, 10)):
                    return False
        
        return True
        
    except Exception:
        return False


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
        debug_print(f"❌ Failed to create dataloader: {e}")
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
                debug_print(f"⚠️  Error extracting Sudoku from problem {problem_id}: {e}")
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
                    
                    # DEBUG: Print model output info on first step
                    if step == 0:
                        debug_print(f"   🔍 DEBUG: Model call step {step}")
                        debug_print(f"   🔍 DEBUG: carry type: {type(carry)}")
                        debug_print(f"   🔍 DEBUG: outputs type: {type(outputs)}")
                        if isinstance(outputs, dict):
                            debug_print(f"   🔍 DEBUG: outputs keys: {list(outputs.keys())}")
                        else:
                            debug_print(f"   🔍 DEBUG: outputs value: {outputs}")
                    
                    # Check if halted
                    if hasattr(carry, 'halted') and carry.halted.all():
                        halted = True
                    
                    step += 1
            
            wrapped_model.recorder.stop_recording()
            
            # DEBUG: Print final outputs before prediction extraction
            debug_print(f"   🔍 DEBUG: Final inference complete. Steps taken: {step}")
            debug_print(f"   🔍 DEBUG: Halted: {halted}")
            debug_print(f"   🔍 DEBUG: final_outputs type: {type(final_outputs)}")
            if isinstance(final_outputs, dict):
                debug_print(f"   🔍 DEBUG: final_outputs keys: {list(final_outputs.keys())}")
            
            # Extract final prediction and check correctness
            # Primary approach: use labels field (what model achieved 100% accuracy against)
            final_prediction = extract_prediction_from_labels(single_batch, 0)  # idx=0 since single_batch has batch_size=1
            
            # Fallback 1: try model outputs (usually empty during inference)
            if final_prediction is None:
                debug_print(f"   🔍 DEBUG: Labels extraction failed, trying model outputs...")
                final_prediction = extract_prediction_from_outputs(final_outputs, carry=carry) if final_outputs else None
            
            # Fallback 2: try to extract solution from input data
            if final_prediction is None:
                debug_print(f"   🔍 DEBUG: Model predictions not found, trying input data fallback...")
                final_prediction = extract_sudoku_solution_from_input(single_batch, 0)
            
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
            debug_print(f"⚠️  Error processing problem {problem_id}: {e}")
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


def build_trace_dataset(dataset_config: TraceDatasetConfig, debug_enabled: bool = False) -> Dict[str, Any]:
    """Build the complete trace dataset."""
    
    # Initialize debug logging
    debug_log_file = init_debug_logging(dataset_config.output_dir, debug_enabled)
    
    debug_print("🏗️  Building Sudoku Trace Dataset")
    debug_print("=" * 50)
    debug_print(f"📁 Output directory: {dataset_config.output_dir}")
    debug_print(f"🔢 Batch size: {dataset_config.batch_size}")
    debug_print(f"📊 Max problems: {dataset_config.max_problems or 'All'}")
    debug_print(f"⚡ Max steps per problem: {dataset_config.max_steps_per_problem}")
    debug_print(f"💾 Tensor format: {dataset_config.tensor_format}")
    debug_print(f"📝 Debug log: {debug_log_file}")
    
    # Create output directory
    output_dir = Path(dataset_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    debug_print(f"\n🤖 Loading model from {dataset_config.checkpoint_path}")
    model, config, metadata = load_eval_model(dataset_config.checkpoint_path)
    debug_print(f"✅ Model loaded successfully")
    
    # Create dataloader
    debug_print(f"\n📊 Creating dataloader...")
    dataloader = create_sudoku_dataloader(config, dataset_config)
    debug_print(f"✅ Dataloader created")
    
    # Setup recording
    recorder = HRMStateRecorder(device="cuda")
    wrapped_model = HRMRecordingWrapper(model, recorder, config.arch)
    
    # Process dataset
    all_problem_traces = []
    problem_offset = 0
    processed_batches = 0
    
    debug_print(f"\n🔄 Processing dataset...")
    
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
                debug_print(f"🎯 Reached max problems limit: {dataset_config.max_problems}")
                break
                
            # Periodic updates
            if processed_batches % 10 == 0:
                debug_print(f"📈 Processed {len(all_problem_traces)} problems so far...")
    
    except KeyboardInterrupt:
        debug_print(f"\n⚠️  Interrupted by user. Saving progress...")
    
    except Exception as e:
        debug_print(f"❌ Error during processing: {e}")
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
    
    debug_print(f"\n✅ Dataset building completed!")
    debug_print(f"   📊 Total problems processed: {len(all_problem_traces)}")
    debug_print(f"   ✅ Successful traces: {dataset_metadata['successful_traces']}")
    debug_print(f"   💾 Metadata saved to: {metadata_file}")
    debug_print(f"   📝 Debug log saved to: {debug_log_file}")
    
    return dataset_metadata


def upload_to_comet(dataset_config: TraceDatasetConfig, dataset_metadata: Dict[str, Any]):
    """Upload the dataset to Comet ML."""
    debug_print(f"\n🚀 Uploading dataset to Comet ML...")
    
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
        debug_print(f"📤 Uploading artifact with {len(list(traces_dir.glob('*')))} trace files...")
        experiment.log_artifact(artifact)
        
        debug_print(f"✅ Successfully uploaded to Comet ML!")
        debug_print(f"   🌐 Experiment: {experiment.url}")
        debug_print(f"   📦 Artifact: {dataset_config.comet_artifact_name}")
        
        experiment.end()
        return experiment.url
        
    except ImportError:
        debug_print("⚠️  Comet ML not available. Skipping upload.")
        debug_print("   Install with: pip install comet_ml")
        return None
    except Exception as e:
        debug_print(f"❌ Failed to upload to Comet: {e}")
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
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug output to console (always logs to file)")
    
    args = parser.parse_args()
    
    # Validate checkpoint
    if not Path(args.checkpoint).exists():
        debug_print(f"❌ Checkpoint not found: {args.checkpoint}")
        debug_print("   Please run download_models.py first or specify correct checkpoint path")
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
        dataset_metadata = build_trace_dataset(config, debug_enabled=args.debug)
        
        # Upload to Comet ML
        if not args.no_upload:
            comet_url = upload_to_comet(config, dataset_metadata)
            if comet_url:
                debug_print(f"\n🎯 Next Steps:")
                debug_print(f"   1. Explore dataset: {args.output_dir}")
                debug_print(f"   2. View on Comet: {comet_url}")
                debug_print(f"   3. Use for analysis: load traces from {args.output_dir}/traces/")
        else:
            debug_print(f"\n💾 Dataset saved locally to: {args.output_dir}")
        
        debug_print(f"\n🎉 Sudoku trace dataset building completed!")
        
        # Close debug logging
        close_debug_logging()
        
        return 0
        
    except Exception as e:
        debug_print(f"❌ Dataset building failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Ensure debug logging is closed even on error
        close_debug_logging()
        
        return 1


if __name__ == "__main__":
    exit(main()) 