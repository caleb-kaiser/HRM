#!/usr/bin/env python3
"""
Test Trace Format Verification

This script verifies that the generated trace files are correctly formatted
and contain all the expected data fields for HRM state analysis.
"""

import torch
import json
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import traceback

def test_file_naming(traces_dir: Path) -> Dict[str, Any]:
    """Test that trace files follow the expected naming convention."""
    print("🔍 Testing file naming convention...")
    
    results = {
        "total_files": 0,
        "correct_format": 0,
        "incorrect_format": [],
        "missing_pairs": [],
        "naming_issues": []
    }
    
    # Expected pattern: trace_XXXXXX_trace_0.pt
    trace_files = list(traces_dir.glob("*.pt"))
    results["total_files"] = len(trace_files)
    
    print(f"   Found {len(trace_files)} .pt files")
    
    for trace_file in trace_files:
        name = trace_file.name
        
        # Check if it matches expected pattern
        if name.startswith("trace_") and name.endswith("_trace_0.pt"):
            try:
                # Extract problem ID
                parts = name.replace(".pt", "").split("_")
                if len(parts) == 4 and parts[0] == "trace" and parts[2] == "trace" and parts[3] == "0":
                    problem_id = int(parts[1])
                    results["correct_format"] += 1
                    print(f"   ✅ {name} -> Problem ID: {problem_id}")
                else:
                    results["incorrect_format"].append(name)
                    print(f"   ❌ {name} -> Invalid format (wrong parts)")
            except ValueError as e:
                results["incorrect_format"].append(name)
                print(f"   ❌ {name} -> Cannot parse problem ID: {e}")
        else:
            results["incorrect_format"].append(name)
            print(f"   ❌ {name} -> Doesn't match expected pattern")
    
    print(f"   ✅ Correctly formatted: {results['correct_format']}/{results['total_files']}")
    
    return results

def test_trace_data_format(trace_file: Path) -> Dict[str, Any]:
    """Test the internal format of a single trace file."""
    print(f"\n🔍 Testing trace data format: {trace_file.name}")
    
    results = {
        "file_loadable": False,
        "expected_keys": [],
        "missing_keys": [],
        "unexpected_keys": [],
        "data_shapes": {},
        "data_types": {},
        "tensor_info": {}
    }
    
    try:
        # Load the trace data
        trace_data = torch.load(trace_file, map_location='cpu')
        results["file_loadable"] = True
        print(f"   ✅ File loaded successfully")
        
        # Expected keys based on HRM state recording
        expected_keys = {
            'h_states',      # H-module hidden states
            'l_states',      # L-module hidden states  
            'q_halt_logits', # Q-head halt logits
            'metadata'       # Execution metadata
        }
        
        actual_keys = set(trace_data.keys())
        results["expected_keys"] = list(expected_keys)
        results["missing_keys"] = list(expected_keys - actual_keys)
        results["unexpected_keys"] = list(actual_keys - expected_keys)
        
        print(f"   📋 Available keys: {list(actual_keys)}")
        print(f"   ✅ Expected keys found: {list(expected_keys & actual_keys)}")
        
        if results["missing_keys"]:
            print(f"   ❌ Missing keys: {results['missing_keys']}")
        
        if results["unexpected_keys"]:
            print(f"   ⚠️  Unexpected keys: {results['unexpected_keys']}")
        
        # Analyze each tensor
        for key, value in trace_data.items():
            if isinstance(value, torch.Tensor):
                results["data_shapes"][key] = list(value.shape)
                results["data_types"][key] = str(value.dtype)
                results["tensor_info"][key] = {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "min": float(value.min().item()) if value.numel() > 0 else None,
                    "max": float(value.max().item()) if value.numel() > 0 else None,
                    "mean": float(value.mean().item()) if value.numel() > 0 else None,
                }
                print(f"   📊 {key}: shape={value.shape}, dtype={value.dtype}")
                if value.numel() > 0:
                    print(f"      Range: [{value.min().item():.4f}, {value.max().item():.4f}], Mean: {value.mean().item():.4f}")
            elif isinstance(value, dict):
                results["data_shapes"][key] = "dict"
                results["data_types"][key] = "dict"
                print(f"   📊 {key}: dict with keys {list(value.keys())}")
            else:
                results["data_shapes"][key] = "other"
                results["data_types"][key] = str(type(value))
                print(f"   📊 {key}: {type(value)} = {value}")
        
        # Specific validation for critical fields
        if 'h_states' in trace_data:
            h_states = trace_data['h_states']
            if h_states.dim() >= 2:
                print(f"   ✅ H-states: {h_states.shape[0]} timesteps, {h_states.shape[1]} hidden dimensions")
            else:
                print(f"   ❌ H-states: Invalid shape {h_states.shape}")
        
        if 'l_states' in trace_data:
            l_states = trace_data['l_states']
            if l_states.dim() >= 2:
                print(f"   ✅ L-states: {l_states.shape[0]} timesteps, {l_states.shape[1]} hidden dimensions")
            else:
                print(f"   ❌ L-states: Invalid shape {l_states.shape}")
        
        if 'q_halt_logits' in trace_data:
            q_logits = trace_data['q_halt_logits']
            if q_logits.dim() >= 1:
                print(f"   ✅ Q-halt logits: {q_logits.shape[0]} timesteps")
                # Check if logits look reasonable (should be real numbers)
                if torch.isfinite(q_logits).all():
                    print(f"   ✅ Q-halt logits: All values are finite")
                else:
                    print(f"   ❌ Q-halt logits: Contains infinite/NaN values")
            else:
                print(f"   ❌ Q-halt logits: Invalid shape {q_logits.shape}")
        
        if 'metadata' in trace_data:
            metadata = trace_data['metadata']
            if isinstance(metadata, dict):
                print(f"   ✅ Metadata: dict with keys {list(metadata.keys())}")
                
                # Check for important metadata fields
                important_fields = ['steps_taken', 'cycles', 'execution_time']
                for field in important_fields:
                    if field in metadata:
                        print(f"      ✅ {field}: {metadata[field]}")
                    else:
                        print(f"      ⚠️  Missing {field}")
            else:
                print(f"   ❌ Metadata: Not a dict, got {type(metadata)}")
        
    except Exception as e:
        print(f"   ❌ Failed to load/analyze trace: {e}")
        traceback.print_exc()
    
    return results

def test_metadata_consistency(traces_dir: Path, metadata_file: Path) -> Dict[str, Any]:
    """Test consistency between trace files and metadata."""
    print(f"\n🔍 Testing metadata consistency...")
    
    results = {
        "metadata_loadable": False,
        "trace_files_found": 0,
        "metadata_entries": 0,
        "consistent_entries": 0,
        "inconsistencies": []
    }
    
    try:
        # Load metadata
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        results["metadata_loadable"] = True
        print(f"   ✅ Metadata file loaded")
        
        # Get problem traces
        problems = metadata.get('problems', [])
        results["metadata_entries"] = len(problems)
        print(f"   📊 Metadata entries: {len(problems)}")
        
        # Check each problem entry
        for problem in problems:
            problem_id = problem.get('problem_id')
            trace_file_path = problem.get('trace_file')
            
            if trace_file_path:
                # Convert to actual file path
                trace_file = Path(trace_file_path)
                if not trace_file.is_absolute():
                    trace_file = traces_dir / trace_file.name
                
                if trace_file.exists():
                    results["trace_files_found"] += 1
                    results["consistent_entries"] += 1
                    print(f"   ✅ Problem {problem_id}: File exists {trace_file.name}")
                else:
                    results["inconsistencies"].append(f"Problem {problem_id}: File not found {trace_file}")
                    print(f"   ❌ Problem {problem_id}: File not found {trace_file}")
            else:
                results["inconsistencies"].append(f"Problem {problem_id}: No trace_file specified")
                print(f"   ❌ Problem {problem_id}: No trace_file specified")
        
        print(f"   ✅ Consistent entries: {results['consistent_entries']}/{results['metadata_entries']}")
        
    except Exception as e:
        print(f"   ❌ Failed to test metadata consistency: {e}")
    
    return results

def test_data_completeness(traces_dir: Path, num_samples: int = 3) -> Dict[str, Any]:
    """Test a sample of trace files for data completeness."""
    print(f"\n🔍 Testing data completeness (sampling {num_samples} files)...")
    
    results = {
        "files_tested": 0,
        "complete_files": 0,
        "incomplete_files": [],
        "common_issues": {}
    }
    
    trace_files = list(traces_dir.glob("*.pt"))
    if not trace_files:
        print(f"   ❌ No trace files found in {traces_dir}")
        return results
    
    # Sample files to test
    sample_files = trace_files[:num_samples] if len(trace_files) >= num_samples else trace_files
    
    for trace_file in sample_files:
        print(f"\n   📄 Testing: {trace_file.name}")
        file_results = test_trace_data_format(trace_file)
        results["files_tested"] += 1
        
        # Check completeness
        if (file_results["file_loadable"] and 
            len(file_results["missing_keys"]) == 0 and
            'h_states' in file_results["data_shapes"] and
            'l_states' in file_results["data_shapes"] and
            'q_halt_logits' in file_results["data_shapes"]):
            results["complete_files"] += 1
            print(f"   ✅ Complete trace file")
        else:
            results["incomplete_files"].append(trace_file.name)
            print(f"   ❌ Incomplete trace file")
            
            # Track common issues
            for missing_key in file_results["missing_keys"]:
                if missing_key not in results["common_issues"]:
                    results["common_issues"][missing_key] = 0
                results["common_issues"][missing_key] += 1
    
    print(f"\n   📊 Summary: {results['complete_files']}/{results['files_tested']} files are complete")
    if results["common_issues"]:
        print(f"   ⚠️  Common issues: {results['common_issues']}")
    
    return results

def main():
    """Main testing function."""
    parser = argparse.ArgumentParser(description="Test trace file format and data integrity")
    parser.add_argument("--traces-dir", 
                       default="./test_fixed_traces/traces",
                       help="Directory containing trace files")
    parser.add_argument("--metadata-file",
                       default="./test_fixed_traces/dataset_metadata.json", 
                       help="Path to dataset metadata file")
    parser.add_argument("--sample-size", type=int, default=5,
                       help="Number of files to test in detail")
    
    args = parser.parse_args()
    
    traces_dir = Path(args.traces_dir)
    metadata_file = Path(args.metadata_file)
    
    print("🧪 HRM Trace Format Verification")
    print("=" * 50)
    print(f"📁 Traces directory: {traces_dir}")
    print(f"📄 Metadata file: {metadata_file}")
    
    # Check if directories exist
    if not traces_dir.exists():
        print(f"❌ Traces directory not found: {traces_dir}")
        return 1
    
    if not metadata_file.exists():
        print(f"❌ Metadata file not found: {metadata_file}")
        return 1
    
    # Run tests
    all_results = {}
    
    # Test 1: File naming
    all_results["naming"] = test_file_naming(traces_dir)
    
    # Test 2: Metadata consistency  
    all_results["metadata"] = test_metadata_consistency(traces_dir, metadata_file)
    
    # Test 3: Data completeness
    all_results["completeness"] = test_data_completeness(traces_dir, args.sample_size)
    
    # Final summary
    print(f"\n🎯 FINAL SUMMARY")
    print("=" * 50)
    
    naming = all_results["naming"]
    print(f"📛 File Naming: {naming['correct_format']}/{naming['total_files']} correct")
    
    metadata = all_results["metadata"]
    print(f"📋 Metadata: {metadata['consistent_entries']}/{metadata['metadata_entries']} consistent")
    
    completeness = all_results["completeness"] 
    print(f"📊 Data Completeness: {completeness['complete_files']}/{completeness['files_tested']} complete")
    
    # Overall assessment
    issues = []
    if naming['incorrect_format']:
        issues.append(f"{len(naming['incorrect_format'])} naming issues")
    if metadata['inconsistencies']:
        issues.append(f"{len(metadata['inconsistencies'])} metadata inconsistencies")
    if completeness['incomplete_files']:
        issues.append(f"{len(completeness['incomplete_files'])} incomplete files")
    
    if issues:
        print(f"\n⚠️  Issues found: {', '.join(issues)}")
        return 1
    else:
        print(f"\n✅ All tests passed! Traces are correctly formatted.")
        return 0

if __name__ == "__main__":
    exit(main()) 