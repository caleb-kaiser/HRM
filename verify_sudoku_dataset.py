#!/usr/bin/env python3
"""
Verify Sudoku Dataset Quality

This script checks if the generated Sudoku dataset contains valid puzzles and solutions.
"""

import numpy as np
import json
from pathlib import Path
import argparse


def is_valid_sudoku_solution(grid):
    """Check if a 9x9 grid is a valid complete Sudoku solution."""
    if grid.shape != (9, 9):
        return False
    
    # Must contain only numbers 1-9 (no zeros for complete solution)
    if not np.all((grid >= 1) & (grid <= 9)):
        return False
    
    # Check rows: each row should have numbers 1-9 exactly once
    for row in grid:
        if len(set(row)) != 9 or set(row) != set(range(1, 10)):
            return False
    
    # Check columns: each column should have numbers 1-9 exactly once  
    for col in range(9):
        column = grid[:, col]
        if len(set(column)) != 9 or set(column) != set(range(1, 10)):
            return False
    
    # Check 3x3 boxes: each box should have numbers 1-9 exactly once
    for box_row in range(3):
        for box_col in range(3):
            box = grid[box_row*3:(box_row+1)*3, box_col*3:(box_col+1)*3].flatten()
            if len(set(box)) != 9 or set(box) != set(range(1, 10)):
                return False
    
    return True


def is_valid_sudoku_problem(grid, solution):
    """Check if a problem grid is consistent with its solution."""
    # Problem grid should have 0s for empty cells and numbers 1-9 for clues
    if not np.all((grid >= 0) & (grid <= 9)):
        return False
    
    # Every non-zero entry in problem should match the solution
    for i in range(9):
        for j in range(9):
            if grid[i, j] != 0 and grid[i, j] != solution[i, j]:
                return False
    
    return True


def verify_dataset(dataset_dir: str, num_samples: int = 100):
    """Verify dataset quality by checking random samples."""
    dataset_path = Path(dataset_dir)
    
    print(f"🔍 Verifying dataset: {dataset_path}")
    
    # Load data
    try:
        inputs = np.load(dataset_path / "train" / "all__inputs.npy")
        labels = np.load(dataset_path / "train" / "all__labels.npy")
        
        print(f"📊 Dataset size: {len(inputs)} samples")
        print(f"   Input shape: {inputs.shape}")
        print(f"   Label shape: {labels.shape}")
        
        # Check encoding
        print(f"   Input range: {inputs.min()}-{inputs.max()}")
        print(f"   Label range: {labels.min()}-{labels.max()}")
        
    except FileNotFoundError as e:
        print(f"❌ Dataset files not found: {e}")
        return False
    
    # Sample random indices
    num_samples = min(num_samples, len(inputs))
    indices = np.random.choice(len(inputs), num_samples, replace=False)
    
    print(f"\n🧪 Testing {num_samples} random samples...")
    
    valid_problems = 0
    valid_solutions = 0
    consistent_pairs = 0
    
    for i, idx in enumerate(indices):
        # Decode from dataset encoding (subtract 1)
        input_grid = (inputs[idx] - 1).reshape(9, 9)
        label_grid = (labels[idx] - 1).reshape(9, 9) 
        
        # Check solution validity
        if is_valid_sudoku_solution(label_grid):
            valid_solutions += 1
        
        # Check problem-solution consistency
        if is_valid_sudoku_problem(input_grid, label_grid):
            consistent_pairs += 1
            valid_problems += 1
        
        # Print first few examples
        if i < 3:
            print(f"\n📋 Sample {idx}:")
            print(f"   Problem first row: {input_grid[0]}")
            print(f"   Solution first row: {label_grid[0]}")
            print(f"   Valid solution: {is_valid_sudoku_solution(label_grid)}")
            print(f"   Consistent pair: {is_valid_sudoku_problem(input_grid, label_grid)}")
    
    # Results
    print(f"\n📈 Verification Results:")
    print(f"   Valid problems: {valid_problems}/{num_samples} ({valid_problems/num_samples*100:.1f}%)")
    print(f"   Valid solutions: {valid_solutions}/{num_samples} ({valid_solutions/num_samples*100:.1f}%)")
    print(f"   Consistent pairs: {consistent_pairs}/{num_samples} ({consistent_pairs/num_samples*100:.1f}%)")
    
    success = (valid_solutions == num_samples and consistent_pairs == num_samples)
    
    if success:
        print(f"\n✅ Dataset verification PASSED! All samples are valid.")
    else:
        print(f"\n❌ Dataset verification FAILED! Found invalid samples.")
        print(f"   Expected: All samples to be valid Sudoku problems and solutions")
        print(f"   This suggests the augmentation is still broken or dataset is corrupted")
    
    return success


def main():
    parser = argparse.ArgumentParser(description="Verify Sudoku dataset quality")
    parser.add_argument("--dataset-dir", default="./data/sudoku-extreme-fixed",
                       help="Path to dataset directory")
    parser.add_argument("--num-samples", type=int, default=100,
                       help="Number of samples to verify")
    
    args = parser.parse_args()
    
    success = verify_dataset(args.dataset_dir, args.num_samples)
    return 0 if success else 1


if __name__ == "__main__":
    exit(main()) 