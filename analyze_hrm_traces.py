#!/usr/bin/env python3
"""
HRM Trace Analysis Script

This script demonstrates how to analyze recorded HRM execution traces
to understand the model's reasoning patterns and state dynamics.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from typing import Dict, List, Any
import torch


def load_traces(filepath: str) -> Dict[str, Any]:
    """Load traces from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def analyze_state_dynamics(traces_data: Dict[str, Any]):
    """Analyze state dynamics across execution."""
    print("🔍 Analyzing State Dynamics")
    print("=" * 40)
    
    for trace_id, trace in enumerate(traces_data['traces']):
        print(f"\n📊 Trace {trace_id}:")
        summary = trace['summary']
        
        print(f"   Total steps: {summary['total_steps']}")
        print(f"   Batch size: {summary['batch_size']}")
        print(f"   H-cycles: {summary['h_cycles']}")
        print(f"   L-cycles: {summary['l_cycles']}")
        print(f"   H-updates: {summary['h_updates']}")
        print(f"   L-updates: {summary['l_updates']}")
        
        # Analyze snapshots
        snapshots = trace['snapshots']
        h_update_steps = [snap['step'] for snap in snapshots if snap['is_h_update']]
        
        print(f"   H-module update steps: {h_update_steps}")
        
        # State magnitude evolution
        if 'z_H_mean' in snapshots[0]:
            h_means = [snap.get('z_H_mean', 0) for snap in snapshots]
            l_means = [snap.get('z_L_mean', 0) for snap in snapshots]
            
            print(f"   H-state mean range: {min(h_means):.4f} - {max(h_means):.4f}")
            print(f"   L-state mean range: {min(l_means):.4f} - {max(l_means):.4f}")


def analyze_halting_behavior(traces_data: Dict[str, Any]):
    """Analyze Q-head halting decisions."""
    print("\n🛑 Analyzing Halting Behavior")
    print("=" * 40)
    
    for trace_id, trace in enumerate(traces_data['traces']):
        print(f"\n📊 Trace {trace_id}:")
        
        snapshots = trace['snapshots']
        halt_decisions = []
        continue_decisions = []
        
        for snap in snapshots:
            if 'q_halt_logits' in snap and 'q_continue_logits' in snap:
                # Convert logits to probabilities
                halt_logits = np.array(snap['q_halt_logits'])
                continue_logits = np.array(snap['q_continue_logits'])
                
                # Apply sigmoid to get probabilities
                halt_probs = 1 / (1 + np.exp(-halt_logits))
                continue_probs = 1 / (1 + np.exp(-continue_logits))
                
                halt_decisions.append(halt_probs.mean())
                continue_decisions.append(continue_probs.mean())
        
        if halt_decisions:
            print(f"   Q-decisions recorded: {len(halt_decisions)}")
            print(f"   Mean halt probability: {np.mean(halt_decisions):.4f}")
            print(f"   Mean continue probability: {np.mean(continue_decisions):.4f}")
            print(f"   Halt probability trend: {halt_decisions[0]:.4f} → {halt_decisions[-1]:.4f}")


def plot_state_evolution(traces_data: Dict[str, Any], output_dir: str = "./plots"):
    """Create visualizations of state evolution."""
    print(f"\n📈 Creating State Evolution Plots")
    print("=" * 40)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    for trace_id, trace in enumerate(traces_data['traces']):
        snapshots = trace['snapshots']
        
        # Check if we have state statistics
        if not any('z_H_mean' in snap for snap in snapshots):
            print(f"   Trace {trace_id}: No state statistics available")
            continue
            
        print(f"   Creating plots for trace {trace_id}")
        
        # Extract data
        steps = [snap['step'] for snap in snapshots]
        h_means = [snap.get('z_H_mean', 0) for snap in snapshots]
        l_means = [snap.get('z_L_mean', 0) for snap in snapshots]
        h_stds = [snap.get('z_H_std', 0) for snap in snapshots]
        l_stds = [snap.get('z_L_std', 0) for snap in snapshots]
        is_h_update = [snap['is_h_update'] for snap in snapshots]
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f'HRM State Evolution - Trace {trace_id}')
        
        # Plot 1: State means
        axes[0, 0].plot(steps, h_means, 'b-', label='H-module', linewidth=2)
        axes[0, 0].plot(steps, l_means, 'r-', label='L-module', linewidth=2)
        # Mark H-updates
        h_update_steps = [step for step, is_h in zip(steps, is_h_update) if is_h]
        h_update_means = [mean for step, mean, is_h in zip(steps, h_means, is_h_update) if is_h]
        axes[0, 0].scatter(h_update_steps, h_update_means, c='blue', s=50, alpha=0.7, marker='o')
        axes[0, 0].set_title('State Means')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Mean Activation')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: State standard deviations
        axes[0, 1].plot(steps, h_stds, 'b--', label='H-module', linewidth=2)
        axes[0, 1].plot(steps, l_stds, 'r--', label='L-module', linewidth=2)
        axes[0, 1].set_title('State Standard Deviations')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Std Deviation')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: H-update pattern
        h_update_binary = [1 if is_h else 0 for is_h in is_h_update]
        axes[1, 0].bar(steps, h_update_binary, alpha=0.6, color='green')
        axes[1, 0].set_title('H-Module Updates')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('H-Update (1=Yes, 0=No)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Q-head decisions (if available)
        halt_probs = []
        for snap in snapshots:
            if 'q_halt_logits' in snap:
                halt_logit = np.array(snap['q_halt_logits']).mean()
                halt_prob = 1 / (1 + np.exp(-halt_logit))
                halt_probs.append(halt_prob)
            else:
                halt_probs.append(None)
        
        if any(p is not None for p in halt_probs):
            valid_steps = [step for step, prob in zip(steps, halt_probs) if prob is not None]
            valid_probs = [prob for prob in halt_probs if prob is not None]
            axes[1, 1].plot(valid_steps, valid_probs, 'mo-', linewidth=2, markersize=6)
            axes[1, 1].axhline(y=0.5, color='k', linestyle=':', alpha=0.5)
            axes[1, 1].set_title('Halt Probability')
            axes[1, 1].set_xlabel('Step')
            axes[1, 1].set_ylabel('P(Halt)')
            axes[1, 1].set_ylim(0, 1)
        else:
            axes[1, 1].text(0.5, 0.5, 'No Q-head data', ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Halt Probability (No Data)')
        
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = output_dir / f"hrm_trace_{trace_id}_evolution.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"   Saved: {plot_file}")
        
        plt.close()


def compare_h_vs_l_dynamics(traces_data: Dict[str, Any]):
    """Compare H-module vs L-module dynamics."""
    print(f"\n⚖️  Comparing H vs L Module Dynamics")
    print("=" * 40)
    
    for trace_id, trace in enumerate(traces_data['traces']):
        snapshots = trace['snapshots']
        
        if not any('z_H_mean' in snap for snap in snapshots):
            continue
            
        print(f"\n📊 Trace {trace_id}:")
        
        h_means = [snap.get('z_H_mean', 0) for snap in snapshots]
        l_means = [snap.get('z_L_mean', 0) for snap in snapshots]
        h_stds = [snap.get('z_H_std', 0) for snap in snapshots]
        l_stds = [snap.get('z_L_std', 0) for snap in snapshots]
        
        # Calculate variability
        h_variability = np.std(h_means)
        l_variability = np.std(l_means)
        
        print(f"   H-module mean activation: {np.mean(h_means):.4f} ± {h_variability:.4f}")
        print(f"   L-module mean activation: {np.mean(l_means):.4f} ± {l_variability:.4f}")
        print(f"   H-module avg std: {np.mean(h_stds):.4f}")
        print(f"   L-module avg std: {np.mean(l_stds):.4f}")
        
        # Ratio analysis
        if np.mean(l_means) != 0:
            h_l_ratio = np.mean(h_means) / np.mean(l_means)
            print(f"   H/L activation ratio: {h_l_ratio:.4f}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Analyze HRM execution traces")
    parser.add_argument("--traces", 
                       default="hrm_trace_test.json",
                       help="Path to traces JSON file")
    parser.add_argument("--plot-dir",
                       default="./plots",
                       help="Directory to save plots")
    parser.add_argument("--no-plots", action="store_true",
                       help="Skip plotting")
    
    args = parser.parse_args()
    
    if not Path(args.traces).exists():
        print(f"❌ Traces file not found: {args.traces}")
        print("   Run test_state_recorder.py first to generate traces")
        return 1
    
    print("🔬 HRM Trace Analyzer")
    print("=" * 50)
    print(f"Loading traces from: {args.traces}")
    
    try:
        traces_data = load_traces(args.traces)
        print(f"✅ Loaded {traces_data['num_traces']} traces")
        
        # Perform analyses
        analyze_state_dynamics(traces_data)
        analyze_halting_behavior(traces_data)
        compare_h_vs_l_dynamics(traces_data)
        
        # Create plots
        if not args.no_plots:
            try:
                plot_state_evolution(traces_data, args.plot_dir)
                print(f"\n📁 Plots saved to: {args.plot_dir}")
            except ImportError:
                print(f"\n⚠️  matplotlib not available, skipping plots")
            except Exception as e:
                print(f"\n❌ Failed to create plots: {e}")
        
        print(f"\n🎉 Analysis completed!")
        return 0
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main()) 