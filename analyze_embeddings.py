#!/usr/bin/env python3
"""
Example: Analyzing HRM Hidden State Embeddings

This script demonstrates how to load and analyze the full tensor data
saved by the HRM state recorder for embedding analysis.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def load_and_analyze_embeddings(base_filepath: str, trace_id: int = 0):
    """Load and analyze HRM embeddings from saved tensor data."""
    
    print("🔬 HRM Embedding Analysis")
    print("=" * 50)
    
    # Load tensor data
    base_path = Path(base_filepath)
    tensor_file = f"{base_path}_trace_{trace_id}.pt"
    
    if not Path(tensor_file).exists():
        print(f"❌ Tensor file not found: {tensor_file}")
        print("   Run test_state_recorder.py first to generate data")
        return
    
    print(f"📂 Loading tensor data from {tensor_file}")
    data = torch.load(tensor_file, map_location="cpu")
    
    # Extract data
    h_states = data.get('h_states')  # Shape: [num_steps, batch_size, seq_len, hidden_size]
    l_states = data.get('l_states')  # Shape: [num_steps, batch_size, seq_len, hidden_size]
    q_halt = data.get('q_halt_logits')  # Shape: [num_steps, batch_size]
    q_continue = data.get('q_continue_logits')  # Shape: [num_steps, batch_size]
    steps = data.get('steps')
    is_h_update = data.get('is_h_update')
    
    print(f"\n📊 Data Overview:")
    if h_states is not None:
        print(f"   H-states shape: {h_states.shape}")
        print(f"   H-states dtype: {h_states.dtype}")
    if l_states is not None:
        print(f"   L-states shape: {l_states.shape}")
        print(f"   L-states dtype: {l_states.dtype}")
    if q_halt is not None:
        print(f"   Q-halt shape: {q_halt.shape}")
        print(f"   Q-continue shape: {q_continue.shape}")
    
    # Analyze embedding evolution
    analyze_embedding_evolution(h_states, l_states, steps, is_h_update)
    
    # Analyze Q-head decisions
    if q_halt is not None and q_continue is not None:
        analyze_halting_decisions(q_halt, q_continue, steps)
    
    # Dimensionality analysis
    analyze_embedding_dimensionality(h_states, l_states)
    
    return data


def analyze_embedding_evolution(h_states, l_states, steps, is_h_update):
    """Analyze how embeddings evolve over time."""
    print(f"\n🔄 Embedding Evolution Analysis:")
    
    if h_states is None or l_states is None:
        print("   No state data available")
        return
    
    # Calculate norms over time
    h_norms = h_states.norm(dim=-1).mean(dim=-1)  # [num_steps, batch_size]
    l_norms = l_states.norm(dim=-1).mean(dim=-1)  # [num_steps, batch_size]
    
    h_mean_norm = h_norms.mean(dim=-1)  # [num_steps]
    l_mean_norm = l_norms.mean(dim=-1)  # [num_steps]
    
    print(f"   H-state norm range: {h_mean_norm.min():.4f} - {h_mean_norm.max():.4f}")
    print(f"   L-state norm range: {l_mean_norm.min():.4f} - {l_mean_norm.max():.4f}")
    
    # Calculate cosine similarities between consecutive steps
    if h_states.shape[0] > 1:
        h_similarities = []
        l_similarities = []
        
        for i in range(1, h_states.shape[0]):
            h_prev = h_states[i-1].flatten(start_dim=1)  # [batch_size, seq_len*hidden_size]
            h_curr = h_states[i].flatten(start_dim=1)
            h_sim = torch.cosine_similarity(h_prev, h_curr, dim=1).mean()
            h_similarities.append(h_sim.item())
            
            l_prev = l_states[i-1].flatten(start_dim=1)
            l_curr = l_states[i].flatten(start_dim=1)  
            l_sim = torch.cosine_similarity(l_prev, l_curr, dim=1).mean()
            l_similarities.append(l_sim.item())
        
        print(f"   H-state cosine similarity (consecutive steps): {np.mean(h_similarities):.4f} ± {np.std(h_similarities):.4f}")
        print(f"   L-state cosine similarity (consecutive steps): {np.mean(l_similarities):.4f} ± {np.std(l_similarities):.4f}")


def analyze_halting_decisions(q_halt, q_continue, steps):
    """Analyze Q-head halting decision patterns."""
    print(f"\n🛑 Halting Decision Analysis:")
    
    # Convert logits to probabilities
    halt_probs = torch.sigmoid(q_halt)  # [num_steps, batch_size]
    continue_probs = torch.sigmoid(q_continue)
    
    # Calculate confidence (difference between halt and continue)
    confidence = halt_probs - continue_probs  # Positive = prefer halt
    
    print(f"   Halt probability range: {halt_probs.min():.4f} - {halt_probs.max():.4f}")
    print(f"   Continue probability range: {continue_probs.min():.4f} - {continue_probs.max():.4f}")
    print(f"   Decision confidence range: {confidence.min():.4f} - {confidence.max():.4f}")
    
    # Analyze trajectory
    mean_halt_over_time = halt_probs.mean(dim=1)  # [num_steps]
    print(f"   Halt probability trend: {mean_halt_over_time[0]:.4f} → {mean_halt_over_time[-1]:.4f}")


def analyze_embedding_dimensionality(h_states, l_states):
    """Analyze the effective dimensionality of embeddings."""
    print(f"\n📐 Dimensionality Analysis:")
    
    if h_states is None or l_states is None:
        print("   No state data available")
        return
    
    # Flatten spatial dimensions and compute participation ratio
    h_flat = h_states.flatten(start_dim=0, end_dim=1).flatten(start_dim=1)  # [num_steps*batch_size, seq_len*hidden_size]
    l_flat = l_states.flatten(start_dim=0, end_dim=1).flatten(start_dim=1)
    
    def participation_ratio(X):
        """Calculate participation ratio as measure of effective dimensionality."""
        # Center the data
        X_centered = X - X.mean(dim=0)
        # Compute covariance matrix
        cov = X_centered.T @ X_centered / (X_centered.shape[0] - 1)
        # Get eigenvalues
        eigenvals = torch.linalg.eigvals(cov).real
        eigenvals = eigenvals[eigenvals > 0]  # Remove numerical zeros
        # Calculate participation ratio
        pr = (eigenvals.sum() ** 2) / (eigenvals ** 2).sum()
        return pr.item()
    
    h_pr = participation_ratio(h_flat)
    l_pr = participation_ratio(l_flat)
    
    print(f"   H-state participation ratio: {h_pr:.2f}")
    print(f"   L-state participation ratio: {l_pr:.2f}")
    print(f"   H/L dimensionality ratio: {h_pr/l_pr:.2f}")
    
    # Compare to biological findings (from paper: H~90, L~30, ratio~3)
    print(f"   Comparison to paper: H-state {'✓' if 70 < h_pr < 110 else '✗'}, L-state {'✓' if 20 < l_pr < 50 else '✗'}")


def plot_embedding_evolution(data, output_dir="./plots"):
    """Create plots of embedding evolution."""
    print(f"\n📈 Creating embedding plots...")
    
    try:
        import matplotlib.pyplot as plt
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        h_states = data.get('h_states')
        l_states = data.get('l_states')
        
        if h_states is None or l_states is None:
            print("   No state data for plotting")
            return
        
        # Plot norm evolution
        h_norms = h_states.norm(dim=-1).mean(dim=(1,2))  # [num_steps]
        l_norms = l_states.norm(dim=-1).mean(dim=(1,2))
        
        plt.figure(figsize=(10, 6))
        plt.plot(h_norms.numpy(), 'b-', label='H-module', linewidth=2)
        plt.plot(l_norms.numpy(), 'r-', label='L-module', linewidth=2)
        plt.xlabel('Step')
        plt.ylabel('Mean Embedding Norm')
        plt.title('HRM Embedding Norm Evolution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plot_file = output_dir / "embedding_norms.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"   Saved: {plot_file}")
        plt.close()
        
    except ImportError:
        print("   matplotlib not available for plotting")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Analyze HRM embedding tensors")
    parser.add_argument("--base-path", 
                       default="hrm_trace_test",
                       help="Base path to tensor files")
    parser.add_argument("--trace-id", type=int, default=0,
                       help="Which trace to analyze")
    parser.add_argument("--plot", action="store_true",
                       help="Create visualization plots")
    
    args = parser.parse_args()
    
    try:
        # Load and analyze embeddings
        data = load_and_analyze_embeddings(args.base_path, args.trace_id)
        
        if data and args.plot:
            plot_embedding_evolution(data)
        
        print(f"\n🎉 Embedding analysis completed!")
        return 0
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main()) 