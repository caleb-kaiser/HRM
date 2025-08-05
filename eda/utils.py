import torch
import os
import json
from typing import Dict, Any, Optional
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import hdbscan
import matplotlib.pyplot as plt
from umap import UMAP

def load_tensor_from_file(file_path: str) -> torch.Tensor:
    """
    Load a tensor from a file.
    """
    return torch.load(file_path)


def load_dataset_metadata(dataset_dir: str) -> Dict[str, Any]:
    """
    Load the dataset metadata from the dataset directory.
    """
    metadata_file = os.path.join(dataset_dir, "dataset_metadata.json")
    with open(metadata_file, "r") as f:
        return json.load(f)

def fix_file_path(file_path: str) -> str:
    """
    We have a naming issue with the trace files.
    """
    return file_path.replace(".pt", "_trace_0.pt")

def create_problem_tensors(problem: Dict[str, Any], dataset_dir: str) -> Dict[str, torch.Tensor]:
    """
    Extract all tensors for a given problem from its trace file.
    
    Args:
        problem: Problem dictionary from dataset metadata containing:
            - problem_id: Unique identifier for the problem
            - trace_file: Path to the trace file (relative to dataset_dir)
            - sudoku_input: The input Sudoku grid
            - sudoku_target: The target Sudoku solution
            - metadata: Problem-specific metadata
        dataset_dir: Base directory containing the dataset
        
    Returns:
        Dictionary containing all tensors and data for the problem:
        {
            'h_states': torch.Tensor,           # H-module hidden states [timesteps, batch, seq, hidden]
            'l_states': torch.Tensor,           # L-module hidden states [timesteps, batch, seq, hidden] 
            'q_halt_logits': torch.Tensor,      # Q-head halt logits [timesteps, batch]
            'q_continue_logits': torch.Tensor,  # Q-head continue logits [timesteps, batch]
            'vocab_logits': torch.Tensor,       # Vocabulary logits [timesteps, batch, seq, vocab_size]
            'steps': torch.Tensor,              # Step numbers [timesteps]
            'h_cycles': torch.Tensor,           # H-cycle numbers [timesteps]
            'l_cycles': torch.Tensor,           # L-cycle numbers [timesteps]
            'is_h_update': torch.Tensor,        # H-update flags [timesteps]
            'halted': torch.Tensor,             # Halting status [timesteps, batch]
            'metadata': dict,                   # Execution metadata
            'problem_info': dict                # Problem-specific info (sudoku grids, etc.)
        }
        
    Raises:
        FileNotFoundError: If the trace file doesn't exist
        KeyError: If required keys are missing from the trace file
    """
    # Get the trace file path
    trace_file_path = problem.get('trace_file')
    if not trace_file_path:
        raise KeyError(f"Problem {problem.get('problem_id', 'unknown')} missing 'trace_file' field")
    
    # Handle both absolute and relative paths
    if not os.path.isabs(trace_file_path):
        # If relative path, join with dataset_dir
        full_trace_path = os.path.join(dataset_dir, trace_file_path)
    else:
        full_trace_path = trace_file_path
    
    # Convert to Path for easier manipulation
    trace_path = Path(full_trace_path)
    
    # If the file doesn't exist, it might be due to the naming issue
    if not trace_path.exists():
        # Try the fixed file path
        fixed_path = fix_file_path(str(trace_path))
        trace_path = Path(fixed_path)
        
        if not trace_path.exists():
            raise FileNotFoundError(f"Trace file not found: {trace_path} (original: {full_trace_path})")
    
    # Load the tensor data
    try:
        tensor_data = torch.load(str(trace_path), map_location='cpu')
    except Exception as e:
        raise RuntimeError(f"Failed to load trace file {trace_path}: {e}")
    
    # Required tensor keys
    required_keys = ['h_states', 'l_states']
    missing_keys = [key for key in required_keys if key not in tensor_data]
    if missing_keys:
        raise KeyError(f"Missing required keys in trace file {trace_path}: {missing_keys}")
    
    # Extract all available tensors
    result = {}
    
    # Core state tensors (required)
    result['h_states'] = tensor_data['h_states']
    result['l_states'] = tensor_data['l_states']
    
    # Q-head tensors (optional but important)
    result['q_halt_logits'] = tensor_data.get('q_halt_logits')
    result['q_continue_logits'] = tensor_data.get('q_continue_logits')
    
    # Vocabulary logits (optional but important for output analysis)
    result['vocab_logits'] = tensor_data.get('vocab_logits')
    
    # Execution metadata tensors (optional)
    result['steps'] = tensor_data.get('steps')
    result['h_cycles'] = tensor_data.get('h_cycles')
    result['l_cycles'] = tensor_data.get('l_cycles')
    result['is_h_update'] = tensor_data.get('is_h_update')
    result['halted'] = tensor_data.get('halted')
    
    # Execution metadata dict (optional)
    result['metadata'] = tensor_data.get('metadata', {})
    
    # Add problem-specific information
    result['problem_info'] = {
        'problem_id': problem.get('problem_id'),
        'sudoku_input': problem.get('sudoku_input'),
        'sudoku_target': problem.get('sudoku_target'),
        'problem_metadata': problem.get('metadata', {})
    }
    
    return result

def segment_l_phases(problem):
    l_states = problem['l_states'].squeeze(1)  # [T, 82, 512]
    pooled = l_states.mean(dim=1).to(torch.float32).cpu().numpy()  # [T, 512]
    
    X_scaled = StandardScaler().fit_transform(pooled)
    X_reduced = PCA(n_components=10).fit_transform(X_scaled)
    
    clusterer = hdbscan.HDBSCAN(min_cluster_size=3, min_samples=2)
    labels = clusterer.fit_predict(X_reduced)
    
    return labels, X_reduced




def plot_umap_with_labels(X_reduced, labels, is_h_update=None, title="UMAP of L-States with Phase Labels"):
    umap_model = UMAP(n_neighbors=5, min_dist=0.3)
    X_umap = umap_model.fit_transform(X_reduced)

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_umap[:, 0], X_umap[:, 1], c=labels, cmap='tab10', s=40)
    plt.title(title)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")

    if is_h_update is not None:
        for i, flag in enumerate(is_h_update):
            if flag:
                plt.scatter(X_umap[i, 0], X_umap[i, 1], c='black', marker='x', s=60, label='H-step' if i == 0 else "")

    plt.legend(*scatter.legend_elements(), title="Cluster")
    plt.grid(True)
    plt.show()


def plot_cluster_timeline(labels, is_h_update=None, title="Cluster Timeline Over L-steps"):
    plt.figure(figsize=(12, 3))
    plt.plot(range(len(labels)), labels, drawstyle='steps-mid', marker='o')
    plt.xlabel("L-step")
    plt.ylabel("Cluster ID")
    plt.title(title)

    if is_h_update is not None:
        for i, flag in enumerate(is_h_update):
            if flag:
                plt.axvline(i, color='gray', linestyle='--', alpha=0.6)

    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_cluster_timeline_with_grid_deltas_and_q(
    labels,
    grid_states=None,
    q_halt_probs=None,
    is_h_update=None,
    title="Cluster Timeline with Grid Change and Q-Halt"
):
    T = len(labels)
    fig, ax1 = plt.subplots(figsize=(14, 4))

    # Plot cluster labels as step function
    ax1.plot(range(T), labels, drawstyle='steps-mid', marker='o', label="Cluster ID", color='blue')
    ax1.set_ylabel("Cluster ID", color='tab10_r')
    ax1.set_xlabel("L-step")
    ax1.set_title(title)
    ax1.grid(True)

    # Overlay H-step markers
    if is_h_update is not None:
        for i, flag in enumerate(is_h_update):
            if flag:
                ax1.axvline(i, color='gray', linestyle='--', alpha=0.4)

    # Compute and overlay grid deltas
    if grid_states is not None:
        deltas = []
        for i in range(1, len(grid_states)):
            prev = np.array(grid_states[i - 1])
            curr = np.array(grid_states[i])
            delta = (prev != curr).sum() / prev.size
            deltas.append(delta)
        deltas = [0] + deltas  # Add 0 for t=0
        cmap = plt.get_cmap("Reds")
        for i, d in enumerate(deltas):
            ax1.axvspan(i - 0.5, i + 0.5, color=cmap(d), alpha=0.2)

    # Overlay Q-halt probability
    if q_halt_probs is not None:
        ax2 = ax1.twinx()
        ax2.plot(range(T), q_halt_probs, color='black', linestyle='--', label='Q-halt prob')
        ax2.set_ylabel("Q-halt probability", color='black')
        ax2.set_ylim(0, 1)

    fig.tight_layout()
    plt.show()



if __name__ == "__main__":
    metadata = load_dataset_metadata("data/sudoku_trace_dataset")
    problem = metadata['problems'][0]
    print(problem)

    problem = create_problem_tensors(problem, "data/")
    labels, X_reduced = segment_l_phases(problem)
    plot_umap_with_labels(X_reduced, labels, problem['is_h_update'])
    plot_cluster_timeline(labels, problem['is_h_update'])
    plot_cluster_timeline_with_grid_deltas_and_q(labels, problem['problem_info']['sudoku_input'], problem['q_halt_logits'], problem['is_h_update'])



def hold(output):
    print("reconstructed problem object keys: ", output.keys())

    print("h_states shape: ", output['h_states'].shape)
    print("l_states shape: ", output['l_states'].shape)
    print("q_halt_logits shape: ", output['q_halt_logits'].shape)
    print("q_continue_logits shape: ", output['q_continue_logits'].shape)
    print("steps: ", output['steps'])
    print("h_cycles: ", output['h_cycles'])
    print("l_cycles: ", output['l_cycles'])
    print("is_h_update: ", output['is_h_update'])
    print("halted: ", output['halted'])
    print("metadata: ", output['metadata'])
    print("problem_info: ", output['problem_info'])