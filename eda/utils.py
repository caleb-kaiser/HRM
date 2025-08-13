import torch
import os
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import hdbscan
import matplotlib.pyplot as plt
from umap import UMAP
import ruptures as rpt
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict
import matplotlib.animation as animation

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

def load_problems_to_tensors(dataset_dir: str, num_problems=100) -> List[Dict[str, torch.Tensor]]:
    """
    Load all problems from the dataset directory and convert them to tensors.
    """
    metadata = load_dataset_metadata(dataset_dir)
    problems = metadata['problems']

    problem_tensors = [create_problem_tensors(problem, f"../data") for problem in problems[:num_problems]]
    return problem_tensors
    
def segment_l_phases(problem):
    l_states = problem['l_states'].squeeze(1)  # [T, 82, 512]
    pooled = l_states.mean(dim=1).to(torch.float32).cpu().numpy()  # [T, 512]
    
    X_scaled = StandardScaler().fit_transform(pooled)
    X_reduced = PCA(n_components=10).fit_transform(X_scaled)
    
    clusterer = hdbscan.HDBSCAN(min_cluster_size=3, min_samples=2)
    labels = clusterer.fit_predict(X_reduced)
    
    return labels, X_reduced

def breakpoints_to_labels(breakpoints, T):
    """
    Convert list of breakpoints into flat segment labels.

    Args:
        breakpoints (List[int]): Output from ruptures.predict()
        T (int): Total number of steps

    Returns:
        np.ndarray: shape [T] segment index for each step
    """
    labels = np.zeros(T, dtype=int)
    prev = 0
    for i, bp in enumerate(breakpoints):
        labels[prev:bp] = i
        prev = bp
    return labels



def segment_with_ruptures(X, model="rbf", penalty=10):
    """
    Segments the latent trajectory using Ruptures.
    
    Args:
        X (np.ndarray): shape [T, D] - pooled L-states
        model (str): cost model to use ("l2", "rbf", etc.)
        penalty (float): penalty for creating a new segment

    Returns:
        List[int]: Change point indices (e.g., [10, 20, 31] means 3 segments)
    """
    algo = rpt.Pelt(model=model).fit(X)
    breakpoints = algo.predict(pen=penalty)
    return breakpoints



def segment_l_phases_rupture(problem, plot=True):
    l_states = problem['l_states'].squeeze(1)  # [T, 82, 512]
    pooled = l_states.mean(dim=1).to(torch.float32).cpu().numpy()  # [T, 512]
    breakpoints = segment_with_ruptures(pooled, model="l2", penalty=8)
    rupture_labels = breakpoints_to_labels(breakpoints, pooled.shape[0])

    # Plot like HDBSCAN results
    if plot:
        plot_cluster_timeline(rupture_labels, is_h_update=problem['is_h_update'], title="Ruptures Segmentation Timeline")

    return breakpoints, rupture_labels



def extract_predicted_grids(vocab_logits):
    """
    Convert vocab_logits to a list of 9x9 predicted Sudoku grids per H-step.

    Args:
        vocab_logits (torch.Tensor): [H, 1, 81, 11] tensor of logits

    Returns:
        List[np.ndarray]: List of [9, 9] integer grids
    """
    logits = vocab_logits.squeeze(1)          # -> [H, 81, 11]
    preds = torch.argmax(logits, dim=-1)      # -> [H, 81]
    grids = preds.view(-1, 9, 9) - 1          # -> [H, 9, 9]
    return [grid.cpu().numpy() for grid in grids]

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



def visualize_sudoku_grids_over_time(grids, title_prefix="HRM Predicted Grid", cmap="viridis", figsize=(12, 12)):
    """
    Visualizes a sequence of Sudoku grids (e.g., one per H-step) as heatmaps.

    Args:
        grids (List[np.ndarray]): List of 9x9 Sudoku grids (each as np.array)
        title_prefix (str): Prefix for subplot titles
        cmap (str): Matplotlib colormap
        figsize (tuple): Size of the full figure
    """
    n = len(grids)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.array(axes).reshape(-1)

    for i, grid in enumerate(grids):
        ax = axes[i]
        im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=9)
        ax.set_title(f"{title_prefix} @ H-step {i}")
        ax.set_xticks([])
        ax.set_yticks([])

        # Optional: overlay digit values
        for y in range(9):
            for x in range(9):
                val = grid[y, x]
                if val != 0:
                    ax.text(x, y, str(val), ha='center', va='center', color='white', fontsize=10, weight='bold')

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


def visualize_grid_deltas(grids, title_prefix="Grid Changes", cmap="coolwarm", figsize=(12, 12)):
    """
    Visualizes cell-by-cell changes between consecutive Sudoku grids as heatmaps.

    Args:
        grids (List[np.ndarray]): List of 9x9 Sudoku grids
        title_prefix (str): Prefix for each subplot title
        cmap (str): Colormap for visualizing changes
        figsize (tuple): Figure size
    """
    n = len(grids)
    delta_grids = []

    for i in range(1, n):
        prev = grids[i - 1]
        curr = grids[i]
        delta = (curr != prev).astype(int)  # 1 where changed, 0 where not
        delta_grids.append(delta)

    # Plot deltas
    cols = min(len(delta_grids), 4)
    rows = (len(delta_grids) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.array(axes).reshape(-1)

    for i, delta in enumerate(delta_grids):
        ax = axes[i]
        im = ax.imshow(delta, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(f"{title_prefix} H{i}→H{i+1}")
        ax.set_xticks([])
        ax.set_yticks([])

        # Overlay changed cell markers
        for y in range(9):
            for x in range(9):
                if delta[y, x] == 1:
                    ax.text(x, y, "•", ha='center', va='center', color='black', fontsize=12, fontweight='bold')

    # Hide unused plots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


def get_segment_ranges(breakpoints, T):
    start = 0
    ranges = []
    for b in breakpoints:
        ranges.append((start, b))
        start = b
    if start < T:
        ranges.append((start, T))  # final segment if ruptures didn't end at T
    return ranges

def cluster_segments(problem_tensors, n_clusters=10):
    all_embeddings = []
    segment_counts = []

    for problem in problem_tensors:
        # Segment and get ranges
        problem['segments'], problem['segment_labels'] = segment_l_phases_rupture(problem, plot=False)
        problem['segment_ranges'] = get_segment_ranges(problem['segments'], T=problem['steps'].shape[0])

        pooled_embeddings = []

        # Mean-pool each segment across time and cells: [T_seg, 82, 512] → [512]
        for start, end in problem['segment_ranges']:
            segment_tensor = problem['l_states'][start:end]  # [T_seg, 1, 82, 512]
            segment_tensor = segment_tensor.squeeze(1)       # [T_seg, 82, 512]
            segment_embedding = segment_tensor.mean(dim=(0, 1))  # → [512]
            pooled_embeddings.append(segment_embedding)

        # Store per-problem
        problem['segment_embeddings_pooled'] = torch.stack(pooled_embeddings)  # [num_segments, 512]
        all_embeddings.append(problem['segment_embeddings_pooled'])
        segment_counts.append(len(pooled_embeddings))

    # Concatenate all segment embeddings across all problems
    combined_embeddings = torch.cat(all_embeddings, dim=0).to(torch.float32)  # [total_segments, 512]

    # Run KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    combined_labels = kmeans.fit_predict(combined_embeddings.cpu().numpy())

    # Assign cluster IDs back to each problem
    offset = 0
    for problem, n_segments in zip(problem_tensors, segment_counts):
        problem['segment_cluster_ids'] = combined_labels[offset:offset + n_segments]
        offset += n_segments

    return problem_tensors

def extract_h_step_deltas(clustered_tensors):
    cluster_to_deltas = defaultdict(list)

    for problem in clustered_tensors:
        h_steps = problem["vocab_logits"]  # shape: [H, 1, 81, 11]
        for cluster_id, seg_id in zip(problem["segment_cluster_ids"], problem["segment_labels"]):
            h_logits = h_steps[seg_id]  # You may need to slice
            decoded = extract_predicted_grids(h_logits)[0] # List of 9x9 grids

            for i in range(len(decoded) - 1):
                delta = (decoded[i+1] != decoded[i])  # Boolean mask
                cluster_to_deltas[cluster_id].append(delta)

    return cluster_to_deltas


def get_grid_from_logits(vocab_logits_t):
    """Convert [81, 11] logits → [9, 9] grid of predicted tokens."""
    preds = vocab_logits_t.argmax(dim=-1)  # [81]
    return preds.view(9, 9).cpu()


def compute_delta_mask(grid1, grid2):
    """Returns binary mask of changed cells between two 9x9 grids."""
    return (grid1 != grid2).int()


def plot_mean_delta_heatmaps_by_cluster(problem_tensors, num_clusters):
    cluster_deltas = defaultdict(list)

    for problem in problem_tensors:
        vocab_logits = problem['vocab_logits']  # [T, 1, 81, 11]
        segment_ranges = problem['segment_ranges']
        cluster_ids = problem['segment_cluster_ids']

        for (start, end), cluster_id in zip(segment_ranges, cluster_ids):
            if end >= vocab_logits.shape[0]:
                continue  # skip segments that go out of bounds

            g_start = get_grid_from_logits(vocab_logits[start][0])  # [9, 9]
            g_end   = get_grid_from_logits(vocab_logits[end - 1][0])
            delta_mask = compute_delta_mask(g_start, g_end).numpy()  # [9, 9]

            cluster_deltas[cluster_id].append(delta_mask)

    # Plot heatmaps
    num_cols = 5
    num_rows = (num_clusters + num_cols - 1) // num_cols
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3, num_rows * 3))

    for cluster_id in range(num_clusters):
        row, col = divmod(cluster_id, num_cols)
        ax = axs[row, col] if num_rows > 1 else axs[col]
        if cluster_id not in cluster_deltas:
            ax.axis('off')
            continue
        masks = np.stack(cluster_deltas[cluster_id])  # [num_segments, 9, 9]
        mean_mask = masks.mean(axis=0)  # [9, 9]

        im = ax.imshow(mean_mask, cmap='viridis', vmin=0, vmax=1)
        ax.set_title(f"Cluster {cluster_id}")
        ax.axis('off')

    plt.tight_layout()
    plt.suptitle("Mean Delta Heatmaps by Cluster", fontsize=16, y=1.02)
    plt.colorbar(im, ax=axs.ravel().tolist(), shrink=0.6)
    plt.show()


import matplotlib.pyplot as plt
import numpy as np

def describe_grid_changes(input_grid, output_grid):
    changes = []
    for i in range(9):
        for j in range(9):
            before = input_grid[i, j].item()
            after = output_grid[i, j].item()
            if before != after:
                changes.append(f"Wrote {after} at (row={i}, col={j}), replacing {before}")
    return changes

def visualize_grid_with_numbers(grid, title=""):
    fig, ax = plt.subplots()
    ax.imshow(np.zeros_like(grid), cmap="Greys", vmin=0, vmax=9)  # blank background
    for i in range(9):
        for j in range(9):
            val = grid[i, j].item()
            if val != 0:
                ax.text(j, i, str(val), va='center', ha='center', fontsize=12)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    return fig

def show_segment_change_with_description(input_grid, output_grid, delta=None, verbose=True):
    if delta is None:
        delta = (input_grid != output_grid).int()

    changes = describe_grid_changes(input_grid, output_grid)
    
    if verbose:
        print("Detected Changes:")
        for change in changes:
            print("•", change)

    # Plot input/output/delta
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for ax, grid, name in zip(axes, [input_grid, output_grid, delta], 
                              ["Input", "Output", "Delta (Highlight)"]):
        ax.imshow(np.zeros((9, 9)), cmap="Greys", vmin=0, vmax=9)
        for i in range(9):
            for j in range(9):
                val = grid[i, j].item()
                if val != 0:
                    ax.text(j, i, str(val), ha='center', va='center', fontsize=12, 
                            color="red" if name == "Delta (Highlight)" else "black")
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    metadata = load_dataset_metadata("data/sudoku_trace_dataset")
    problem = metadata['problems'][0]
    print(problem)

    problem = create_problem_tensors(problem, "data/")
    for key in problem.keys():
        print(key)
        #print(problem[key].shape)
       # print(problem[key])
        #print("-"*100

    print(problem['vocab_logits'].shape)
    #labels, X_reduced = segment_l_phases(problem)
    #plot_umap_with_labels(X_reduced, labels, problem['is_h_update'])
    #plot_cluster_timeline(labels, problem['is_h_update'])
    #plot_cluster_timeline_with_grid_deltas_and_q(labels, problem['problem_info']['sudoku_input'], problem['q_halt_logits'], problem['is_h_update'])



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