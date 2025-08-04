import torch
import os
import json
from typing import Dict, Any

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

def construct_problem_tensors(problem: Dict[str, Any], dataset_dir: str) -> Dict[str, torch.Tensor]:
    """
    Construct the problem tensors from the problem dictionary.
    """

    print(problem)

    trace_file_path = os.path.join(dataset_dir, fix_file_path(problem['trace_file']))
    tensor_data = load_tensor_from_file(trace_file_path)
    h_states = tensor_data['h_states']
    l_states = tensor_data['l_states']
    for key in tensor_data.keys():
        print(key)
    q_halt_logits = tensor_data['q_halt_logits']
    q_continue_logits = tensor_data['q_continue_logits']

    return h_states, l_states, q_halt_logits, q_continue_logits


if __name__ == "__main__":
    metadata = load_dataset_metadata("data/sudoku_full_trace_dataset")
    problem = metadata['problems'][2]
    print(problem)

    output = construct_problem_tensors(problem, "data/")
    print(output)
