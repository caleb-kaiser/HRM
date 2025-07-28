#!/usr/bin/env python3
"""
HRM State Recorder

This module provides functionality to record and analyze the hidden states
of Hierarchical Reasoning Models (HRM) during execution.

Captures:
- L-module hidden states at each timestep
- H-module hidden states at each timestep  
- Q-head halting decisions
- Execution metadata (cycles, steps, etc.)
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import torch
import torch.nn as nn
from pathlib import Path
import json
import numpy as np
import datetime


@dataclass
class HRMStateSnapshot:
    """Single timestep snapshot of HRM states."""
    step: int
    h_cycle: int
    l_cycle: int
    
    # Hidden states
    z_H: torch.Tensor  # High-level module state
    z_L: torch.Tensor  # Low-level module state
    
    # Q-head outputs (if available)
    q_halt_logits: Optional[torch.Tensor] = None
    q_continue_logits: Optional[torch.Tensor] = None
    
    # Metadata
    is_h_update: bool = False  # Whether H-module was updated this step
    halted: Optional[torch.Tensor] = None  # Halting status per batch element


@dataclass  
class HRMExecutionTrace:
    """Complete execution trace for a batch of inputs."""
    snapshots: List[HRMStateSnapshot] = field(default_factory=list)
    
    # Input metadata
    batch_size: int = 0
    seq_len: int = 0
    
    # Model config
    h_cycles: int = 0
    l_cycles: int = 0
    halt_max_steps: int = 0
    
    # Final outputs
    final_logits: Optional[torch.Tensor] = None
    final_halt_decisions: Optional[torch.Tensor] = None

    def add_snapshot(self, snapshot: HRMStateSnapshot):
        """Add a state snapshot to the trace."""
        self.snapshots.append(snapshot)

    def get_h_states(self) -> List[torch.Tensor]:
        """Get all H-module states in order."""
        return [snap.z_H for snap in self.snapshots]
    
    def get_l_states(self) -> List[torch.Tensor]:
        """Get all L-module states in order."""
        return [snap.z_L for snap in self.snapshots]
    
    def get_q_decisions(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Get all Q-head decisions (halt, continue) in order."""
        decisions = []
        for snap in self.snapshots:
            if snap.q_halt_logits is not None and snap.q_continue_logits is not None:
                decisions.append((snap.q_halt_logits, snap.q_continue_logits))
        return decisions

    def get_execution_summary(self) -> Dict[str, Any]:
        """Get a summary of the execution."""
        return {
            "total_steps": len(self.snapshots),
            "batch_size": self.batch_size,
            "seq_len": self.seq_len,
            "h_cycles": self.h_cycles,
            "l_cycles": self.l_cycles,
            "halt_max_steps": self.halt_max_steps,
            "h_updates": sum(1 for snap in self.snapshots if snap.is_h_update),
            "l_updates": len(self.snapshots),
        }


class HRMStateRecorder:
    """Records HRM hidden states during execution."""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.traces: List[HRMExecutionTrace] = []
        self.current_trace: Optional[HRMExecutionTrace] = None
        self._hooks = []
        
    def start_recording(self, model: nn.Module) -> HRMExecutionTrace:
        """Start recording a new execution trace."""
        self.current_trace = HRMExecutionTrace()
        self.traces.append(self.current_trace)
        
        # Extract model config
        if hasattr(model, 'config'):
            config = model.config
            self.current_trace.h_cycles = getattr(config, 'H_cycles', 0)
            self.current_trace.l_cycles = getattr(config, 'L_cycles', 0) 
            self.current_trace.halt_max_steps = getattr(config, 'halt_max_steps', 0)
        
        return self.current_trace
    
    def stop_recording(self):
        """Stop recording the current trace."""
        self.current_trace = None
        self._remove_hooks()
    
    def record_snapshot(self, 
                       step: int,
                       h_cycle: int, 
                       l_cycle: int,
                       z_H: torch.Tensor,
                       z_L: torch.Tensor,
                       q_halt_logits: Optional[torch.Tensor] = None,
                       q_continue_logits: Optional[torch.Tensor] = None,
                       is_h_update: bool = False,
                       halted: Optional[torch.Tensor] = None):
        """Record a state snapshot."""
        if self.current_trace is None:
            return
            
        snapshot = HRMStateSnapshot(
            step=step,
            h_cycle=h_cycle,
            l_cycle=l_cycle,
            z_H=z_H.detach().cpu() if z_H is not None else None,
            z_L=z_L.detach().cpu() if z_L is not None else None,
            q_halt_logits=q_halt_logits.detach().cpu() if q_halt_logits is not None else None,
            q_continue_logits=q_continue_logits.detach().cpu() if q_continue_logits is not None else None,
            is_h_update=is_h_update,
            halted=halted.detach().cpu() if halted is not None else None
        )
        
        self.current_trace.add_snapshot(snapshot)
        
        # Update trace metadata
        if z_H is not None:
            self.current_trace.batch_size = z_H.shape[0]
            self.current_trace.seq_len = z_H.shape[1]

    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def clear_traces(self):
        """Clear all recorded traces."""
        self.traces.clear()
        self.current_trace = None

    def save_traces(self, filepath: str, include_states: bool = False):
        """Save traces to file."""
        filepath = Path(filepath)
        
        data = {
            "num_traces": len(self.traces),
            "traces": []
        }
        
        for i, trace in enumerate(self.traces):
            trace_data = {
                "trace_id": i,
                "summary": trace.get_execution_summary(),
                "snapshots": []
            }
            
            for j, snapshot in enumerate(trace.snapshots):
                snap_data = {
                    "snapshot_id": j,
                    "step": snapshot.step,
                    "h_cycle": snapshot.h_cycle,
                    "l_cycle": snapshot.l_cycle,
                    "is_h_update": snapshot.is_h_update,
                }
                
                if include_states:
                    # Convert tensors to lists for JSON serialization
                    if snapshot.z_H is not None:
                        snap_data["z_H_shape"] = list(snapshot.z_H.shape)
                        snap_data["z_H_mean"] = float(snapshot.z_H.mean())
                        snap_data["z_H_std"] = float(snapshot.z_H.std())
                    
                    if snapshot.z_L is not None:
                        snap_data["z_L_shape"] = list(snapshot.z_L.shape)
                        snap_data["z_L_mean"] = float(snapshot.z_L.mean())
                        snap_data["z_L_std"] = float(snapshot.z_L.std())
                        
                    if snapshot.q_halt_logits is not None:
                        snap_data["q_halt_logits"] = snapshot.q_halt_logits.tolist()
                    
                    if snapshot.q_continue_logits is not None:
                        snap_data["q_continue_logits"] = snapshot.q_continue_logits.tolist()
                
                trace_data["snapshots"].append(snap_data)
            
            data["traces"].append(trace_data)
        
        with open(str(filepath), 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"Saved {len(self.traces)} traces to {filepath}")

    def save_traces_with_tensors(self, base_filepath: str, save_format: str = "pt"):
        """
        Save traces with full tensor data for embedding analysis.
        
        Args:
            base_filepath: Base path (without extension) for saving files
            save_format: "pt" for PyTorch, "npz" for NumPy, "both" for both formats
        """
        base_path = Path(base_filepath)
        base_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save metadata as JSON
        metadata_file = f"{base_path}_metadata.json"
        self.save_traces(metadata_file, include_states=False)
        
        # Save tensor data
        for trace_id, trace in enumerate(self.traces):
            trace_data = {
                'h_states': [],
                'l_states': [], 
                'q_halt_logits': [],
                'q_continue_logits': [],
                'steps': [],
                'h_cycles': [],
                'l_cycles': [],
                'is_h_update': [],
                'halted': []
            }
            
            for snapshot in trace.snapshots:
                # Collect tensor data
                if snapshot.z_H is not None:
                    trace_data['h_states'].append(snapshot.z_H)
                if snapshot.z_L is not None:
                    trace_data['l_states'].append(snapshot.z_L)
                if snapshot.q_halt_logits is not None:
                    trace_data['q_halt_logits'].append(snapshot.q_halt_logits)
                if snapshot.q_continue_logits is not None:
                    trace_data['q_continue_logits'].append(snapshot.q_continue_logits)
                if snapshot.halted is not None:
                    trace_data['halted'].append(snapshot.halted)
                    
                # Collect metadata
                trace_data['steps'].append(snapshot.step)
                trace_data['h_cycles'].append(snapshot.h_cycle)
                trace_data['l_cycles'].append(snapshot.l_cycle)
                trace_data['is_h_update'].append(snapshot.is_h_update)
            
            # Stack tensors if available
            stacked_data = {}
            if trace_data['h_states']:
                stacked_data['h_states'] = torch.stack(trace_data['h_states'])
            if trace_data['l_states']:
                stacked_data['l_states'] = torch.stack(trace_data['l_states'])
            if trace_data['q_halt_logits']:
                stacked_data['q_halt_logits'] = torch.stack(trace_data['q_halt_logits'])
            if trace_data['q_continue_logits']:
                stacked_data['q_continue_logits'] = torch.stack(trace_data['q_continue_logits'])
            if trace_data['halted']:
                stacked_data['halted'] = torch.stack(trace_data['halted'])
                
            # Add metadata
            stacked_data['steps'] = torch.tensor(trace_data['steps'])
            stacked_data['h_cycles'] = torch.tensor(trace_data['h_cycles'])
            stacked_data['l_cycles'] = torch.tensor(trace_data['l_cycles'])
            stacked_data['is_h_update'] = torch.tensor(trace_data['is_h_update'])
            
            # Save in requested format(s)
            if save_format in ["pt", "both"]:
                pt_file = f"{base_path}_trace_{trace_id}.pt"
                torch.save(stacked_data, str(pt_file))
                print(f"Saved trace {trace_id} tensors to {pt_file}")
                
            if save_format in ["npz", "both"]:
                npz_file = f"{base_path}_trace_{trace_id}.npz"
                numpy_data = {}
                for key, tensor in stacked_data.items():
                    if isinstance(tensor, torch.Tensor):
                        # Handle BFloat16 and other unsupported dtypes for NumPy
                        if tensor.dtype == torch.bfloat16:
                            numpy_data[key] = tensor.cpu().float().numpy()
                        elif tensor.dtype == torch.float16:
                            numpy_data[key] = tensor.cpu().float().numpy()
                        else:
                            numpy_data[key] = tensor.cpu().numpy()
                    else:
                        numpy_data[key] = tensor
                        
                import numpy as np
                np.savez_compressed(str(npz_file), **numpy_data)
                print(f"Saved trace {trace_id} tensors to {npz_file}")
                if any(tensor.dtype in [torch.bfloat16, torch.float16] for tensor in stacked_data.values() if isinstance(tensor, torch.Tensor)):
                    print(f"   Note: BFloat16/Float16 tensors converted to Float32 for NumPy compatibility")
        
        print(f"\n✅ Saved {len(self.traces)} traces with full tensor data")
        print(f"   Metadata: {metadata_file}")
        print(f"   Tensors: {base_path}_trace_*.{save_format}")

    def load_traces_with_tensors(self, base_filepath: str, trace_id: int = 0, load_format: str = "pt"):
        """
        Load traces with full tensor data for analysis.
        
        Args:
            base_filepath: Base path used when saving
            trace_id: Which trace to load (default: 0)
            load_format: "pt" for PyTorch, "npz" for NumPy
            
        Returns:
            Dictionary with tensor data and metadata
        """
        base_path = Path(base_filepath)
        
        if load_format == "pt":
            tensor_file = f"{base_path}_trace_{trace_id}.pt"
            data = torch.load(str(tensor_file), map_location="cpu")
        elif load_format == "npz":
            tensor_file = f"{base_path}_trace_{trace_id}.npz"
            import numpy as np
            npz_data = np.load(str(tensor_file))
            # Convert back to PyTorch tensors
            # Note: BFloat16 tensors were converted to Float32 for NumPy compatibility
            data = {key: torch.from_numpy(npz_data[key]) for key in npz_data.keys()}
        else:
            raise ValueError(f"Unsupported load_format: {load_format}")
            
        print(f"Loaded trace {trace_id} from {tensor_file}")
        return data

    def upload_traces_to_comet(self, 
                              experiment,
                              artifact_name: str = "hrm-traces",
                              artifact_type: str = "dataset",
                              version: Optional[str] = None,
                              aliases: Optional[List[str]] = None,
                              description: str = "HRM execution traces with hidden states",
                              model_checkpoint: Optional[str] = None,
                              include_tensors: bool = True,
                              tensor_format: str = "pt") -> str:
        """
        Upload HRM traces to Comet ML as a versioned artifact.
        
        Args:
            experiment: Comet experiment object (from comet_ml.start() or comet_ml.Experiment())
            artifact_name: Name for the artifact in Comet
            artifact_type: Type of artifact (default: "dataset")
            version: Specific version string, if None auto-generated
            aliases: List of aliases for this version (e.g., ["latest", "sudoku-experiment"])
            description: Description of the traces
            model_checkpoint: Path to model checkpoint used (for metadata)
            include_tensors: Whether to include full tensor data
            tensor_format: Format for tensors ("pt", "npz", or "both")
            
        Returns:
            The version string of the uploaded artifact
        """
        try:
            import comet_ml
        except ImportError:
            raise ImportError("comet_ml is required for Comet integration. Install with: pip install comet_ml")
        
        if not self.traces:
            raise ValueError("No traces to upload. Run some recordings first.")
        
        print(f"🚀 Uploading {len(self.traces)} HRM traces to Comet ML...")
        
        # Create temporary directory for staging files
        staging_dir = Path(f"./comet_staging_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
        staging_dir.mkdir(exist_ok=True)
        
        try:
            # Create artifact with metadata
            artifact_metadata = {
                "created_at": datetime.datetime.utcnow().isoformat(),
                "num_traces": len(self.traces),
                "description": description,
                "recorder_version": "1.0.0",
                "tensor_format": tensor_format if include_tensors else "metadata_only"
            }
            
            if model_checkpoint:
                artifact_metadata["model_checkpoint"] = str(model_checkpoint)
            
            # Add trace summaries to metadata
            trace_summaries = []
            for i, trace in enumerate(self.traces):
                summary = trace.get_execution_summary()
                summary["trace_id"] = i
                trace_summaries.append(summary)
            artifact_metadata["trace_summaries"] = trace_summaries
            
            # Create Comet artifact
            artifact = comet_ml.Artifact(
                name=artifact_name,
                artifact_type=artifact_type,
                version=version,
                aliases=aliases or ["latest"],
                metadata=artifact_metadata
            )
            
            # Save and add metadata file
            metadata_file = staging_dir / "traces_metadata.json"
            self.save_traces(str(metadata_file), include_states=True)
            artifact.add(str(metadata_file), logical_path="traces_metadata.json", 
                        metadata={"asset_type": "metadata", "format": "json"})
            
            # Save and add tensor data if requested
            if include_tensors:
                base_path = staging_dir / "traces"
                self.save_traces_with_tensors(str(base_path), save_format=tensor_format)
                
                # Add all generated files
                for trace_id in range(len(self.traces)):
                    if tensor_format in ["pt", "both"]:
                        pt_file = f"{base_path}_trace_{trace_id}.pt"
                        if Path(pt_file).exists():
                            artifact.add(pt_file, logical_path=f"tensors/trace_{trace_id}.pt",
                                       metadata={"asset_type": "tensors", "format": "pytorch", "trace_id": trace_id})
                    
                    if tensor_format in ["npz", "both"]:
                        npz_file = f"{base_path}_trace_{trace_id}.npz"
                        if Path(npz_file).exists():
                            artifact.add(npz_file, logical_path=f"tensors/trace_{trace_id}.npz",
                                       metadata={"asset_type": "tensors", "format": "numpy", "trace_id": trace_id})
            
            # Log the artifact to Comet
            experiment.log_artifact(artifact)
            
            # Get the actual version that was created
            logged_version = getattr(artifact, 'version', 'unknown')
            
            print(f"✅ Successfully uploaded artifact '{artifact_name}' version {logged_version}")
            print(f"   📊 {len(self.traces)} traces with metadata")
            if include_tensors:
                print(f"   🧠 Full tensor data in {tensor_format} format")
            print(f"   🏷️  Aliases: {aliases or ['latest']}")
            
            return logged_version
            
        finally:
            # Clean up staging directory
            import shutil
            if staging_dir.exists():
                shutil.rmtree(staging_dir)
                print(f"🧹 Cleaned up staging directory")

    def download_traces_from_comet(self,
                                  experiment, 
                                  artifact_name: str,
                                  version_or_alias: Optional[str] = None,
                                  download_tensors: bool = True,
                                  local_path: str = "./downloaded_traces") -> Dict[str, Any]:
        """
        Download HRM traces from a Comet ML artifact.
        
        Args:
            experiment: Comet experiment object
            artifact_name: Name of the artifact to download
            version_or_alias: Specific version or alias, if None gets latest
            download_tensors: Whether to download tensor data
            local_path: Local path to download to
            
        Returns:
            Dictionary with metadata and paths to downloaded files
        """
        try:
            import comet_ml
        except ImportError:
            raise ImportError("comet_ml is required for Comet integration. Install with: pip install comet_ml")
        
        print(f"📥 Downloading HRM traces from Comet ML...")
        print(f"   Artifact: {artifact_name}")
        if version_or_alias:
            print(f"   Version/Alias: {version_or_alias}")
        
        # Get the artifact
        logged_artifact = experiment.get_artifact(artifact_name, version_or_alias=version_or_alias)
        
        # Download to local path
        download_path = Path(local_path)
        download_path.mkdir(parents=True, exist_ok=True)
        
        local_artifact = logged_artifact.download(str(download_path))
        
        # Parse the downloaded content
        result = {
            "artifact_name": artifact_name,
            "version": logged_artifact.version,
            "aliases": list(logged_artifact.aliases) if logged_artifact.aliases else [],
            "metadata": logged_artifact.metadata,
            "download_path": str(download_path),
            "files": {}
        }
        
        # Find downloaded files
        metadata_file = download_path / "traces_metadata.json"
        if metadata_file.exists():
            result["files"]["metadata"] = str(metadata_file)
            print(f"✅ Downloaded metadata: {metadata_file}")
        
        if download_tensors:
            tensor_files = []
            for tensor_file in download_path.glob("tensors/trace_*.pt"):
                tensor_files.append(str(tensor_file))
            for tensor_file in download_path.glob("tensors/trace_*.npz"):
                tensor_files.append(str(tensor_file))
            
            result["files"]["tensors"] = tensor_files
            if tensor_files:
                print(f"✅ Downloaded {len(tensor_files)} tensor files")
        
        # Load the traces into this recorder
        if metadata_file.exists():
            self.load_traces_from_metadata(str(metadata_file))
            print(f"✅ Loaded {len(self.traces)} traces into recorder")
        
        print(f"🎉 Successfully downloaded artifact version {logged_artifact.version}")
        return result

    def load_traces_from_metadata(self, metadata_file: str):
        """Load traces from a metadata JSON file (without full tensors)."""
        with open(metadata_file, 'r') as f:
            data = json.load(f)
        
        self.traces.clear()
        for trace_data in data.get("traces", []):
            trace = HRMExecutionTrace()
            summary = trace_data.get("summary", {})
            
            # Set trace metadata
            trace.batch_size = summary.get("batch_size", 0)
            trace.seq_len = summary.get("seq_len", 0)
            trace.h_cycles = summary.get("h_cycles", 0)
            trace.l_cycles = summary.get("l_cycles", 0)
            trace.halt_max_steps = summary.get("halt_max_steps", 0)
            
            # Create placeholder snapshots (without tensor data)
            for snap_data in trace_data.get("snapshots", []):
                snapshot = HRMStateSnapshot(
                    step=snap_data["step"],
                    h_cycle=snap_data["h_cycle"],
                    l_cycle=snap_data["l_cycle"],
                    z_H=None,  # Placeholder
                    z_L=None,  # Placeholder
                    is_h_update=snap_data.get("is_h_update", False)
                )
                trace.snapshots.append(snapshot)
            
            self.traces.append(trace)

    @staticmethod
    def create_comet_experiment(project_name: str = "hrm-traces", 
                               experiment_name: Optional[str] = None,
                               tags: Optional[List[str]] = None,
                               api_key: Optional[str] = None,
                               workspace: Optional[str] = None):
        """
        Create a Comet ML experiment for HRM trace logging.
        
        Args:
            project_name: Comet project name
            experiment_name: Name for this experiment
            tags: Tags to add to the experiment
            api_key: Comet API key (if not set via environment)
            workspace: Comet workspace name
            
        Returns:
            Comet experiment object
        """
        try:
            import comet_ml
        except ImportError:
            raise ImportError("comet_ml is required for Comet integration. Install with: pip install comet_ml")
        
        # Set API key if provided
        if api_key:
            comet_ml.init(api_key=api_key)
        
        experiment = comet_ml.Experiment(
            project_name=project_name,
            workspace=workspace
        )
        
        if experiment_name:
            experiment.set_name(experiment_name)
        
        if tags:
            for tag in tags:
                experiment.add_tag(tag)
        
        # Log some basic info
        experiment.log_parameter("recorder_version", "1.0.0")
        experiment.log_parameter("framework", "pytorch")
        
        print(f"🧪 Created Comet experiment: {experiment.get_name()}")
        print(f"   Project: {project_name}")
        if workspace:
            print(f"   Workspace: {workspace}")
        print(f"   URL: {experiment.url}")
        
        return experiment

    def get_latest_trace(self) -> Optional[HRMExecutionTrace]:
        """Get the most recent execution trace."""
        return self.traces[-1] if self.traces else None


class HRMRecordingWrapper(nn.Module):
    """Wrapper that instruments an HRM model for state recording."""
    
    def __init__(self, model: nn.Module, recorder: HRMStateRecorder, config=None):
        super().__init__()
        self.model = model
        self.recorder = recorder
        self.config = config  # Store config separately
        self._step_counter = 0
        
    def _get_model_config(self):
        """Get the model config from various possible locations."""
        # If config was provided directly, use it
        if self.config is not None:
            return self.config
            
        # Try to get config from the model itself
        if hasattr(self.model, 'config'):
            return self.model.config
        
        # If model is wrapped (e.g., in ACTLossHead), try to get config from inner model
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'config'):
            return self.model.model.config
            
        # If no config found, return None (will use defaults)
        return None
        
    def forward(self, carry, batch, return_keys=None):
        """Forward pass with state recording."""
        # Start recording if not already started
        if self.recorder.current_trace is None:
            self.recorder.start_recording(self.model)
        
        # Reset step counter for new sequences
        if hasattr(carry, 'halted') and carry.halted.all():
            self._step_counter = 0
        
        # Get model config for cycles
        model_config = self._get_model_config()
        if model_config is not None:
            h_cycles = getattr(model_config, 'H_cycles', 2)
            l_cycles = getattr(model_config, 'L_cycles', 2)
        else:
            # Default values if no config available
            h_cycles = 2
            l_cycles = 2
        
        # Record initial state
        if hasattr(carry, 'inner_carry'):
            initial_z_H = carry.inner_carry.z_H
            initial_z_L = carry.inner_carry.z_L
        else:
            initial_z_H = getattr(carry, 'z_H', None)
            initial_z_L = getattr(carry, 'z_L', None)
            
        # Record the initial state
        self.recorder.record_snapshot(
            step=self._step_counter,
            h_cycle=0,
            l_cycle=0,
            z_H=initial_z_H,
            z_L=initial_z_L,
            is_h_update=False
        )
        
        # Forward pass
        new_carry, loss, metrics, preds, all_finish = self.model(carry=carry, batch=batch, return_keys=return_keys)
        
        # Combine outputs for compatibility
        outputs = {
            'loss': loss,
            'metrics': metrics, 
            'preds': preds,
            'all_finish': all_finish
        }
        
        # Add any outputs that might be in metrics (like q_halt_logits)
        if metrics:
            outputs.update(metrics)
        
        # Record final state and Q-head outputs
        if hasattr(new_carry, 'inner_carry'):
            final_z_H = new_carry.inner_carry.z_H
            final_z_L = new_carry.inner_carry.z_L
        else:
            final_z_H = getattr(new_carry, 'z_H', None)
            final_z_L = getattr(new_carry, 'z_L', None)
            
        q_halt = outputs.get('q_halt_logits')
        q_continue = outputs.get('q_continue_logits')
        halted = getattr(new_carry, 'halted', None)
        
        self._step_counter += 1
        
        # Record final state after full execution
        self.recorder.record_snapshot(
            step=self._step_counter,
            h_cycle=h_cycles,
            l_cycle=l_cycles,
            z_H=final_z_H,
            z_L=final_z_L,
            q_halt_logits=q_halt,
            q_continue_logits=q_continue,
            is_h_update=True,
            halted=halted
        )
        
        # Store final outputs in trace
        if self.recorder.current_trace:
            self.recorder.current_trace.final_logits = outputs.get('logits')
            self.recorder.current_trace.final_halt_decisions = halted
        
        return new_carry, outputs
        
    def __getattr__(self, name):
        """Delegate attribute access to the wrapped model."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name) 