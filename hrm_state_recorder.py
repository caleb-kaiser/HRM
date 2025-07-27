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

    def get_latest_trace(self) -> Optional[HRMExecutionTrace]:
        """Get the most recent execution trace."""
        return self.traces[-1] if self.traces else None


class HRMRecordingWrapper(nn.Module):
    """Wrapper that instruments an HRM model for state recording."""
    
    def __init__(self, model: nn.Module, recorder: HRMStateRecorder):
        super().__init__()
        self.model = model
        self.recorder = recorder
        self._step_counter = 0
        
    def forward(self, carry, batch, return_keys=None):
        """Forward pass with state recording."""
        # Start recording if not already started
        if self.recorder.current_trace is None:
            self.recorder.start_recording(self.model)
        
        # Reset step counter for new sequences
        if hasattr(carry, 'halted') and carry.halted.all():
            self._step_counter = 0
        
        # Get model config for cycles
        h_cycles = getattr(self.model.config, 'H_cycles', 2)
        l_cycles = getattr(self.model.config, 'L_cycles', 2)
        
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
        new_carry, outputs = self.model(carry, batch, return_keys)
        
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