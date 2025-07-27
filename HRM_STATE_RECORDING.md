# HRM State Recording System 🎬

A comprehensive system for capturing and analyzing the internal reasoning dynamics of Hierarchical Reasoning Models (HRM) during inference.

## 🎯 Overview

This system allows you to record and analyze:
- **L-module hidden states** - Low-level detailed computations
- **H-module hidden states** - High-level abstract planning  
- **Q-head decisions** - Halt/continue reasoning choices
- **Execution metadata** - Cycles, steps, halting patterns

## 🚀 Key Features

✅ **Memory Efficient** - Uses evaluation-only loading (67% memory reduction)  
✅ **Non-intrusive** - Transparent wrapper around existing models  
✅ **Universal** - Works with any HRM architecture  
✅ **Rich Analysis** - Multiple visualization and analysis tools  
✅ **Configurable** - Adjustable recording granularity  

## 📦 Components

### Core Modules
- `hrm_state_recorder.py` - Main recording infrastructure
- `eval_utils.py` - Efficient evaluation-only model loading
- `test_state_recorder.py` - Demonstration and testing script
- `analyze_hrm_traces.py` - Analysis and visualization tools

### Data Structures
- `HRMStateSnapshot` - Single timestep state capture
- `HRMExecutionTrace` - Complete reasoning trajectory
- `HRMRecordingWrapper` - Model instrumentation wrapper

## 🛠 Quick Start

### 1. Download Models
```bash
python download_models.py
```

### 2. Record States
```bash
# Basic test with Sudoku model
python test_state_recorder.py --samples 3 --max-steps 10

# Use specific checkpoint
python test_state_recorder.py --checkpoint ./checkpoints/arc-agi-2/step_12345
```

### 3. Analyze Results
```bash
# Comprehensive analysis with plots
python analyze_hrm_traces.py --traces hrm_trace_test.json

# Text-only analysis
python analyze_hrm_traces.py --no-plots
```

## 📊 What Gets Captured

### Per Timestep
```python
HRMStateSnapshot(
    step=1,
    h_cycle=2,
    l_cycle=2,
    z_H=tensor([...]),        # H-module state
    z_L=tensor([...]),        # L-module state  
    q_halt_logits=tensor([...]),    # Halt decision
    q_continue_logits=tensor([...]), # Continue decision
    is_h_update=True,         # Whether H-module updated
    halted=tensor([False, True])     # Per-sequence halt status
)
```

### Complete Trace
```python
HRMExecutionTrace(
    snapshots=[...],          # All timestep snapshots
    batch_size=2,
    seq_len=200,
    h_cycles=2,
    l_cycles=2,
    halt_max_steps=16,
    final_logits=tensor([...]),
    final_halt_decisions=tensor([...])
)
```

## 🔍 Analysis Features

### State Dynamics
- H vs L module activation patterns
- State magnitude evolution over time
- Convergence and stability analysis
- Cross-timestep correlations

### Halting Behavior  
- Q-head decision trajectories
- Halt probability evolution
- Early vs late stopping patterns
- Exploration vs exploitation balance

### Visualization
- State evolution plots
- Activation heatmaps
- Decision boundary analysis
- Module comparison charts

## 💾 Memory Efficiency

### Old Approach (Training Mode)
```
Model: ~270MB
Optimizers: ~540MB  
Total: ~810MB ❌
```

### New Approach (Evaluation Mode)
```
Model: ~270MB
Optimizers: 0MB
Total: ~270MB ✅ (67% reduction!)
```

## 🔧 Advanced Usage

### Custom Recording
```python
from hrm_state_recorder import HRMStateRecorder, HRMRecordingWrapper
from eval_utils import load_eval_model

# Load model efficiently
model, config, metadata = load_eval_model("./checkpoints/sudoku/step_12345")

# Set up recording
recorder = HRMStateRecorder(device="cuda")
wrapped_model = HRMRecordingWrapper(model, recorder)

# Run inference with recording
carry = model.initial_carry(batch)
for step in range(max_steps):
    carry, outputs = wrapped_model(carry, batch)
    if carry.halted.all():
        break

# Analyze results
trace = recorder.get_latest_trace()
h_states = trace.get_h_states()
q_decisions = trace.get_q_decisions()
```

### Batch Processing
```python
# Record multiple examples
recorder = HRMStateRecorder()
for batch in dataloader:
    wrapped_model = HRMRecordingWrapper(model, recorder)
    # ... run inference ...
    
# Analyze all traces together
all_traces = recorder.traces
for i, trace in enumerate(all_traces):
    print(f"Trace {i}: {trace.get_execution_summary()}")
```

### Custom Analysis
```python
# Extract specific patterns
trace = recorder.get_latest_trace()

# Find H-module update points
h_updates = [snap.step for snap in trace.snapshots if snap.is_h_update]

# Analyze halt decision confidence
halt_confidences = []
for snap in trace.snapshots:
    if snap.q_halt_logits is not None:
        confidence = torch.sigmoid(snap.q_halt_logits - snap.q_continue_logits)
        halt_confidences.append(confidence)

# State magnitude tracking
state_magnitudes = [snap.z_H.norm() for snap in trace.snapshots]
```

## 📁 Output Files

### JSON Trace Files
```json
{
  "num_traces": 1,
  "traces": [{
    "trace_id": 0,
    "summary": {
      "total_steps": 10,
      "batch_size": 2,
      "h_cycles": 2,
      "l_cycles": 2,
      "h_updates": 5,
      "l_updates": 10
    },
    "snapshots": [...]
  }]
}
```

### Analysis Plots
- `hrm_trace_0_evolution.png` - State evolution over time
- `hrm_trace_0_decisions.png` - Q-head decision patterns
- `hrm_trace_0_comparison.png` - H vs L module analysis

## 🧠 Understanding HRM Reasoning

### Hierarchical Convergence
The system captures HRM's unique "hierarchical convergence" pattern:
1. **L-module** converges quickly within each cycle
2. **H-module** updates slowly, providing new context
3. **L-module** resets and converges to new equilibrium
4. Process repeats for deep reasoning

### Adaptive Computation
Track how the model dynamically adjusts reasoning time:
- **Easy problems**: Early halting, fewer steps
- **Hard problems**: Extended reasoning, more cycles
- **Q-learning**: Balance exploration vs exploitation

### Brain-Inspired Patterns
Observe patterns similar to biological cognition:
- **Multi-timescale processing**: Fast vs slow thinking
- **Hierarchical organization**: Abstract vs detailed computation
- **Dynamic resource allocation**: Adaptive reasoning depth

## 🔬 Research Applications

### Model Analysis
- Understand reasoning strategies
- Debug convergence issues
- Optimize architecture design
- Compare reasoning patterns across tasks

### Interpretability
- Visualize internal decision processes
- Track reasoning trajectory evolution  
- Identify critical reasoning steps
- Analyze failure modes

### Performance Optimization
- Find optimal halting strategies
- Reduce unnecessary computation
- Improve convergence stability
- Enhance reasoning efficiency

## 📚 Example Outputs

### Console Analysis
```
🔍 Analyzing State Dynamics
========================================

📊 Trace 0:
   Total steps: 10
   Batch size: 2
   H-cycles: 2
   L-cycles: 2
   H-updates: 5
   L-updates: 10
   H-module update steps: [2, 4, 6, 8, 10]
   H-state mean range: 0.1247 - 0.3891
   L-state mean range: 0.0856 - 0.2134

🛑 Analyzing Halting Behavior
========================================

📊 Trace 0:
   Q-decisions recorded: 5
   Mean halt probability: 0.3245
   Mean continue probability: 0.6755
   Halt probability trend: 0.1234 → 0.8901
```

### State Evolution Visualization
The analysis generates rich visualizations showing:
- State activation patterns over time
- H-module update timing
- Halt probability evolution
- Module comparison dynamics

## 🎉 Get Started

1. **Download models**: `python download_models.py`
2. **Test recording**: `python test_state_recorder.py`
3. **Analyze results**: `python analyze_hrm_traces.py`
4. **Explore patterns**: Open generated plots and JSON files

The system is ready to help you understand the fascinating internal dynamics of hierarchical reasoning! 🧠✨ 