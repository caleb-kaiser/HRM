# Comet ML Integration for HRM State Recording

This guide shows how to use Comet ML to store and manage your HRM execution traces as versioned dataset artifacts.

## 🚀 Quick Start

### Prerequisites

1. **Install Comet ML**:
   ```bash
   pip install comet_ml
   ```

2. **Set up your API key**:
   ```bash
   # Option 1: Environment variable
   export COMET_API_KEY="your-api-key-here"
   
   # Option 2: Interactive login
   python -c "import comet_ml; comet_ml.login()"
   ```

3. **Get your API key**:
   - Sign up at [comet.ml](https://www.comet.ml)
   - Go to Account Settings → API Keys
   - Copy your API key

### Basic Usage

```python
import comet_ml
from hrm_state_recorder import HRMStateRecorder

# Record your HRM traces (same as before)
recorder = HRMStateRecorder()
# ... run your HRM inference with recording ...

# Create Comet experiment
experiment = HRMStateRecorder.create_comet_experiment(
    project_name="hrm-experiments",
    experiment_name="my-trace-experiment"
)

# Upload traces to Comet as a versioned artifact
version = recorder.upload_traces_to_comet(
    experiment=experiment,
    artifact_name="my-hrm-traces",
    aliases=["latest", "experiment-1"],
    description="HRM traces from Sudoku model",
    include_tensors=True
)

print(f"Uploaded artifact version: {version}")
experiment.end()
```

## 📊 Complete Workflow Example

### 1. Record HRM States

```python
from hrm_state_recorder import HRMStateRecorder, HRMRecordingWrapper
from eval_utils import load_eval_model

# Load your model
model, config, metadata = load_eval_model("./checkpoints/sudoku-extreme/checkpoint")

# Set up recording
recorder = HRMStateRecorder(device="cuda")
wrapped_model = HRMRecordingWrapper(model, recorder, config.arch)

# Run inference with recording
# ... your inference code here ...

recorder.stop_recording()
print(f"Recorded {len(recorder.traces)} traces")
```

### 2. Upload to Comet ML

```python
import comet_ml

# Create experiment
experiment = comet_ml.Experiment(project_name="hrm-research")
experiment.set_name("sudoku-state-analysis")
experiment.add_tag("state-recording")

# Log experiment parameters
experiment.log_parameter("model_type", "HRM")
experiment.log_parameter("task", "sudoku")
experiment.log_parameter("num_samples", 10)

# Upload traces with full tensor data
artifact_version = recorder.upload_traces_to_comet(
    experiment=experiment,
    artifact_name="hrm-sudoku-traces",
    artifact_type="dataset",
    aliases=["latest", "sudoku-v1", "baseline"],
    description="HRM execution traces from Sudoku inference with hidden states",
    model_checkpoint="./checkpoints/sudoku-extreme/checkpoint",
    include_tensors=True,
    tensor_format="both"  # Saves both .pt and .npz formats
)

print(f"✅ Uploaded artifact version: {artifact_version}")
print(f"🌐 View at: {experiment.url}")

experiment.end()
```

### 3. Download and Analyze

```python
# Later, in a different session/script...
import comet_ml

# Connect to your experiment
experiment = comet_ml.ExistingExperiment(previous_experiment="your-experiment-key")

# Download the traces
new_recorder = HRMStateRecorder()
download_result = new_recorder.download_traces_from_comet(
    experiment=experiment,
    artifact_name="hrm-sudoku-traces",
    version_or_alias="latest",  # or specific version like "1.0.0"
    download_tensors=True
)

print(f"Downloaded to: {download_result['download_path']}")

# Load tensor data for analysis
tensor_files = download_result["files"]["tensors"]
for tensor_file in tensor_files:
    if tensor_file.endswith('.pt'):
        data = torch.load(tensor_file, map_location="cpu")
        h_states = data['h_states']  # [steps, batch, seq, hidden]
        l_states = data['l_states']  # [steps, batch, seq, hidden]
        
        # Your analysis here...
        print(f"H-states shape: {h_states.shape}")
        print(f"L-states shape: {l_states.shape}")
```

## 🔄 Artifact Versioning

Comet automatically versions your artifacts. You can manage versions with aliases:

```python
# Upload different versions with meaningful aliases
recorder.upload_traces_to_comet(
    experiment=experiment,
    artifact_name="hrm-traces",
    aliases=["baseline"],
    description="Initial baseline traces"
)

# Later, upload an improved version
recorder.upload_traces_to_comet(
    experiment=experiment,
    artifact_name="hrm-traces",  # Same name
    aliases=["optimized", "latest"],  # New aliases
    description="Traces with optimized hyperparameters"
)

# Access different versions
baseline_traces = experiment.get_artifact("hrm-traces", version_or_alias="baseline")
latest_traces = experiment.get_artifact("hrm-traces", version_or_alias="latest")
specific_version = experiment.get_artifact("hrm-traces", version="1.0.0")
```

## 🧪 Testing the Integration

Run the complete test script:

```bash
# Basic test
python test_comet_integration.py

# Custom parameters
python test_comet_integration.py \
    --checkpoint ./checkpoints/sudoku-extreme/checkpoint \
    --project my-hrm-project \
    --experiment sudoku-analysis \
    --samples 5 \
    --max-steps 10

# Demo artifact versioning
python test_comet_integration.py --demo-versioning
```

## 📁 What Gets Uploaded

### Metadata File (`traces_metadata.json`)
- Execution summaries for all traces
- Model configuration
- Timing information
- Step counts and cycles

### Tensor Files (Optional)
- **PyTorch format** (`.pt`): Native PyTorch tensors
- **NumPy format** (`.npz`): Compressed NumPy arrays
- Contains full hidden states:
  - `h_states`: H-module states `[steps, batch, seq, hidden]`
  - `l_states`: L-module states `[steps, batch, seq, hidden]`
  - `q_halt_logits`: Q-head halt decisions `[steps, batch]`
  - `q_continue_logits`: Q-head continue decisions `[steps, batch]`

### Artifact Metadata
- Creation timestamp
- Model checkpoint path
- Number of traces
- Tensor format information
- Custom description and tags

## 🔍 Advanced Usage

### Custom Experiment Setup

```python
# Advanced experiment creation
experiment = HRMStateRecorder.create_comet_experiment(
    project_name="hrm-research",
    experiment_name="advanced-sudoku-analysis",
    tags=["hrm", "sudoku", "state-analysis", "production"],
    workspace="my-team-workspace"
)

# Log additional metadata
experiment.log_parameter("batch_size", 16)
experiment.log_parameter("learning_rate", 1e-4)
experiment.log_parameter("optimizer", "AdamW")
experiment.log_metric("final_accuracy", 0.95)
```

### Selective Tensor Upload

```python
# Upload only metadata (lightweight)
recorder.upload_traces_to_comet(
    experiment=experiment,
    artifact_name="hrm-metadata-only",
    include_tensors=False,  # No tensor data
    description="Lightweight metadata for quick analysis"
)

# Upload with specific tensor format
recorder.upload_traces_to_comet(
    experiment=experiment,
    artifact_name="hrm-numpy-traces",
    tensor_format="npz",  # Only NumPy format
    description="Traces in NumPy format for non-PyTorch analysis"
)
```

### Collaborative Workflows

```python
# Team member A uploads traces
experiment_a = comet_ml.Experiment(project_name="team-hrm-project")
recorder_a.upload_traces_to_comet(
    experiment=experiment_a,
    artifact_name="shared-hrm-traces",
    aliases=["team-baseline"],
    description="Baseline traces for team analysis"
)

# Team member B downloads and analyzes
experiment_b = comet_ml.Experiment(project_name="team-hrm-project")  
recorder_b = HRMStateRecorder()
traces = recorder_b.download_traces_from_comet(
    experiment=experiment_b,
    artifact_name="shared-hrm-traces",
    version_or_alias="team-baseline"
)
```

## 🎯 Integration with Analysis Pipeline

Combine with the existing analysis tools:

```python
# 1. Record and upload traces
recorder = HRMStateRecorder()
# ... record traces ...
recorder.upload_traces_to_comet(experiment, "my-traces")

# 2. Download for analysis
downloaded = recorder.download_traces_from_comet(experiment, "my-traces")

# 3. Analyze with existing tools
from analyze_embeddings import load_and_analyze_embeddings

# Load tensor data
tensor_file = downloaded["files"]["tensors"][0]  # First .pt file
data = load_and_analyze_embeddings(tensor_file.replace('.pt', ''), trace_id=0)

# 4. Log analysis results back to Comet
experiment.log_metric("h_state_dimensionality", data["h_participation_ratio"])
experiment.log_metric("l_state_dimensionality", data["l_participation_ratio"])
```

## 📈 Benefits

### Data Lineage
- **Track which model produced which traces**
- **Link traces to specific experiments**
- **Version your datasets alongside your models**

### Collaboration
- **Share traces across team members**
- **Centralized trace storage**
- **Access control via Comet workspaces**

### Reproducibility
- **Exact trace reproduction with versioning**
- **Complete metadata tracking**
- **Easy experiment comparison**

### Scalability
- **Cloud storage for large trace datasets**
- **Efficient compressed formats**
- **Remote access from anywhere**

## 🔧 Troubleshooting

### Common Issues

1. **API Key Not Found**:
   ```bash
   export COMET_API_KEY="your-key"
   # or
   python -c "import comet_ml; comet_ml.login()"
   ```

2. **Large Upload Times**:
   ```python
   # Upload metadata only for quick iterations
   recorder.upload_traces_to_comet(experiment, "traces", include_tensors=False)
   ```

3. **Storage Limits**:
   ```python
   # Use compressed NumPy format
   recorder.upload_traces_to_comet(experiment, "traces", tensor_format="npz")
   ```

4. **Version Conflicts**:
   ```python
   # Use specific versions instead of aliases
   traces = experiment.get_artifact("traces", version="1.2.3")
   ```

5. **Artifact Not Found After Upload**:
   ```python
   # Sometimes there's a delay in artifact processing. Wait and retry:
   import time
   time.sleep(10)
   
   # Or use the experiment URL to check if artifact appears in UI
   print(f"Check artifacts at: {experiment.url}")
   
   # List all artifacts to debug
   api = comet_ml.API()
   artifacts = api.get_artifacts(workspace="your-workspace", project="your-project")
   print(f"Available artifacts: {[a.name for a in artifacts]}")
   ```

6. **Manual Download Workaround**:
   ```python
   # If automatic download fails, use the Comet UI or API directly
   import comet_ml
   
   api = comet_ml.API()
   experiment = api.get_experiment("your-workspace", "your-project", "experiment-id")
   
   # List artifacts
   artifacts = experiment.get_artifacts()
   for artifact in artifacts:
       print(f"Artifact: {artifact.name}, Version: {artifact.version}")
   ```

## 🌟 Best Practices

1. **Use meaningful artifact names**: `hrm-sudoku-traces` vs `traces`
2. **Add descriptive aliases**: `["baseline", "optimized", "production"]`
3. **Include model checkpoint paths** in metadata
4. **Tag experiments** for easy filtering
5. **Use workspace organization** for different projects
6. **Start with metadata-only uploads** for quick iterations
7. **Document your trace collection process** in experiment descriptions

## 🔗 Next Steps

- **Explore the Comet UI** to visualize your artifacts
- **Set up automated trace collection** in your training pipelines  
- **Create analysis notebooks** that load traces from Comet
- **Build dashboards** to compare traces across experiments
- **Integrate with CI/CD** for automated trace validation

---

**Happy tracing! 🚀**

For more details, see the [Comet ML documentation](https://www.comet.ml/docs/) and the `test_comet_integration.py` example script. 