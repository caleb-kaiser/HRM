

def get_artifact(artifact_name: str, output_dir: str):
    """
    Download the artifact from Comet. Returns the path to the downloaded artifact.
    """
    from comet_ml import Experiment

    experiment = Experiment()
    artifact = experiment.get_artifact(artifact_name)
    artifact.download(output_dir)
    return artifact


if __name__ == "__main__":
    get_artifact("sudoku-training-traces", "data/sudoku_full_trace_dataset")