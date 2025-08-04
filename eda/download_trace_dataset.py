

def get_artifact(artifact_name: str, output_dir: str):
    """
    Download the artifact from Comet. Returns the path to the downloaded artifact.
    """
    from comet_ml import Experiment

    experiment = Experiment()
    artifact = experiment.get_artifact(artifact_name)
    artifact.download(output_dir)
    return artifact.path


if __name__ == "__main__":
    get_artifact("sudoku-complete-traces", "data/trace_dataset")