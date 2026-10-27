"""Notebook Generation and Colab Export Tests."""
import json
from pathlib import Path
import nbformat as nbf
from integrations.colab_exporter import ColabExporter
from services.notebook_service import NotebookService
from database.repository import ExperimentRepository

def test_colab_exporter_creates_valid_ipynb():
    exporter = ColabExporter()
    dataset_meta = {
        "name": "UCI: Iris Flower Dataset",
        "task_type": "classification",
        "target_column": "target"
    }
    prep_config = {"test_size": 0.2, "random_state": 42, "scale_numeric": "standard"}
    model_config = {"model_name": "RandomForestClassifier", "framework": "scikit-learn"}

    nb_path = exporter.generate_notebook(
        experiment_name="Iris Classification Benchmark",
        problem_statement="Classify iris species",
        dataset_meta=dataset_meta,
        preprocessing_config=prep_config,
        model_config=model_config,
        experiment_id="test_exp_colab_123",
        version_number=1
    )

    assert nb_path.exists()
    assert nb_path.suffix == ".ipynb"

    # Verify structural validity with nbformat
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbf.read(f, as_version=4)

    assert len(nb.cells) > 5
    # Check GPU detection cell exists
    cell_sources = [c.source for c in nb.cells]
    assert any("torch.cuda.is_available()" in src for src in cell_sources)
    # Check export results payload exists
    assert any("experiment_results.json" in src for src in cell_sources)

    # Clean up generated test notebook file
    if nb_path.exists():
        nb_path.unlink()
