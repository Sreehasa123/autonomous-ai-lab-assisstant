"""Notebook Service."""
import json
from pathlib import Path
from typing import Dict, Any, Optional
from integrations.colab_exporter import ColabExporter
from database.repository import ExperimentRepository

class NotebookService:
    """Service for generating and managing Jupyter notebooks for Google Colab."""

    def __init__(self, repo: Optional[ExperimentRepository] = None):
        self.exporter = ColabExporter()
        self.repo = repo or ExperimentRepository()

    def export_experiment_notebook(
        self,
        experiment_id: str,
        version_number: int = 1
    ) -> Path:
        """Generates a real .ipynb for the experiment and registers it in the repository."""
        exp = self.repo.get_experiment(experiment_id)
        if not exp:
            raise ValueError(f"Experiment {experiment_id} not found")

        dataset_meta = json.loads(exp.selected_dataset_json or "{}")
        preprocessing_config = json.loads(exp.preprocessing_config_json or "{}")
        model_config = json.loads(exp.model_config_json or "{}")

        nb_path = self.exporter.generate_notebook(
            experiment_name=exp.name,
            problem_statement=exp.problem_statement,
            dataset_meta=dataset_meta,
            preprocessing_config=preprocessing_config,
            model_config=model_config,
            experiment_id=exp.id,
            version_number=version_number
        )

        # Register artifact
        self.repo.add_artifact(
            experiment_id=exp.id,
            artifact_type="notebook",
            file_name=nb_path.name,
            file_path=str(nb_path),
            file_size_bytes=nb_path.stat().st_size
        )
        return nb_path
