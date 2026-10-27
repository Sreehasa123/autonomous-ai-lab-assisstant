"""Artifact Management Service."""
import hashlib
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import joblib

from config.settings import settings
from database.repository import ExperimentRepository

logger = logging.getLogger(__name__)

class ArtifactService:
    """Manages files, checkpoints, plots, logs, and reports stored in artifact storage."""

    def __init__(self, repo: Optional[ExperimentRepository] = None):
        self.repo = repo or ExperimentRepository()
        settings.ensure_directories()

    @staticmethod
    def calculate_checksum(file_path: Path) -> str:
        """Calculate SHA-256 checksum for artifact integrity."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def save_model(
        self,
        experiment_id: str,
        model_object: Any,
        model_name: str,
        run_id: Optional[str] = None
    ) -> Tuple[str, str]:
        """Save a trained model using joblib and register in database."""
        safe_name = f"{model_name.replace(' ', '_')}_{experiment_id[:8]}.joblib"
        target_path = settings.models_dir / safe_name
        joblib.dump(model_object, target_path)
        
        file_size = target_path.stat().st_size
        checksum = self.calculate_checksum(target_path)
        
        # Register in database
        self.repo.add_artifact(
            experiment_id=experiment_id,
            run_id=run_id,
            artifact_type="model",
            file_name=safe_name,
            file_path=str(target_path),
            file_size_bytes=file_size
        )
        return str(target_path), checksum

    def load_model(self, file_path: str) -> Optional[Any]:
        """Safely load a trained model checkpoint."""
        p = Path(file_path)
        if not p.exists():
            logger.warning(f"Model file does not exist: {file_path}")
            return None
        try:
            return joblib.load(p)
        except Exception as e:
            logger.error(f"Error loading model from {file_path}: {e}")
            return None

    def save_plot_data(
        self,
        experiment_id: str,
        plot_name: str,
        plot_json: str,
        run_id: Optional[str] = None
    ) -> str:
        """Save interactive Plotly chart JSON."""
        safe_name = f"{plot_name}_{experiment_id[:8]}.json"
        target_path = settings.plots_dir / safe_name
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(plot_json)

        self.repo.add_artifact(
            experiment_id=experiment_id,
            run_id=run_id,
            artifact_type="plot",
            file_name=safe_name,
            file_path=str(target_path),
            file_size_bytes=target_path.stat().st_size
        )
        return str(target_path)

    def save_execution_log(
        self,
        experiment_id: str,
        logs: str,
        run_id: Optional[str] = None
    ) -> str:
        """Save text execution logs to disk."""
        safe_name = f"run_log_{experiment_id[:8]}_{run_id or 'main'}.txt"
        target_path = settings.logs_dir / safe_name
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(logs)

        self.repo.add_artifact(
            experiment_id=experiment_id,
            run_id=run_id,
            artifact_type="log",
            file_name=safe_name,
            file_path=str(target_path),
            file_size_bytes=target_path.stat().st_size
        )
        return str(target_path)
