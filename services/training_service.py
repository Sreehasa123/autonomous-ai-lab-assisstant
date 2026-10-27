"""Training Execution Service."""
import io
import time
import logging
from typing import Dict, Any, Tuple, Optional, List
import numpy as np

from services.model_service import ModelService
from services.evaluation_service import EvaluationService
from services.artifact_service import ArtifactService
from schemas.experiment_schema import ModelConfig
from schemas.result_schema import EvaluationMetrics

logger = logging.getLogger(__name__)

class TrainingService:
    """Executes model training, captures diagnostics, logs, and artifacts."""

    def __init__(self):
        self.model_service = ModelService()
        self.evaluation_service = EvaluationService()
        self.artifact_service = ArtifactService()

    def train_and_evaluate(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        model_config: ModelConfig,
        task_type: str,
        experiment_id: str,
        run_id: Optional[str] = None,
        feature_names: Optional[List[str]] = None,
        class_labels: Optional[List[str]] = None
    ) -> Tuple[Any, EvaluationMetrics, str, str]:
        """
        Fits candidate estimator, performs evaluation, records logs and saves model checkpoint.
        Returns: (fitted_model, evaluation_metrics, logs, artifact_path)
        """
        logs_stream = io.StringIO()
        logs_stream.write(f"=== Starting Training Session for Experiment {experiment_id} ===\n")
        logs_stream.write(f"Architecture: {model_config.model_name} ({model_config.model_family})\n")
        logs_stream.write(f"Hyperparameters: {model_config.hyperparameters}\n")
        logs_stream.write(f"Train Shape: {X_train.shape}, Test Shape: {X_test.shape}\n")

        # Instantiate
        model = self.model_service.instantiate_model(model_config)

        # Fit
        start_time = time.time()
        logs_stream.write("Fitting estimator on training split...\n")
        model.fit(X_train, y_train)
        duration_sec = time.time() - start_time
        logs_stream.write(f"Training completed successfully in {duration_sec:.3f}s.\n")

        # Evaluate
        logs_stream.write("Running unbiased evaluation on test split...\n")
        metrics = self.evaluation_service.evaluate(
            model=model,
            X_test=X_test,
            y_test=y_test,
            task_type=task_type,
            class_labels=class_labels,
            feature_names=feature_names,
            duration_sec=duration_sec
        )
        logs_stream.write(f"Primary Metric ({metrics.primary_metric_name}): {metrics.primary_metric_value}\n")
        logs_stream.write(f"Metrics breakdown: {metrics.metrics}\n")

        # Save model checkpoint
        logs_stream.write("Persisting model checkpoint artifact...\n")
        artifact_path, checksum = self.artifact_service.save_model(
            experiment_id=experiment_id,
            model_object=model,
            model_name=model_config.model_name,
            run_id=run_id
        )
        logs_stream.write(f"Artifact saved: {artifact_path} (SHA-256: {checksum[:12]}...)\n")

        full_logs = logs_stream.getvalue()
        # Save log file artifact
        self.artifact_service.save_execution_log(experiment_id, full_logs, run_id=run_id)

        return model, metrics, full_logs, artifact_path
