"""Evaluation Service for Unbiased Model Performance Metrics."""
import logging
from typing import Dict, Any, List, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from schemas.result_schema import EvaluationMetrics

logger = logging.getLogger(__name__)

class EvaluationService:
    """Calculates rigorous, unbiased test metrics and evaluation diagnostics."""

    def evaluate(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        task_type: str = "classification",
        class_labels: Optional[List[str]] = None,
        feature_names: Optional[List[str]] = None,
        duration_sec: float = 0.0
    ) -> EvaluationMetrics:
        """Evaluates model against test split only."""
        is_regression = "regression" in task_type.lower()
        y_pred = model.predict(X_test)

        # Extract feature importances if available
        feature_importances = None
        if feature_names and hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            feature_importances = {
                name: float(imp) for name, imp in zip(feature_names, importances)
            }
        elif feature_names and hasattr(model, "coef_"):
            coef = np.abs(model.coef_)
            if coef.ndim > 1:
                coef = np.mean(coef, axis=0)
            feature_importances = {
                name: float(val) for name, val in zip(feature_names, coef)
            }

        if is_regression:
            mse = float(mean_squared_error(y_test, y_pred))
            rmse = float(np.sqrt(mse))
            mae = float(mean_absolute_error(y_test, y_pred))
            r2 = float(r2_score(y_test, y_pred))

            # Store actual vs predicted sample points (up to 100 points for visualization)
            n_samples = min(100, len(y_test))
            actual_vs_pred = [
                {"actual": float(y_test[i]), "predicted": float(y_pred[i])}
                for i in range(n_samples)
            ]

            metrics_dict = {
                "r2_score": round(r2, 4),
                "rmse": round(rmse, 4),
                "mae": round(mae, 4),
                "mse": round(mse, 4)
            }

            return EvaluationMetrics(
                task_type="regression",
                primary_metric_name="R² Score",
                primary_metric_value=round(r2, 4),
                metrics=metrics_dict,
                actual_vs_predicted=actual_vs_pred,
                feature_importances=feature_importances,
                execution_duration_sec=round(duration_sec, 3)
            )
        else:
            # Classification
            acc = float(accuracy_score(y_test, y_pred))
            prec = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
            rec = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
            f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

            cm = confusion_matrix(y_test, y_pred).tolist()
            unique_classes = np.unique(np.concatenate([y_test, y_pred]))
            labels = class_labels or [str(c) for c in unique_classes]

            # Compute per-class precision and recall
            per_class = {}
            for idx, c in enumerate(unique_classes):
                c_label = labels[idx] if idx < len(labels) else str(c)
                bin_true = (y_test == c).astype(int)
                bin_pred = (y_pred == c).astype(int)
                per_class[c_label] = {
                    "precision": round(float(precision_score(bin_true, bin_pred, zero_division=0)), 4),
                    "recall": round(float(recall_score(bin_true, bin_pred, zero_division=0)), 4),
                    "f1": round(float(f1_score(bin_true, bin_pred, zero_division=0)), 4)
                }

            metrics_dict = {
                "accuracy": round(acc, 4),
                "f1_score": round(f1, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4)
            }

            # Optional ROC-AUC for binary or multiclass if predict_proba is supported
            if hasattr(model, "predict_proba"):
                try:
                    y_prob = model.predict_proba(X_test)
                    if len(unique_classes) == 2:
                        roc_auc = float(roc_auc_score(y_test, y_prob[:, 1]))
                        metrics_dict["roc_auc"] = round(roc_auc, 4)
                    else:
                        roc_auc = float(roc_auc_score(y_test, y_prob, multi_class="ovr"))
                        metrics_dict["roc_auc_ovr"] = round(roc_auc, 4)
                except Exception:
                    pass

            return EvaluationMetrics(
                task_type="classification",
                primary_metric_name="Accuracy",
                primary_metric_value=round(acc, 4),
                metrics=metrics_dict,
                per_class_metrics=per_class,
                confusion_matrix=cm,
                class_labels=labels,
                feature_importances=feature_importances,
                execution_duration_sec=round(duration_sec, 3)
            )
