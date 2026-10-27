"""Evaluation Agent."""
from typing import Dict, Any
from agents.base_agent import BaseAgent
from services.evaluation_service import EvaluationService

class EvaluationAgent(BaseAgent):
    """Calculates, verifies, and packages unbiased test performance metrics."""

    def __init__(self):
        super().__init__(
            name="Evaluation Agent",
            description="Computes rigorous performance metrics strictly on held-out test splits."
        )
        self.evaluation_service = EvaluationService()

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        model = context.get("model")
        X_test = context.get("X_test")
        y_test = context.get("y_test")
        task_type = context.get("task_type", "classification")

        if model is None or X_test is None or y_test is None:
            raise ValueError("Model and test evaluation partitions are required.")

        metrics = self.evaluation_service.evaluate(
            model=model,
            X_test=X_test,
            y_test=y_test,
            task_type=task_type,
            class_labels=context.get("class_labels"),
            feature_names=context.get("feature_names"),
            duration_sec=context.get("duration_sec", 0.0)
        )
        return {"evaluation_metrics": metrics.dict()}
