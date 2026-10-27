"""Model Selection Agent."""
from typing import Dict, Any, List
from agents.base_agent import BaseAgent
from services.model_service import ModelService
from schemas.experiment_schema import ModelConfig

class ModelSelectionAgent(BaseAgent):
    """Selects and recommends optimal candidate models for the task."""

    def __init__(self):
        super().__init__(
            name="Model Selection Agent",
            description="Recommends verified machine learning and deep learning architectures with clear rationales."
        )
        self.model_service = ModelService()

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        task_type = context.get("problem_type", "tabular_classification")
        analysis = context.get("analysis_result", {})
        shape = analysis.get("shape", [500, 10])
        num_rows, num_cols = shape[0], shape[1]
        is_dl = context.get("is_deep_learning", False)

        models = self.model_service.recommend_models(
            task_type=task_type,
            num_rows=num_rows,
            num_features=num_cols,
            is_deep_learning=is_dl
        )

        selected_model = models[0] if models else ModelConfig()
        return {
            "recommended_models": [getattr(m, "model_dump", m.dict)() for m in models],
            "default_model": getattr(selected_model, "model_dump", selected_model.dict)()
        }
