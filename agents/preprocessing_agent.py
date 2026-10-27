"""Preprocessing Agent."""
from typing import Dict, Any
from agents.base_agent import BaseAgent
from schemas.experiment_schema import PreprocessingConfig

class PreprocessingAgent(BaseAgent):
    """Proposes clean, leak-free preprocessing pipelines tailored to data findings."""

    def __init__(self):
        super().__init__(
            name="Preprocessing Agent",
            description="Configures feature imputation, scaling, categorical encoding, and splitting."
        )

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        analysis = context.get("analysis_result", {})
        problem_type = context.get("problem_type", "tabular_classification")

        # Determine scaling and imputation based on data
        missing_counts = analysis.get("missing_counts", {})
        has_missing = any(cnt > 0 for cnt in missing_counts.values())

        config = PreprocessingConfig(
            handle_missing="median_mode" if has_missing else "none",
            scale_numeric="standard",
            encode_categorical="onehot",
            remove_duplicates=True,
            handle_imbalance="none",
            test_size=0.2,
            val_size=0.1,
            random_state=42
        )

        return {"preprocessing_config": getattr(config, "model_dump", config.dict)()}
