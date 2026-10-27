"""Dataset Analysis Agent."""
from typing import Dict, Any
from agents.base_agent import BaseAgent
from services.dataset_service import DatasetService

class DatasetAnalysisAgent(BaseAgent):
    """Inspects dataset structure, dimensions, dtypes, and statistical summaries."""

    def __init__(self):
        super().__init__(
            name="Dataset Analysis Agent",
            description="Performs deep statistical profiling of the selected dataset."
        )
        self.dataset_service = DatasetService()

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        dataset_meta = context.get("selected_dataset", {})
        if not dataset_meta:
            raise ValueError("No dataset provided for analysis.")

        df, target_col = self.dataset_service.load_dataset(dataset_meta)
        analysis_result = self.dataset_service.inspect_and_analyze(df, target_column=target_col)
        return {
            "analysis_result": getattr(analysis_result, "model_dump", analysis_result.dict)(),
            "target_column": target_col,
            "row_count": len(df),
            "column_count": len(df.columns)
        }
