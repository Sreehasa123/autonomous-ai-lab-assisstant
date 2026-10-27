"""Dataset Comparison Agent."""
from typing import Dict, Any, List
from agents.base_agent import BaseAgent

class DatasetComparisonAgent(BaseAgent):
    """Evaluates and ranks candidate datasets based on objective suitability criteria."""

    def __init__(self):
        super().__init__(
            name="Dataset Comparison Agent",
            description="Computes transparent suitability scores and highlights comparative strengths and limitations."
        )

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        datasets = context.get("datasets", [])
        problem_type = context.get("problem_type", "tabular_classification")

        scored_datasets = []
        for ds in datasets:
            score, reason = self._compute_suitability(ds, problem_type)
            ds_copy = dict(ds)
            ds_copy["suitability_score"] = score
            ds_copy["suitability_reason"] = reason
            scored_datasets.append(ds_copy)

        # Sort descending by suitability score
        scored_datasets.sort(key=lambda x: x.get("suitability_score", 0), reverse=True)
        return {"ranked_datasets": scored_datasets}

    def _compute_suitability(self, ds: Dict[str, Any], target_task: str) -> (float, str):
        """Calculates 0-100 score based on task compatibility, size, features, and accessibility."""
        score = 50.0  # Base
        reasons = []

        ds_task = ds.get("task_type", "").lower()
        if target_task.lower() in ds_task or ds_task in target_task.lower():
            score += 25.0
            reasons.append("Exact task type match")
        elif "classification" in target_task and "classification" in ds_task:
            score += 20.0
            reasons.append("Compatible classification task")
        elif "regression" in target_task and "regression" in ds_task:
            score += 20.0
            reasons.append("Compatible continuous regression task")

        # Provider verification
        if ds.get("source_provider") in ("uci", "openml"):
            score += 15.0
            reasons.append("Verified standard scientific benchmark repository")
        elif ds.get("source_provider") == "huggingface":
            score += 10.0
            reasons.append("Accessible Hugging Face community repository")

        # Row counts
        rows = ds.get("num_rows")
        if rows:
            if 100 <= rows <= 50000:
                score += 10.0
                reasons.append(f"Ideal sample size for fast local training ({rows} rows)")
            elif rows > 50000:
                score += 5.0
                reasons.append("Large volume dataset; recommend Colab GPU")

        license_val = ds.get("license", "").lower()
        if any(l in license_val for l in ["mit", "cc-by", "public domain", "apache"]):
            score += 5.0
            reasons.append("Permissive open license")

        final_score = min(100.0, max(0.0, score))
        summary_reason = "; ".join(reasons) if reasons else "General candidate dataset"
        return round(final_score, 1), summary_reason
