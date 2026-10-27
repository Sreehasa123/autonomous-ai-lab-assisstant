"""Improvement Agent."""
from typing import Dict, Any, List
from agents.base_agent import BaseAgent
from schemas.result_schema import ImprovementRecommendation

class ImprovementAgent(BaseAgent):
    """Generates concrete next steps and config revisions for subsequent experiment versions."""

    def __init__(self):
        super().__init__(
            name="Improvement Agent",
            description="Recommends specific hyperparameter updates, preprocessing refinements, or alternative models."
        )

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        error_report = context.get("error_analysis", {})
        recommendations = error_report.get("recommendations", [])
        
        # If no specific recommendations, supply standard fine-tuning action
        if not recommendations:
            recommendations.append(ImprovementRecommendation(
                category="hyperparameter",
                title="Hyperparameter Exploration",
                action="Increase n_estimators to 200 and search max_depth in [5, 10, 15].",
                rationale="Reduces variance of bagging estimators.",
                estimated_impact="moderate"
            ).dict())

        return {"actionable_improvements": recommendations}
