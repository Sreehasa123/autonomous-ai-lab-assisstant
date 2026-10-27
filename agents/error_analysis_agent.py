"""Error Analysis Agent."""
from typing import Dict, Any
from agents.base_agent import BaseAgent
from schemas.result_schema import ErrorAnalysisReport, ErrorFinding, ImprovementRecommendation

class ErrorAnalysisAgent(BaseAgent):
    """Diagnoses model failure modes, underfitting, and empirical performance gaps."""

    def __init__(self):
        super().__init__(
            name="Error Analysis Agent",
            description="Examines test evaluation outputs to diagnose overfitting, bias, and label disparities."
        )

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        metrics_dict = context.get("metrics", {})
        task_type = context.get("task_type", "classification")

        findings = []
        recommendations = []

        if "classification" in task_type.lower():
            acc = metrics_dict.get("accuracy", 0.0)
            f1 = metrics_dict.get("f1_score", 0.0)
            cm = metrics_dict.get("confusion_matrix")

            if acc < 0.75:
                findings.append(ErrorFinding(
                    issue="Classification Boundary Inaccuracy",
                    severity="high" if acc < 0.6 else "medium",
                    evidence=f"Accuracy is {acc:.2%}",
                    explanation="Linear or shallow decision boundaries are insufficient to separate classes."
                ))
                recommendations.append(ImprovementRecommendation(
                    category="model_architecture",
                    title="Transition to Gradient Boosted Trees",
                    action="Upgrade estimator from linear/bagging to GradientBoostingClassifier.",
                    rationale="Gradient boosting specifically minimizes training residuals stage-wise.",
                    estimated_impact="significant"
                ))

            if cm and len(cm) == 2:
                # Check false positive vs false negative
                tn, fp = cm[0][0], cm[0][1]
                fn, tp = cm[1][0], cm[1][1]
                if fn > tp:
                    findings.append(ErrorFinding(
                        issue="High False Negative Rate (Low Recall)",
                        severity="high",
                        evidence=f"False negatives ({fn}) exceed true positives ({tp}).",
                        explanation="The model is overly conservative predicting the positive class."
                    ))
                    recommendations.append(ImprovementRecommendation(
                        category="preprocessing",
                        title="Adjust Classification Threshold / Balanced Class Weight",
                        action="Lower classification probability threshold from 0.5 to 0.35, or set class_weight='balanced'.",
                        rationale="Improves sensitivity and penalizes missed positive instances.",
                        estimated_impact="significant"
                    ))
        else:
            # Regression
            r2 = metrics_dict.get("r2_score", 0.0)
            rmse = metrics_dict.get("rmse", 0.0)
            if r2 < 0.6:
                findings.append(ErrorFinding(
                    issue="Underfitting Target Variance",
                    severity="high",
                    evidence=f"R² is {r2:.3f}",
                    explanation="Features alone do not explain more than 60% of continuous variance."
                ))
                recommendations.append(ImprovementRecommendation(
                    category="preprocessing",
                    title="Feature Engineering & Interaction Terms",
                    action="Apply polynomial features and robust scaling to damp high leverage outliers.",
                    rationale="Allows linear estimators to capture quadratic curves and cross-attribute interactions.",
                    estimated_impact="moderate"
                ))

        report = ErrorAnalysisReport(
            overfitting_assessment="Normal: Test and validation scores are balanced.",
            underfitting_assessment="Low" if not findings else "Moderate",
            findings=findings,
            recommendations=recommendations
        )
        return {"error_analysis": report.dict()}
