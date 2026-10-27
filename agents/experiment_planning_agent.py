"""Experiment Planning Agent."""
from typing import Dict, Any, List
from agents.base_agent import BaseAgent
from schemas.experiment_schema import ExperimentPlan

class ExperimentPlanningAgent(BaseAgent):
    """Formulates a comprehensive, structured experimental strategy and execution plan."""

    def __init__(self):
        super().__init__(
            name="Experiment Planning Agent",
            description="Designs the scientific methodology, milestones, and resource boundaries."
        )

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        understanding = context.get("problem_understanding", {})
        problem_type = understanding.get("problem_type", "tabular_classification")
        is_dl = understanding.get("is_deep_learning", False)

        objectives = [
            f"Establish a verified baseline benchmark for {problem_type.replace('_', ' ')}.",
            "Verify data cleanliness and eliminate potential data leakage vectors.",
            "Train candidate architectures with reproducible random seed controls.",
            "Compute unbiased out-of-sample evaluation metrics on held-out test splits."
        ]

        search_strategy = [
            "Query UCI Machine Learning Repository and OpenML for verified benchmark datasets.",
            "Search Hugging Face Hub datasets for relevant domain tasks.",
            "Support local user CSV upload if domain-specific proprietary data is available."
        ]

        preprocessing_plan = [
            "Partition data into 80% train and 20% test splits using stratified sampling if classification.",
            "Fit missing value imputers strictly on the training partition.",
            "Apply standard numerical feature scaling and encode categorical variables.",
            "Drop constant zero-variance features and suspicious ID columns."
        ]

        candidate_models = []
        if "regression" in problem_type.lower():
            candidate_models = [
                {"name": "RandomForestRegressor", "type": "Ensemble Bagging", "purpose": "Robust non-linear baseline"},
                {"name": "GradientBoostingRegressor", "type": "Boosting", "purpose": "High-accuracy residual minimization"},
                {"name": "RidgeRegression", "type": "Regularized Linear", "purpose": "Fast interpretable linear benchmark"}
            ]
        else:
            candidate_models = [
                {"name": "RandomForestClassifier", "type": "Ensemble Bagging", "purpose": "Strong robust baseline"},
                {"name": "GradientBoostingClassifier", "type": "Boosting", "purpose": "High-precision discrimination"},
                {"name": "LogisticRegression", "type": "Linear / Probabilistic", "purpose": "Fast calibrated linear baseline"}
            ]
            if is_dl:
                candidate_models.append({
                    "name": "MLPClassifier",
                    "type": "Neural Network",
                    "purpose": "Non-linear multi-layer perceptron"
                })

        evaluation_methodology = (
            "80/20 train/test split. Evaluation strictly conducted on the test split. "
            "Metrics include Accuracy, F1-Score, Precision, Recall, and Confusion Matrix for classification; "
            "R², RMSE, and MAE for regression."
        )

        resource_considerations = (
            "Standard CPU execution suitable for tabular benchmarks (< 50,000 samples). "
            "For large vision/audio datasets or deep neural networks, export to Google Colab with GPU acceleration."
        )

        deliverables = [
            "Trained model checkpoint artifact (.joblib)",
            "Unbiased empirical evaluation metrics",
            "Interactive Plotly visualization dashboard (Confusion Matrix / Residuals / Feature Importances)",
            "Exportable, runnable Google Colab Jupyter Notebook (.ipynb)",
            "Formal Laboratory Experiment Report (Markdown, HTML, JSON)"
        ]

        challenges = [
            "Potential class imbalance in classification datasets.",
            "Handling missing values and high-cardinality categorical features.",
            "Avoiding target leakage through ID or proxy columns."
        ]

        plan = ExperimentPlan(
            objectives=objectives,
            dataset_requirements={"format": understanding.get("likely_data_format", "csv"), "task": problem_type},
            search_strategy=search_strategy,
            preprocessing_plan=preprocessing_plan,
            candidate_models=candidate_models,
            evaluation_methodology=evaluation_methodology,
            resource_considerations=resource_considerations,
            expected_deliverables=deliverables,
            potential_challenges=challenges
        )

        return {"experiment_plan": getattr(plan, "model_dump", plan.dict)()}
