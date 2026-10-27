"""Problem Understanding Agent."""
import re
from typing import Dict, Any
from agents.base_agent import BaseAgent
from schemas.experiment_schema import ProblemUnderstanding

class ProblemUnderstandingAgent(BaseAgent):
    """Analyzes natural language problem statements and maps them to technical ML tasks."""

    def __init__(self):
        super().__init__(
            name="Problem Understanding Agent",
            description="Deconstructs natural language requirements into structured ML/DL objectives."
        )

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        problem_statement = context.get("problem_statement", "").strip()
        user_preference = context.get("ml_dl_preference", "Auto")

        if not problem_statement:
            raise ValueError("Problem statement is required.")

        # Check if LLM is available
        if self.llm_service.is_configured():
            prompt = (
                f"Analyze this AI/ML problem statement and return a JSON structure:\n"
                f"Problem: {problem_statement}\n"
                f"Preference: {user_preference}\n"
                f"Format: {{'problem_type': '...', 'objective': '...', 'expected_input': '...', "
                f"'expected_output': '...', 'likely_data_format': '...', 'target_variable': '...', "
                f"'candidate_metrics': [...], 'possible_model_families': [...], 'preprocessing_requirements': [...], "
                f"'potential_risks': [...], 'is_deep_learning': bool, 'explanation': '...'}}"
            )
            success, content, source = self.llm_service.generate_completion(
                "You are an expert AI lab assistant. Respond strictly in valid JSON.", prompt
            )
            if success:
                try:
                    import json
                    parsed = json.loads(content)
                    return {"problem_understanding": ProblemUnderstanding(**parsed).dict(), "source": source}
                except Exception:
                    pass

        # Robust, comprehensive Rule-Based Understanding
        p_lower = problem_statement.lower()

        is_dl = False
        if any(term in p_lower for term in ["deep learning", "neural", "image", "vision", "audio", "cnn", "transformer", "bert", "resnet"]):
            is_dl = True
        if user_preference.lower() == "deep learning":
            is_dl = True

        # Infer Problem Type
        if any(w in p_lower for w in ["price", "cost", "salary", "temperature", "forecast", "revenue", "continuous", "housing", "regression", "value"]):
            problem_type = "tabular_regression"
            objective = "Predict continuous numerical value from input features."
            expected_input = "Tabular numeric and categorical feature vectors."
            expected_output = "Continuous real-valued prediction."
            likely_format = "csv"
            candidate_metrics = ["R² Score", "RMSE", "MAE", "MSE"]
            model_families = ["Random Forest Regressor", "Gradient Boosting Regressor", "Ridge Regression", "MLP Regressor"]
            preprocessing = ["Missing value imputation", "Standard feature scaling", "Categorical encoding"]
            risks = ["Sensitivity to extreme outliers", "Multicollinearity between features", "Target skewness"]

        elif any(w in p_lower for w in ["image", "images", "picture", "photo", "pixels", "scan", "x-ray", "mri", "tumor scan", "vision"]):
            problem_type = "image_classification"
            objective = "Classify visual images into target categories or diagnoses."
            expected_input = "2D image tensors (RGB / Grayscale)."
            expected_output = "Discrete diagnostic or categorical label."
            likely_format = "image_folder / png"
            candidate_metrics = ["Accuracy", "Weighted F1-Score", "Recall", "ROC-AUC"]
            model_families = ["Convolutional Neural Networks (CNN)", "Pretrained ResNet/MobileNet", "Vision Transformer"]
            preprocessing = ["Image resizing (e.g., 224x224)", "Pixel normalization (0-1 / ImageNet stats)", "Data augmentation (flip, rotation)"]
            risks = ["Class imbalance in rare disease categories", "High computational memory requirements", "Domain shift in lighting/angles"]

        elif any(w in p_lower for w in ["text", "sentiment", "review", "nlp", "tweet", "comment", "document", "classification"]):
            problem_type = "text_classification"
            objective = "Categorize text sequences into semantic sentiment or topic classes."
            expected_input = "Raw natural language text strings."
            expected_output = "Class probability distribution or discrete category."
            likely_format = "csv / text"
            candidate_metrics = ["Accuracy", "Precision", "Recall", "Macro F1-Score"]
            model_families = ["TF-IDF + Logistic Regression", "Random Forest Classifier", "Transformer / DistilBERT"]
            preprocessing = ["Text lowercasing and punctuation cleaning", "Tokenization & stop-word handling", "TF-IDF vectorization or subword embeddings"]
            risks = ["Out-of-vocabulary words", "Slang and contextual ambiguity", "Severe class imbalance"]

        elif any(w in p_lower for w in ["cluster", "grouping", "segmentation", "unsupervised"]):
            problem_type = "clustering"
            objective = "Discover natural clusters or groupings within data without explicit ground-truth labels."
            expected_input = "Unlabeled feature vectors."
            expected_output = "Cluster partition assignments."
            likely_format = "csv"
            candidate_metrics = ["Silhouette Score", "Calinski-Harabasz Index", "Davies-Bouldin Score"]
            model_families = ["K-Means", "DBSCAN", "Hierarchical Agglomerative Clustering"]
            preprocessing = ["Robust numeric scaling", "Outlier removal", "Dimensionality reduction (PCA)"]
            risks = ["Arbitrary choice of K", "Curse of dimensionality", "Sensitive to scale differences"]

        else:
            # Default Tabular Classification
            problem_type = "tabular_classification"
            objective = "Classify structured samples into discrete categories."
            expected_input = "Tabular records of independent attributes."
            expected_output = "Discrete class label or target probability."
            likely_format = "csv"
            candidate_metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
            model_families = ["Random Forest Classifier", "Gradient Boosting Classifier", "Logistic Regression", "MLP Classifier"]
            preprocessing = ["Median/mode missing imputation", "StandardScaler normalization", "One-hot encoding", "Leakage-free train/test splitting"]
            risks = ["Class distribution skewness", "Multicollinear attributes", "Data leakage between training and testing splits"]

        understanding = ProblemUnderstanding(
            problem_type=problem_type,
            objective=objective,
            expected_input=expected_input,
            expected_output=expected_output,
            likely_data_format=likely_format,
            target_variable="target",
            candidate_metrics=candidate_metrics,
            possible_model_families=model_families,
            preprocessing_requirements=preprocessing,
            potential_risks=risks,
            is_deep_learning=is_dl,
            explanation=f"Parsed problem statement '{problem_statement[:60]}...' as a {problem_type.replace('_', ' ')} task."
        )

        return {"problem_understanding": getattr(understanding, "model_dump", understanding.dict)(), "source": "rule_based"}
