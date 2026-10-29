"""Problem Understanding Agent — Upgraded with dataset-free detection."""
import re
from typing import Dict, Any
from agents.base_agent import BaseAgent
from schemas.experiment_schema import ProblemUnderstanding


# Patterns that indicate NO external dataset is needed
DATASET_FREE_PATTERNS = [
    # Mathematical optimisation
    r"gradient\s*descent", r"minimize\s*(a\s*)?function", r"optimiz(e|ation)",
    r"convergence", r"loss\s*function", r"backpropagation\s*demo",
    # Algorithm demonstrations
    r"activation\s*function", r"compare\s*(activation|optimizer)", r"relu|sigmoid|tanh|softmax",
    r"visualize\s*(activation|convergence|gradient)",
    r"implement\s*(gradient|algorithm|backprop|perceptron)",
    # Synthetic / demo
    r"synthetic\s*data", r"make_blobs", r"make_classification", r"make_regression",
    r"demonstrate\s*(clustering|k.?means)", r"dummy\s*data", r"random\s*dataset",
    # Math / visualization only
    r"visualize\s*(math|function|curve|manifold)", r"plot\s*(function|math|curve)",
    r"compute\s*(gradient|jacobian|hessian)", r"numerical\s*(method|differentiation)",
    r"fourier|wavelet|pca\s*demo", r"dimensionality\s*reduction\s*demo",
]

# Synthetic data patterns (need code but not real dataset)
SYNTHETIC_PATTERNS = [
    r"k.?means", r"clustering\s*(demo|example|tutorial|visualization)",
    r"gaussian\s*mixture", r"dbscan\s*demo", r"anomaly\s*detection\s*demo",
    r"decision\s*boundary", r"svm\s*demo", r"hyperplane",
]

# Mathematical / optimization patterns
MATH_PATTERNS = [
    r"gradient\s*descent", r"stochastic\s*gradient", r"adam\s*optimizer",
    r"convergence", r"loss\s*landscape", r"backprop",
    r"implement\s*perceptron", r"xor\s*problem", r"linear\s*algebra",
]

# Comparison/visualization patterns
COMPARISON_PATTERNS = [
    r"compare\s+.*activation", r"compare\s+(different\s+)?(activation|optimizer|model|algorithm)",
    r"activation\s*function.*comparison", r"different\s*activation",
    r"which\s*(activation|optimizer|algorithm)\s*is\s*better",
]


def _matches_any(text: str, patterns: list) -> bool:
    for p in patterns:
        if re.search(p, text, re.IGNORECASE):
            return True
    return False


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
                f"'candidate_metrics': [...], 'possible_model_families': [...], "
                f"'preprocessing_requirements': [...], 'potential_risks': [...], "
                f"'is_deep_learning': bool, 'explanation': '...', 'dataset_required': bool, "
                f"'experiment_category': '...', 'can_run_without_dataset': bool}}"
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

        # ─────────────────────────────────────────────────────────────────────
        # Robust Rule-Based Understanding (upgraded)
        # ─────────────────────────────────────────────────────────────────────
        p_lower = problem_statement.lower()

        is_dl = False
        if any(term in p_lower for term in ["deep learning", "neural", "image", "vision", "audio",
                                             "cnn", "transformer", "bert", "resnet", "lstm", "rnn"]):
            is_dl = True
        if user_preference.lower() == "deep learning":
            is_dl = True

        # ── Detect dataset_required ──────────────────────────────────────────
        dataset_required = True
        can_run_without_dataset = False
        experiment_category = "tabular_ml"

        if _matches_any(p_lower, DATASET_FREE_PATTERNS):
            dataset_required = False
            can_run_without_dataset = True

        if _matches_any(p_lower, SYNTHETIC_PATTERNS):
            dataset_required = False
            can_run_without_dataset = True

        if _matches_any(p_lower, MATH_PATTERNS):
            dataset_required = False
            can_run_without_dataset = True
            experiment_category = "mathematical"

        if _matches_any(p_lower, COMPARISON_PATTERNS):
            dataset_required = False
            can_run_without_dataset = True
            experiment_category = "comparison"

        # ── Infer Problem Type and Category ───────────────────────────────────
        if _matches_any(p_lower, COMPARISON_PATTERNS) or ("compare" in p_lower and "activation" in p_lower):
            problem_type = "algorithm_comparison"
            experiment_category = "comparison"
            objective = "Compare multiple algorithms, activation functions, or optimizers and visualize differences."
            expected_input = "Configuration parameters for each algorithm."
            expected_output = "Side-by-side comparison plots and performance metrics."
            likely_format = "none"
            candidate_metrics = ["Visual comparison", "Convergence speed", "Output range"]
            model_families = ["Sigmoid", "ReLU", "Tanh", "Leaky ReLU", "Softmax"]
            preprocessing = ["Define input range", "Compute outputs", "Plot comparison"]
            risks = ["Misinterpreting scale differences", "Gradient vanishing in sigmoid"]
            workflow_stages = ["problem_understanding", "code_generation", "execution", "visualization", "report"]

        elif _matches_any(p_lower, MATH_PATTERNS) or re.search(r"implement.*(algorithm|gradient|perceptron)", p_lower):
            problem_type = "mathematical_optimization"
            experiment_category = "mathematical"
            objective = "Implement and visualize a mathematical algorithm or optimization process."
            expected_input = "Mathematical function definition or algorithm parameters."
            expected_output = "Convergence plot, loss curve, or visualization of the algorithm."
            likely_format = "none"
            candidate_metrics = ["Convergence Rate", "Final Loss Value", "Iterations to Converge"]
            model_families = ["NumPy-based Implementation", "PyTorch Autograd"]
            preprocessing = ["Define mathematical function", "Set learning rate", "Choose stopping criterion"]
            risks = ["Divergence if learning rate too high", "Slow convergence", "Local minima"]
            workflow_stages = ["problem_understanding", "code_generation", "execution", "visualization", "report"]

        elif _matches_any(p_lower, SYNTHETIC_PATTERNS):
            problem_type = "synthetic_clustering"
            experiment_category = "synthetic_data"
            objective = "Demonstrate clustering or ML algorithm on synthetically generated data."
            expected_input = "Generated synthetic dataset (make_blobs or similar)."
            expected_output = "Cluster assignments, silhouette score, cluster plot."
            likely_format = "generated"
            candidate_metrics = ["Silhouette Score", "Calinski-Harabasz Index", "Davies-Bouldin Score"]
            model_families = ["K-Means", "DBSCAN", "Hierarchical Agglomerative Clustering"]
            preprocessing = ["Generate synthetic data", "Scale features", "Choose number of clusters"]
            risks = ["Arbitrary choice of K", "Sensitivity to initialization"]
            workflow_stages = ["problem_understanding", "code_generation", "execution", "visualization", "report"]

        elif any(w in p_lower for w in ["price", "cost", "salary", "temperature", "forecast",
                                         "revenue", "continuous", "housing", "regression", "value"]):
            problem_type = "tabular_regression"
            experiment_category = "tabular_ml"
            objective = "Predict continuous numerical value from input features."
            expected_input = "Tabular numeric and categorical feature vectors."
            expected_output = "Continuous real-valued prediction."
            likely_format = "csv"
            candidate_metrics = ["R² Score", "RMSE", "MAE", "MSE"]
            model_families = ["Random Forest Regressor", "Gradient Boosting Regressor", "Ridge Regression", "MLP Regressor"]
            preprocessing = ["Missing value imputation", "Standard feature scaling", "Categorical encoding"]
            risks = ["Sensitivity to extreme outliers", "Multicollinearity between features", "Target skewness"]
            workflow_stages = ["problem_understanding", "dataset_search", "dataset_selection", "data_analysis",
                               "preprocessing", "code_generation", "execution", "evaluation", "visualization", "report"]

        elif any(w in p_lower for w in ["image", "images", "picture", "photo", "pixels", "scan",
                                         "x-ray", "mri", "tumor scan", "vision", "leaf disease", "plant disease", "leaf classification"]):
            problem_type = "image_classification"
            experiment_category = "computer_vision"
            is_dl = True
            objective = "Classify visual images into target categories or diagnoses."
            expected_input = "2D image tensors (RGB / Grayscale)."
            expected_output = "Discrete diagnostic or categorical label."
            likely_format = "image_folder / png"
            candidate_metrics = ["Accuracy", "Weighted F1-Score", "Recall", "ROC-AUC"]
            model_families = ["Convolutional Neural Networks (CNN)", "Pretrained ResNet/MobileNet", "Vision Transformer"]
            preprocessing = ["Image resizing (e.g., 224×224)", "Pixel normalization", "Data augmentation"]
            risks = ["Class imbalance in rare disease categories", "High computational memory requirements", "Domain shift"]
            workflow_stages = ["problem_understanding", "dataset_search", "dataset_selection", "data_analysis",
                               "preprocessing", "code_generation", "execution", "evaluation", "visualization", "report"]

        elif any(w in p_lower for w in ["text", "sentiment", "review", "nlp", "tweet", "comment",
                                         "document", "classification", "spam"]):
            problem_type = "text_classification"
            experiment_category = "nlp"
            objective = "Categorize text sequences into semantic sentiment or topic classes."
            expected_input = "Raw natural language text strings."
            expected_output = "Class probability distribution or discrete category."
            likely_format = "csv / text"
            candidate_metrics = ["Accuracy", "Precision", "Recall", "Macro F1-Score"]
            model_families = ["TF-IDF + Logistic Regression", "Random Forest Classifier", "Transformer / DistilBERT"]
            preprocessing = ["Text lowercasing and punctuation cleaning", "Tokenization", "TF-IDF vectorization"]
            risks = ["Out-of-vocabulary words", "Slang and contextual ambiguity", "Severe class imbalance"]
            workflow_stages = ["problem_understanding", "dataset_search", "dataset_selection", "data_analysis",
                               "preprocessing", "code_generation", "execution", "evaluation", "visualization", "report"]

        elif any(w in p_lower for w in ["cluster", "grouping", "unsupervised", "segment"]):
            problem_type = "clustering"
            experiment_category = "tabular_ml"
            objective = "Discover natural clusters or groupings within data without explicit ground-truth labels."
            expected_input = "Unlabeled feature vectors."
            expected_output = "Cluster partition assignments."
            likely_format = "csv"
            candidate_metrics = ["Silhouette Score", "Calinski-Harabasz Index", "Davies-Bouldin Score"]
            model_families = ["K-Means", "DBSCAN", "Hierarchical Agglomerative Clustering"]
            preprocessing = ["Robust numeric scaling", "Outlier removal", "Dimensionality reduction (PCA)"]
            risks = ["Arbitrary choice of K", "Curse of dimensionality", "Sensitive to scale differences"]
            workflow_stages = ["problem_understanding", "dataset_search", "dataset_selection", "data_analysis",
                               "preprocessing", "code_generation", "execution", "evaluation", "visualization", "report"]

        elif any(w in p_lower for w in ["time series", "time-series", "forecast", "arima", "lstm forecast",
                                         "sequence", "temporal"]):
            problem_type = "time_series_forecasting"
            experiment_category = "time_series"
            objective = "Forecast future values of a temporal sequence."
            expected_input = "Ordered time-indexed data points."
            expected_output = "Future value predictions with confidence intervals."
            likely_format = "csv"
            candidate_metrics = ["MAE", "RMSE", "MAPE", "Directional Accuracy"]
            model_families = ["ARIMA", "Prophet", "LSTM", "Gradient Boosting Regressor"]
            preprocessing = ["Resampling", "Differencing for stationarity", "Lag feature engineering"]
            risks = ["Non-stationarity", "Concept drift", "Insufficient history"]
            workflow_stages = ["problem_understanding", "dataset_search", "dataset_selection", "data_analysis",
                               "preprocessing", "code_generation", "execution", "evaluation", "visualization", "report"]

        else:
            # Default Tabular Classification
            problem_type = "tabular_classification"
            experiment_category = "tabular_ml"
            objective = "Classify structured samples into discrete categories."
            expected_input = "Tabular records of independent attributes."
            expected_output = "Discrete class label or target probability."
            likely_format = "csv"
            candidate_metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
            model_families = ["Random Forest Classifier", "Gradient Boosting Classifier", "Logistic Regression", "MLP Classifier"]
            preprocessing = ["Median/mode missing imputation", "StandardScaler normalization", "One-hot encoding", "Leakage-free train/test splitting"]
            risks = ["Class distribution skewness", "Multicollinear attributes", "Data leakage between training and testing splits"]
            workflow_stages = ["problem_understanding", "dataset_search", "dataset_selection", "data_analysis",
                               "preprocessing", "code_generation", "execution", "evaluation", "visualization", "report"]

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
            explanation=f"Parsed problem statement '{problem_statement[:60]}...' as a {problem_type.replace('_', ' ')} task.",
            dataset_required=dataset_required,
            experiment_category=experiment_category,
            workflow_stages=workflow_stages,
            can_run_without_dataset=can_run_without_dataset,
        )

        return {
            "problem_understanding": getattr(understanding, "model_dump", understanding.dict)(),
            "source": "rule_based"
        }
