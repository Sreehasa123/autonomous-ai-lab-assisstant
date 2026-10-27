"""Model Service for Architecture Selection and Instantiation."""
import logging
from typing import Dict, Any, List, Tuple
from schemas.experiment_schema import ModelConfig

logger = logging.getLogger(__name__)

class ModelService:
    """Recommends and instantiates machine learning and deep learning models."""

    def recommend_models(
        self,
        task_type: str,
        num_rows: int = 500,
        num_features: int = 10,
        is_deep_learning: bool = False
    ) -> List[ModelConfig]:
        """Provides a tailored candidate set of models with explanations."""
        recommendations: List[ModelConfig] = []
        is_regression = "regression" in task_type.lower()

        if is_regression:
            # 1. Ensemble Baseline (Primary)
            recommendations.append(ModelConfig(
                model_name="RandomForestRegressor",
                model_family="ensemble",
                framework="scikit-learn",
                hyperparameters={"n_estimators": 100, "max_depth": 10, "random_state": 42},
                is_baseline=True,
                explanation="Non-linear tree ensemble; robust against outliers, minimal hyperparameter tuning needed, and provides natural feature importance."
            ))
            # 2. Gradient Boosting
            recommendations.append(ModelConfig(
                model_name="GradientBoostingRegressor",
                model_family="boosting",
                framework="scikit-learn",
                hyperparameters={"n_estimators": 100, "learning_rate": 0.1, "random_state": 42},
                is_baseline=False,
                explanation="Sequentially minimizes residual errors; often achieves superior predictive accuracy on structured tabular benchmarks."
            ))
            # 3. Regularized Linear Baseline
            recommendations.append(ModelConfig(
                model_name="RidgeRegression",
                model_family="linear",
                framework="scikit-learn",
                hyperparameters={"alpha": 1.0},
                is_baseline=False,
                explanation="L2 regularized linear model; provides high interpretability, fast training, and serves as an essential linear benchmark."
            ))
            # 4. Neural Network (MLP) if DL preferred or requested
            if is_deep_learning:
                recommendations.append(ModelConfig(
                    model_name="MLPRegressor",
                    model_family="neural_network",
                    framework="scikit-learn",
                    hyperparameters={"hidden_layer_sizes": (64, 32), "max_iter": 300, "random_state": 42},
                    is_baseline=False,
                    explanation="Multi-layer Perceptron neural network with non-linear activations for complex continuous surface approximations."
                ))
        else:
            # Classification
            # 1. Ensemble Baseline (Primary)
            recommendations.append(ModelConfig(
                model_name="RandomForestClassifier",
                model_family="ensemble",
                framework="scikit-learn",
                hyperparameters={"n_estimators": 100, "max_depth": 10, "random_state": 42},
                is_baseline=True,
                explanation="High-capacity bagging ensemble; robust against overfitting, supports multiclass naturally, and resilient to scale variations."
            ))
            # 2. Gradient Boosting
            recommendations.append(ModelConfig(
                model_name="GradientBoostingClassifier",
                model_family="boosting",
                framework="scikit-learn",
                hyperparameters={"n_estimators": 100, "learning_rate": 0.1, "random_state": 42},
                is_baseline=False,
                explanation="Boosted decision trees; builds shallow trees iteratively correcting errors to maximize F1-score and discrimination power."
            ))
            # 3. Logistic Regression
            recommendations.append(ModelConfig(
                model_name="LogisticRegression",
                model_family="linear",
                framework="scikit-learn",
                hyperparameters={"C": 1.0, "max_iter": 500, "random_state": 42},
                is_baseline=False,
                explanation="Standard calibrated probabilistic baseline; highly interpretable odds ratios and fast convergence."
            ))
            # 4. Neural Network (MLP)
            if is_deep_learning or num_rows > 1000:
                recommendations.append(ModelConfig(
                    model_name="MLPClassifier",
                    model_family="neural_network",
                    framework="scikit-learn",
                    hyperparameters={"hidden_layer_sizes": (64, 32), "max_iter": 300, "random_state": 42},
                    is_baseline=False,
                    explanation="Deep Multi-Layer Perceptron neural network with backpropagation for non-linear decision boundary modeling."
                ))

        return recommendations

    def instantiate_model(self, config: ModelConfig) -> Any:
        """Instantiates a real Python model object with specified parameters."""
        name = config.model_name
        params = config.hyperparameters or {}

        if name == "RandomForestClassifier":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(
                n_estimators=params.get("n_estimators", 100),
                max_depth=params.get("max_depth", None),
                random_state=params.get("random_state", 42)
            )
        elif name == "GradientBoostingClassifier":
            from sklearn.ensemble import GradientBoostingClassifier
            return GradientBoostingClassifier(
                n_estimators=params.get("n_estimators", 100),
                learning_rate=params.get("learning_rate", 0.1),
                random_state=params.get("random_state", 42)
            )
        elif name == "LogisticRegression":
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(
                C=params.get("C", 1.0),
                max_iter=params.get("max_iter", 500),
                random_state=params.get("random_state", 42)
            )
        elif name == "MLPClassifier":
            from sklearn.neural_network import MLPClassifier
            return MLPClassifier(
                hidden_layer_sizes=tuple(params.get("hidden_layer_sizes", (64, 32))),
                max_iter=params.get("max_iter", 300),
                random_state=params.get("random_state", 42)
            )
        elif name == "RandomForestRegressor":
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(
                n_estimators=params.get("n_estimators", 100),
                max_depth=params.get("max_depth", None),
                random_state=params.get("random_state", 42)
            )
        elif name == "GradientBoostingRegressor":
            from sklearn.ensemble import GradientBoostingRegressor
            return GradientBoostingRegressor(
                n_estimators=params.get("n_estimators", 100),
                learning_rate=params.get("learning_rate", 0.1),
                random_state=params.get("random_state", 42)
            )
        elif name in ("RidgeRegression", "Ridge"):
            from sklearn.linear_model import Ridge
            return Ridge(
                alpha=params.get("alpha", 1.0),
                random_state=params.get("random_state", 42)
            )
        elif name == "MLPRegressor":
            from sklearn.neural_network import MLPRegressor
            return MLPRegressor(
                hidden_layer_sizes=tuple(params.get("hidden_layer_sizes", (64, 32))),
                max_iter=params.get("max_iter", 300),
                random_state=params.get("random_state", 42)
            )
        else:
            # Default fallback classifier
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(n_estimators=100, random_state=42)
