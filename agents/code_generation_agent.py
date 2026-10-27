"""Code Generation Agent."""
import json
from typing import Dict, Any
from agents.base_agent import BaseAgent

class CodeGenerationAgent(BaseAgent):
    """Generates complete, readable, reproducible Python training scripts."""

    def __init__(self):
        super().__init__(
            name="Code Generation Agent",
            description="Constructs executable, leak-free Python pipelines matching the exact dataset schema and model."
        )

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        experiment_id = context.get("experiment_id", "exp_001")
        dataset_meta = context.get("selected_dataset", {})
        prep_config = context.get("preprocessing_config", {})
        model_config = context.get("model_config", {})
        task_type = dataset_meta.get("task_type", "classification")
        target_col = dataset_meta.get("target_column", "target")
        model_name = model_config.get("model_name", "RandomForestClassifier")
        scale_type = prep_config.get("scale_numeric", "standard")
        test_size = prep_config.get("test_size", 0.2)
        random_state = prep_config.get("random_state", 42)

        code_lines = [
            '"""',
            f'Autonomous AI Lab Assistant - Generated Experiment Script',
            f'Experiment ID: {experiment_id}',
            f'Model Architecture: {model_name}',
            f'Task: {task_type}',
            '"""',
            'import os',
            'import sys',
            'import time',
            'import json',
            'import joblib',
            'import numpy as np',
            'import pandas as pd',
            'from sklearn.model_selection import train_test_split',
            'from sklearn.preprocessing import StandardScaler, MinMaxScaler',
            'from sklearn.impute import SimpleImputer',
            ''
        ]

        if "regression" in task_type.lower():
            code_lines.extend([
                'from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score',
                'from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor',
                'from sklearn.linear_model import Ridge',
                'from sklearn.neural_network import MLPRegressor'
            ])
        else:
            code_lines.extend([
                'from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix',
                'from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier',
                'from sklearn.linear_model import LogisticRegression',
                'from sklearn.neural_network import MLPClassifier'
            ])

        code_lines.extend([
            '',
            '# 1. Reproducibility Seeds',
            f'np.random.seed({random_state})',
            '',
            '# 2. Data Loading',
            'print("Loading dataset...")',
            f'target_column = "{target_col}"'
        ])

        # Dataset ingestion code
        ds_id = dataset_meta.get("dataset_id", "")
        if "iris" in ds_id:
            code_lines.append('from sklearn.datasets import load_iris\ndf = load_iris(as_frame=True).frame')
        elif "wine" in ds_id:
            code_lines.append('from sklearn.datasets import load_wine\ndf = load_wine(as_frame=True).frame')
        elif "cancer" in ds_id:
            code_lines.append('from sklearn.datasets import load_breast_cancer\ndf = load_breast_cancer(as_frame=True).frame')
        elif "housing" in ds_id:
            code_lines.append('from sklearn.datasets import fetch_california_housing\ndf = fetch_california_housing(as_frame=True).frame')
        elif "diabetes" in ds_id:
            code_lines.append('from sklearn.datasets import load_diabetes\ndf = load_diabetes(as_frame=True).frame')
        else:
            code_lines.extend([
                'from sklearn.datasets import make_classification, make_regression',
                f'X_raw, y_raw = make_classification(n_samples=400, n_features=8, random_state={random_state})' if "classification" in task_type else f'X_raw, y_raw = make_regression(n_samples=400, n_features=8, random_state={random_state})',
                'df = pd.DataFrame(X_raw, columns=[f"feat_{i}" for i in range(8)])',
                'df[target_column] = y_raw'
            ])

        code_lines.extend([
            f'print(f"Dataset shape: {{df.shape}}")',
            '',
            '# 3. Features & Target Partitioning',
            'if target_column not in df.columns:',
            '    target_column = df.columns[-1]',
            'X = df.drop(columns=[target_column]).select_dtypes(include=[np.number])',
            'y = df[target_column]',
            '',
            '# 4. Leakage-Free Train/Test Splitting',
            f'X_train, X_test, y_train, y_test = train_test_split(X, y, test_size={test_size}, random_state={random_state})',
            f'print(f"Train split: {{X_train.shape[0]}}, Test split: {{X_test.shape[0]}}")',
            '',
            '# 5. Preprocessing Transformations',
            'imputer = SimpleImputer(strategy="median")',
            'X_train_imp = imputer.fit_transform(X_train)',
            'X_test_imp = imputer.transform(X_test)',
            ''
        ])

        if scale_type == "standard":
            code_lines.extend([
                'scaler = StandardScaler()',
                'X_train_scaled = scaler.fit_transform(X_train_imp)',
                'X_test_scaled = scaler.transform(X_test_imp)'
            ])
        else:
            code_lines.extend([
                'X_train_scaled = X_train_imp',
                'X_test_scaled = X_test_imp'
            ])

        # Model instantiator
        code_lines.extend([
            '',
            '# 6. Model Instantiation & Fitting',
            f'print("Initializing {model_name}...")',
            f'model = {model_name}(random_state={random_state})',
            'start_time = time.time()',
            'model.fit(X_train_scaled, y_train)',
            'duration = time.time() - start_time',
            'print(f"Training completed in {duration:.3f} seconds.")',
            '',
            '# 7. Unbiased Out-of-Sample Evaluation',
            'y_pred = model.predict(X_test_scaled)',
        ])

        if "regression" in task_type.lower():
            code_lines.extend([
                'mse = mean_squared_error(y_test, y_pred)',
                'rmse = float(np.sqrt(mse))',
                'mae = float(mean_absolute_error(y_test, y_pred))',
                'r2 = float(r2_score(y_test, y_pred))',
                'print(f"Results -> R2: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")'
            ])
        else:
            code_lines.extend([
                'acc = float(accuracy_score(y_test, y_pred))',
                'f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))',
                'print(f"Results -> Accuracy: {acc:.4f}, Weighted F1: {f1:.4f}")'
            ])

        generated_script = "\n".join(code_lines)
        return {"generated_code": generated_script}
