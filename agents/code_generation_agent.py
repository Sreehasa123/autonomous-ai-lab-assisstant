"""Code Generation Agent — Upgraded with problem-specific specialized generators."""
import json
from typing import Dict, Any
from agents.base_agent import BaseAgent


class CodeGenerationAgent(BaseAgent):
    """Generates complete, readable, reproducible Python training scripts."""

    def __init__(self):
        super().__init__(
            name="Code Generation Agent",
            description="Constructs executable Python pipelines matching the exact problem category and model."
        )

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        experiment_id = context.get("experiment_id", "exp_001")
        dataset_meta = context.get("selected_dataset", {})
        prep_config = context.get("preprocessing_config", {})
        model_config = context.get("model_config", {})
        problem_understanding = context.get("problem_understanding", {})

        experiment_category = problem_understanding.get("experiment_category", "tabular_ml")
        problem_type = problem_understanding.get("problem_type",
                       dataset_meta.get("task_type", "tabular_classification"))
        problem_statement = context.get("problem_statement", "")

        # ── Route to specialized generator ───────────────────────────────────
        if experiment_category == "mathematical" or "gradient_descent" in problem_type or "optimization" in problem_type:
            code = self._generate_gradient_descent_code(experiment_id, problem_statement)

        elif experiment_category == "comparison" or "comparison" in problem_type or "activation" in problem_type:
            code = self._generate_activation_comparison_code(experiment_id, problem_statement)

        elif experiment_category == "synthetic_data" or "synthetic" in problem_type:
            code = self._generate_kmeans_synthetic_code(experiment_id, problem_statement)

        elif experiment_category == "computer_vision" or "image" in problem_type:
            code = self._generate_image_classification_code(experiment_id, dataset_meta, prep_config, model_config)

        elif experiment_category == "nlp" or "text" in problem_type:
            code = self._generate_text_classification_code(experiment_id, dataset_meta, prep_config, model_config)

        elif experiment_category == "time_series" or "time_series" in problem_type:
            code = self._generate_time_series_code(experiment_id, dataset_meta, prep_config, model_config)

        else:
            # Default: tabular ML (classification or regression)
            code = self._generate_tabular_ml_code(
                experiment_id, dataset_meta, prep_config, model_config, problem_type
            )

        return {"generated_code": code}

    # ─────────────────────────────────────────────────────────────────────────
    # Specialized Generators
    # ─────────────────────────────────────────────────────────────────────────

    def _generate_gradient_descent_code(self, experiment_id: str, problem_statement: str) -> str:
        return f'''"""
Autonomous AI Lab Assistant — Gradient Descent Visualization
Experiment ID: {experiment_id}
Problem: {problem_statement[:80]}
"""
import numpy as np
import json
import os

# ── Define objective function ────────────────────────────────────────────────
def f(x):
    """Rosenbrock-style function: f(x) = x^4 - 4x^2 + x  (simple 1D version)"""
    return x**4 - 4*x**2 + x

def df(x):
    """Analytical derivative."""
    return 4*x**3 - 8*x + 1

# ── Gradient Descent Implementation ─────────────────────────────────────────
def gradient_descent(start_x=3.0, learning_rate=0.01, n_iterations=200, tolerance=1e-7):
    x = start_x
    history = []
    for i in range(n_iterations):
        grad = df(x)
        loss = f(x)
        history.append({{"iteration": i, "x": float(x), "loss": float(loss), "gradient": float(grad)}})
        x = x - learning_rate * grad
        if abs(grad) < tolerance:
            print(f"Converged at iteration {{i}} with |grad| = {{abs(grad):.2e}}")
            break
    return x, history

print("Running gradient descent optimization...")
x_opt, history = gradient_descent(start_x=3.0, learning_rate=0.01, n_iterations=200)

losses = [h["loss"] for h in history]
gradients = [h["gradient"] for h in history]

print(f"Optimization complete.")
print(f"Optimal x: {{x_opt:.6f}}")
print(f"Minimum f(x): {{f(x_opt):.6f}}")
print(f"Total iterations: {{len(history)}}")
print(f"Initial loss: {{losses[0]:.4f}}")
print(f"Final loss: {{losses[-1]:.4f}}")
print(f"Loss reduction: {{losses[0] - losses[-1]:.4f}}")

# ── Save results ─────────────────────────────────────────────────────────────
results = {{
    "experiment_id": "{experiment_id}",
    "optimal_x": float(x_opt),
    "minimum_value": float(f(x_opt)),
    "total_iterations": len(history),
    "initial_loss": float(losses[0]),
    "final_loss": float(losses[-1]),
    "convergence_rate": float((losses[0] - losses[-1]) / max(abs(losses[0]), 1e-8)),
    "history": history[:50]  # first 50 steps for storage
}}
print("RESULTS_JSON:" + json.dumps(results))
'''

    def _generate_activation_comparison_code(self, experiment_id: str, problem_statement: str) -> str:
        return f'''"""
Autonomous AI Lab Assistant — Activation Function Comparison
Experiment ID: {experiment_id}
Problem: {problem_statement[:80]}
"""
import numpy as np
import json

# ── Activation functions ──────────────────────────────────────────────────────
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def swish(x):
    return x * sigmoid(x)

# ── Derivatives ───────────────────────────────────────────────────────────────
def d_sigmoid(x):
    s = sigmoid(x)
    return s * (1 - s)

def d_relu(x):
    return (x > 0).astype(float)

def d_tanh(x):
    return 1 - np.tanh(x)**2

# ── Compute over input range ──────────────────────────────────────────────────
x = np.linspace(-5, 5, 500)

activations = {{
    "Sigmoid": sigmoid(x).tolist(),
    "ReLU": relu(x).tolist(),
    "Tanh": tanh(x).tolist(),
    "Leaky ReLU": leaky_relu(x).tolist(),
    "ELU": elu(x).tolist(),
    "Swish": swish(x).tolist(),
}}

derivatives = {{
    "Sigmoid (derivative)": d_sigmoid(x).tolist(),
    "ReLU (derivative)": d_relu(x).tolist(),
    "Tanh (derivative)": d_tanh(x).tolist(),
}}

# ── Analysis ──────────────────────────────────────────────────────────────────
analysis = {{}}
for name, vals in activations.items():
    arr = np.array(vals)
    analysis[name] = {{
        "output_range": [float(arr.min()), float(arr.max())],
        "mean_output": float(arr.mean()),
        "std_output": float(arr.std()),
        "saturates_positive": bool(abs(arr[-1] - arr[-10]) < 0.01),
        "saturates_negative": bool(abs(arr[0] - arr[10]) < 0.01),
    }}

print("=== Activation Function Comparison ===")
for name, stats in analysis.items():
    print(f"{{name}}:")
    print(f"  Output range: [{{stats['output_range'][0]:.3f}}, {{stats['output_range'][1]:.3f}}]")
    print(f"  Mean output : {{stats['mean_output']:.4f}}")
    print(f"  Saturates (+): {{stats['saturates_positive']}} | Saturates (-): {{stats['saturates_negative']}}")
    print()

results = {{
    "experiment_id": "{experiment_id}",
    "x_values": x.tolist(),
    "activations": activations,
    "derivatives": derivatives,
    "analysis": analysis,
    "summary": "Compared 6 activation functions: Sigmoid, ReLU, Tanh, Leaky ReLU, ELU, Swish"
}}
print("RESULTS_JSON:" + json.dumps(results))
'''

    def _generate_kmeans_synthetic_code(self, experiment_id: str, problem_statement: str) -> str:
        return f'''"""
Autonomous AI Lab Assistant — K-Means Clustering on Synthetic Data
Experiment ID: {experiment_id}
Problem: {problem_statement[:80]}
"""
import numpy as np
import json
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

np.random.seed(42)

# ── Generate synthetic data ───────────────────────────────────────────────────
print("Generating synthetic dataset...")
n_clusters = 4
n_samples = 400
X, y_true = make_blobs(n_samples=n_samples, centers=n_clusters, n_features=2,
                        cluster_std=1.2, random_state=42)

print(f"Dataset shape: {{X.shape}}")
print(f"True clusters: {{n_clusters}}")

# ── Preprocessing ─────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── Elbow method to find optimal K ───────────────────────────────────────────
print("\\nRunning Elbow Method...")
inertias = []
k_range = range(2, 9)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(float(km.inertia_))
    print(f"  K={{k}}: Inertia={{km.inertia_:.2f}}")

# ── Fit optimal K-Means ───────────────────────────────────────────────────────
print(f"\\nFitting K-Means with K={{n_clusters}}...")
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
labels = kmeans.fit_predict(X_scaled)

# ── Evaluation metrics ────────────────────────────────────────────────────────
sil_score = float(silhouette_score(X_scaled, labels))
ch_score = float(calinski_harabasz_score(X_scaled, labels))
db_score = float(davies_bouldin_score(X_scaled, labels))
inertia = float(kmeans.inertia_)

print(f"\\n=== Clustering Evaluation ===")
print(f"Silhouette Score         : {{sil_score:.4f}} (higher is better, max=1.0)")
print(f"Calinski-Harabasz Index  : {{ch_score:.2f}} (higher is better)")
print(f"Davies-Bouldin Score     : {{db_score:.4f}} (lower is better)")
print(f"Inertia (WCSS)          : {{inertia:.2f}}")

# ── Cluster stats ─────────────────────────────────────────────────────────────
cluster_stats = {{}}
for c in range(n_clusters):
    mask = labels == c
    cluster_stats[f"Cluster {{c}}"] = {{
        "size": int(mask.sum()),
        "center_x": float(kmeans.cluster_centers_[c, 0]),
        "center_y": float(kmeans.cluster_centers_[c, 1]),
        "percentage": float(mask.sum() / n_samples * 100),
    }}
    print(f"  Cluster {{c}}: {{mask.sum()}} points ({{mask.sum()/n_samples*100:.1f}}%)")

results = {{
    "experiment_id": "{experiment_id}",
    "n_samples": n_samples,
    "n_clusters": n_clusters,
    "silhouette_score": sil_score,
    "calinski_harabasz_score": ch_score,
    "davies_bouldin_score": db_score,
    "inertia": inertia,
    "cluster_stats": cluster_stats,
    "elbow_data": {{"k_values": list(k_range), "inertias": inertias}},
    "X": X.tolist(),
    "labels": labels.tolist(),
    "centers": kmeans.cluster_centers_.tolist(),
    "summary": f"K-Means clustering with K={{n_clusters}} clusters. Silhouette={{sil_score:.3f}}"
}}
print("RESULTS_JSON:" + json.dumps(results))
'''

    def _generate_image_classification_code(self, experiment_id: str, dataset_meta: Dict,
                                              prep_config: Dict, model_config: Dict) -> str:
        random_state = prep_config.get("random_state", 42)
        return f'''"""
Autonomous AI Lab Assistant — Image Classification Demo
Experiment ID: {experiment_id}
Uses sklearn digits dataset as a runnable demo.
"""
import numpy as np
import json
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import time

np.random.seed({random_state})

print("Loading digits dataset (8x8 pixel images)...")
digits = load_digits()
X, y = digits.data, digits.target
print(f"Dataset shape: {{X.shape}}, Classes: {{len(np.unique(y))}}")

# Normalize pixel values
X = X / 16.0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state={random_state}, stratify=y)
print(f"Train: {{X_train.shape[0]}}, Test: {{X_test.shape[0]}}")

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

print("\\nTraining Random Forest Classifier...")
start = time.time()
model = RandomForestClassifier(n_estimators=100, random_state={random_state})
model.fit(X_train_s, y_train)
duration = time.time() - start
print(f"Training completed in {{duration:.2f}}s")

y_pred = model.predict(X_test_s)
acc = float(accuracy_score(y_test, y_pred))
f1 = float(f1_score(y_test, y_pred, average="weighted"))

print(f"\\n=== Evaluation Results ===")
print(f"Accuracy       : {{acc:.4f}}")
print(f"Weighted F1    : {{f1:.4f}}")
print(classification_report(y_test, y_pred))

results = {{
    "experiment_id": "{experiment_id}",
    "dataset": "sklearn digits (8x8 pixel images)",
    "n_classes": int(len(np.unique(y))),
    "accuracy": acc,
    "weighted_f1": f1,
    "training_time": duration,
    "n_train": int(X_train.shape[0]),
    "n_test": int(X_test.shape[0]),
    "y_test": y_test.tolist(),
    "y_pred": y_pred.tolist(),
    "summary": f"Image classification with RF: Accuracy={{acc:.3f}}, F1={{f1:.3f}}"
}}
print("RESULTS_JSON:" + json.dumps(results))
'''

    def _generate_text_classification_code(self, experiment_id: str, dataset_meta: Dict,
                                             prep_config: Dict, model_config: Dict) -> str:
        random_state = prep_config.get("random_state", 42)
        return f'''"""
Autonomous AI Lab Assistant — Text Classification Demo
Experiment ID: {experiment_id}
Uses 20 Newsgroups dataset (subset) as a runnable demo.
"""
import numpy as np
import json
import time
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

np.random.seed({random_state})

# Use 4-class subset for speed
categories = ["sci.med", "sci.space", "comp.graphics", "rec.sport.baseball"]
print("Loading 20 Newsgroups dataset (4 categories)...")
data = fetch_20newsgroups(subset="all", categories=categories, remove=("headers", "footers", "quotes"))
X_text, y = data.data, data.target
print(f"Dataset size: {{len(X_text)}} documents, {{len(np.unique(y))}} classes")

X_train_t, X_test_t, y_train, y_test = train_test_split(
    X_text, y, test_size=0.2, random_state={random_state}, stratify=y
)

print("Vectorizing text with TF-IDF...")
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)
X_train_v = vectorizer.fit_transform(X_train_t)
X_test_v = vectorizer.transform(X_test_t)
print(f"Feature matrix: {{X_train_v.shape}}")

print("\\nTraining Logistic Regression...")
start = time.time()
model = LogisticRegression(max_iter=1000, random_state={random_state}, C=1.0)
model.fit(X_train_v, y_train)
duration = time.time() - start
print(f"Training completed in {{duration:.2f}}s")

y_pred = model.predict(X_test_v)
acc = float(accuracy_score(y_test, y_pred))
f1 = float(f1_score(y_test, y_pred, average="weighted"))

print(f"\\n=== Evaluation Results ===")
print(f"Accuracy       : {{acc:.4f}}")
print(f"Weighted F1    : {{f1:.4f}}")
print(classification_report(y_test, y_pred, target_names=data.target_names))

results = {{
    "experiment_id": "{experiment_id}",
    "dataset": "20 Newsgroups (4 categories)",
    "n_classes": int(len(np.unique(y))),
    "accuracy": acc,
    "weighted_f1": f1,
    "training_time": duration,
    "n_train": int(X_train_v.shape[0]),
    "n_test": int(X_test_v.shape[0]),
    "y_test": y_test.tolist(),
    "y_pred": y_pred.tolist(),
    "summary": f"Text classification with LR+TF-IDF: Accuracy={{acc:.3f}}, F1={{f1:.3f}}"
}}
print("RESULTS_JSON:" + json.dumps(results))
'''

    def _generate_time_series_code(self, experiment_id: str, dataset_meta: Dict,
                                    prep_config: Dict, model_config: Dict) -> str:
        random_state = prep_config.get("random_state", 42)
        return f'''"""
Autonomous AI Lab Assistant — Time Series Forecasting Demo
Experiment ID: {experiment_id}
Generates synthetic time series and applies ML forecasting.
"""
import numpy as np
import json
import time
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

np.random.seed({random_state})

print("Generating synthetic time series...")
n = 365
t = np.arange(n)
# Trend + seasonality + noise
trend = 0.05 * t
seasonal = 10 * np.sin(2 * np.pi * t / 52)
noise = np.random.normal(0, 2, n)
series = 50 + trend + seasonal + noise

print(f"Time series length: {{n}} points")
print(f"Mean: {{series.mean():.2f}}, Std: {{series.std():.2f}}")

# Create lag features
def make_lag_features(data, lags=12):
    rows = []
    for i in range(lags, len(data)):
        row = [data[i - l] for l in range(1, lags+1)]
        rows.append((row, data[i]))
    X = np.array([r[0] for r in rows])
    y = np.array([r[1] for r in rows])
    return X, y

X, y = make_lag_features(series, lags=12)

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"Train: {{len(X_train)}}, Test: {{len(X_test)}}")

print("\\nTraining Gradient Boosting Regressor...")
start = time.time()
model = GradientBoostingRegressor(n_estimators=100, random_state={random_state})
model.fit(X_train, y_train)
duration = time.time() - start

y_pred = model.predict(X_test)
mae = float(mean_absolute_error(y_test, y_pred))
rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
r2 = float(r2_score(y_test, y_pred))

print(f"\\n=== Evaluation Results ===")
print(f"MAE    : {{mae:.4f}}")
print(f"RMSE   : {{rmse:.4f}}")
print(f"R²     : {{r2:.4f}}")
print(f"Training time: {{duration:.2f}}s")

results = {{
    "experiment_id": "{experiment_id}",
    "dataset": "Synthetic time series (365 points)",
    "mae": mae,
    "rmse": rmse,
    "r2_score": r2,
    "training_time": duration,
    "n_train": int(len(X_train)),
    "n_test": int(len(X_test)),
    "y_test": y_test.tolist()[:50],
    "y_pred": y_pred.tolist()[:50],
    "summary": f"Time series forecasting with GBR: MAE={{mae:.3f}}, RMSE={{rmse:.3f}}, R²={{r2:.3f}}"
}}
print("RESULTS_JSON:" + json.dumps(results))
'''

    def _generate_tabular_ml_code(self, experiment_id: str, dataset_meta: Dict,
                                    prep_config: Dict, model_config: Dict, task_type: str) -> str:
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
            'import os', 'import sys', 'import time', 'import json', 'import joblib',
            'import numpy as np', 'import pandas as pd',
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
            '', f'np.random.seed({random_state})',
            '', 'print("Loading dataset...")',
            f'target_column = "{target_col}"'
        ])

        ds_id = dataset_meta.get("dataset_id", "")
        if "iris" in ds_id:
            code_lines.append('from sklearn.datasets import load_iris\ndf = load_iris(as_frame=True).frame')
        elif "wine" in ds_id:
            code_lines.append('from sklearn.datasets import load_wine\ndf = load_wine(as_frame=True).frame')
        elif "cancer" in ds_id:
            code_lines.append('from sklearn.datasets import load_breast_cancer\ndf = load_breast_cancer(as_frame=True).frame')
        elif "housing" in ds_id or "california" in ds_id:
            code_lines.append('from sklearn.datasets import fetch_california_housing\ndf = fetch_california_housing(as_frame=True).frame')
        elif "diabetes" in ds_id:
            code_lines.append('from sklearn.datasets import load_diabetes\ndf = load_diabetes(as_frame=True).frame')
        else:
            if "regression" in task_type.lower():
                code_lines.extend([
                    'from sklearn.datasets import make_regression',
                    f'X_raw, y_raw = make_regression(n_samples=500, n_features=10, noise=20, random_state={random_state})',
                    'df = pd.DataFrame(X_raw, columns=[f"feat_{{i}}" for i in range(10)])',
                    'df[target_column] = y_raw'
                ])
            else:
                code_lines.extend([
                    'from sklearn.datasets import make_classification',
                    f'X_raw, y_raw = make_classification(n_samples=500, n_features=10, random_state={random_state})',
                    'df = pd.DataFrame(X_raw, columns=[f"feat_{{i}}" for i in range(10)])',
                    'df[target_column] = y_raw'
                ])

        code_lines.extend([
            f'print(f"Dataset shape: {{df.shape}}")',
            '',
            'if target_column not in df.columns:',
            '    target_column = df.columns[-1]',
            'X = df.drop(columns=[target_column]).select_dtypes(include=[np.number])',
            'y = df[target_column]',
            '',
            f'X_train, X_test, y_train, y_test = train_test_split(X, y, test_size={test_size}, random_state={random_state})',
            f'print(f"Train split: {{X_train.shape[0]}}, Test split: {{X_test.shape[0]}}")',
            '',
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

        code_lines.extend([
            '', f'print("Initializing {model_name}...")',
            f'model = {model_name}(random_state={random_state})',
            'start_time = time.time()',
            'model.fit(X_train_scaled, y_train)',
            'duration = time.time() - start_time',
            'print(f"Training completed in {duration:.3f} seconds.")',
            '',
            'y_pred = model.predict(X_test_scaled)',
        ])

        if "regression" in task_type.lower():
            code_lines.extend([
                'mse = mean_squared_error(y_test, y_pred)',
                'rmse = float(np.sqrt(mse))',
                'mae = float(mean_absolute_error(y_test, y_pred))',
                'r2 = float(r2_score(y_test, y_pred))',
                'print(f"Results -> R2: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")',
                '',
                'results = {"experiment_id": "' + experiment_id + '", "r2_score": r2, "rmse": rmse,',
                '    "mae": mae, "training_time": duration, "y_test": y_test.tolist()[:50], "y_pred": y_pred.tolist()[:50],',
                '    "summary": f"Regression: R2={r2:.3f}, RMSE={rmse:.3f}, MAE={mae:.3f}"}',
                'print("RESULTS_JSON:" + json.dumps(results))'
            ])
        else:
            code_lines.extend([
                'acc = float(accuracy_score(y_test, y_pred))',
                'f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))',
                'print(f"Results -> Accuracy: {acc:.4f}, Weighted F1: {f1:.4f}")',
                '',
                'results = {"experiment_id": "' + experiment_id + '", "accuracy": acc, "weighted_f1": f1,',
                '    "training_time": duration, "y_test": [int(v) for v in y_test.tolist()[:50]],',
                '    "y_pred": [int(v) for v in y_pred.tolist()[:50]],',
                '    "summary": f"Classification: Accuracy={acc:.3f}, F1={f1:.3f}"}',
                'print("RESULTS_JSON:" + json.dumps(results))'
            ])

        return "\n".join(code_lines)
