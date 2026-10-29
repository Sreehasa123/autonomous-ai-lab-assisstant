"""Execution Repair Agent — Categorizes errors and applies targeted code fixes."""
import re
from typing import Dict, Any, Tuple
from agents.base_agent import BaseAgent


class ExecutionRepairAgent(BaseAgent):
    """
    Analyzes execution errors and applies targeted repairs to generated code.
    Works with the CodeExecutionService's repair loop.
    """

    def __init__(self):
        super().__init__(
            name="Execution Repair Agent",
            description="Categorizes execution errors and applies targeted string-level code repairs."
        )

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze error and attempt code repair.

        Context keys:
        - code: str — the failing code
        - stderr: str — the error output
        - error_type: str — classified error type
        - attempt: int — current attempt number

        Returns:
        - repaired_code: str
        - repair_applied: bool
        - repair_description: str
        - can_repair: bool
        """
        code = context.get("code", "")
        stderr = context.get("stderr", "")
        error_type = context.get("error_type", "")
        attempt = context.get("attempt", 0)

        if not code or not stderr:
            return {
                "repaired_code": code,
                "repair_applied": False,
                "repair_description": "No code or error provided.",
                "can_repair": False,
            }

        # Cannot repair after too many attempts
        if attempt >= 3:
            return {
                "repaired_code": code,
                "repair_applied": False,
                "repair_description": "Maximum repair attempts reached.",
                "can_repair": False,
            }

        repaired, description, can_repair = self._apply_repair(code, stderr, error_type)

        return {
            "repaired_code": repaired,
            "repair_applied": repaired != code,
            "repair_description": description,
            "can_repair": can_repair,
        }

    def _apply_repair(self, code: str, stderr: str, error_type: str) -> Tuple[str, str, bool]:
        """Apply targeted repair based on error type."""

        if error_type == "ImportError":
            return self._repair_import_error(code, stderr)

        elif error_type == "NameError":
            return self._repair_name_error(code, stderr)

        elif error_type == "FileNotFoundError":
            return self._repair_file_not_found(code, stderr)

        elif error_type == "SyntaxError":
            return self._repair_syntax_error(code, stderr)

        elif error_type == "ValueError":
            return self._repair_value_error(code, stderr)

        elif error_type == "TypeError":
            return self._repair_type_error(code, stderr)

        elif error_type == "TimeoutError":
            return self._repair_timeout(code)

        elif error_type == "MemoryError":
            return self._repair_memory_error(code)

        else:
            return code, f"Unknown error type: {error_type}. Cannot auto-repair.", False

    def _repair_import_error(self, code: str, stderr: str) -> Tuple[str, str, bool]:
        """Fix import errors by adding fallbacks or using alternatives."""
        module_match = re.search(r"No module named '([^']+)'", stderr)
        if not module_match:
            return code, "Could not identify missing module.", False

        missing = module_match.group(1)
        description = f"Handling missing module: {missing}"

        # Known alternative mappings
        ALTERNATIVES = {
            "torch": "# PyTorch not available — using sklearn MLP instead\nfrom sklearn.neural_network import MLPClassifier",
            "tensorflow": "# TensorFlow not available — using sklearn instead\nfrom sklearn.neural_network import MLPClassifier",
            "keras": "# Keras not available — using sklearn instead\nfrom sklearn.neural_network import MLPClassifier",
            "cv2": "# OpenCV not available — using PIL instead\nfrom PIL import Image\nimport numpy as np",
            "PIL": "# PIL not available — will skip image loading",
            "transformers": "# HuggingFace Transformers not available — using sklearn instead\nfrom sklearn.linear_model import LogisticRegression",
            "seaborn": "# seaborn not available — using matplotlib only",
            "xgboost": "# XGBoost not available — using GradientBoostingClassifier\nfrom sklearn.ensemble import GradientBoostingClassifier\nXGBClassifier = GradientBoostingClassifier",
            "lightgbm": "# LightGBM not available — using GradientBoostingClassifier\nfrom sklearn.ensemble import GradientBoostingClassifier",
        }

        if missing in ALTERNATIVES:
            # Replace the import line
            lines = code.split("\n")
            new_lines = []
            for line in lines:
                if f"import {missing}" in line or f"from {missing}" in line:
                    new_lines.append(ALTERNATIVES[missing])
                else:
                    new_lines.append(line)
            return "\n".join(new_lines), f"Replaced '{missing}' with available alternative.", True

        # Generic: wrap with try-except
        lines = code.split("\n")
        new_lines = []
        for line in lines:
            if f"import {missing}" in line or f"from {missing}" in line:
                new_lines.extend([
                    "try:",
                    f"    {line}",
                    "except ImportError:",
                    f"    print(f'Warning: {missing} not available, skipping.')",
                    f"    {missing.split('.')[0]} = None",
                ])
            else:
                new_lines.append(line)
        return "\n".join(new_lines), f"Wrapped '{missing}' import with try-except.", True

    def _repair_name_error(self, code: str, stderr: str) -> Tuple[str, str, bool]:
        """Fix NameError for common undefined names."""
        name_match = re.search(r"name '([^']+)' is not defined", stderr)
        if not name_match:
            return code, "Could not identify undefined name.", False

        undefined = name_match.group(1)

        if undefined == "display":
            repaired = code.replace("display(", "print(")
            return repaired, "Replaced display() with print().", True

        if undefined == "plt":
            header = "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n"
            return header + code, "Added matplotlib import.", True

        if undefined == "np":
            return "import numpy as np\n" + code, "Added numpy import.", True

        if undefined == "pd":
            return "import pandas as pd\n" + code, "Added pandas import.", True

        if undefined == "json":
            return "import json\n" + code, "Added json import.", True

        return code, f"Cannot auto-fix undefined name: {undefined}", False

    def _repair_file_not_found(self, code: str, stderr: str) -> Tuple[str, str, bool]:
        """Replace file loading with synthetic data generation."""
        if "read_csv" not in code and "open(" not in code:
            return code, "File error not related to data loading.", False

        synthetic = '''
# File not found — replacing with synthetic dataset
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

X_syn, y_syn = make_classification(n_samples=400, n_features=8, random_state=42)
df = pd.DataFrame(X_syn, columns=[f"feat_{i}" for i in range(8)])
df["target"] = y_syn
target_column = "target"
print("Using synthetic dataset (file not available)")
'''
        lines = code.split("\n")
        filtered = [l for l in lines if "read_csv" not in l and "open(" not in l
                    and "pd.read_" not in l]
        return synthetic + "\n".join(filtered), "Replaced file loading with synthetic data.", True

    def _repair_syntax_error(self, code: str, stderr: str) -> Tuple[str, str, bool]:
        """Try common syntax fixes."""
        # Extract line number
        line_match = re.search(r"line (\d+)", stderr)

        # Fix common issues
        repaired = code
        # Fix unmatched quotes in f-strings
        repaired = re.sub(r"f'([^']*)'", lambda m: f'f"{m.group(1)}"', repaired)

        if repaired != code:
            return repaired, "Fixed f-string quote mismatch.", True
        return code, "Syntax error could not be auto-repaired.", False

    def _repair_value_error(self, code: str, stderr: str) -> Tuple[str, str, bool]:
        """Fix common ValueError issues."""
        repaired = code

        # Fix f1_score without zero_division
        if "zero_division" not in code and "f1_score" in code:
            repaired = code.replace(
                "f1_score(y_test, y_pred, average='weighted')",
                "f1_score(y_test, y_pred, average='weighted', zero_division=0)"
            ).replace(
                'f1_score(y_test, y_pred, average="weighted")',
                'f1_score(y_test, y_pred, average="weighted", zero_division=0)'
            )
            if repaired != code:
                return repaired, "Added zero_division=0 to f1_score.", True

        # Fix NaN handling
        if "nan" in stderr.lower() or "inf" in stderr.lower():
            nan_fix = "\n# Fix NaN/Inf values\nimport numpy as np\n"
            repaired = code.replace(
                "model.fit(", "X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=1e6, neginf=-1e6)\nX_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=1e6, neginf=-1e6)\nmodel.fit("
            )
            if repaired != code:
                return repaired, "Added NaN/Inf handling before model fitting.", True

        return code, "ValueError could not be auto-repaired.", False

    def _repair_type_error(self, code: str, stderr: str) -> Tuple[str, str, bool]:
        """Fix common TypeError issues."""
        if "str" in stderr and "float" in stderr:
            repaired = code.replace(
                "y_pred",
                "y_pred.astype(float)"
            )
            if repaired != code:
                return repaired, "Added float cast to y_pred.", True
        return code, "TypeError could not be auto-repaired.", False

    def _repair_timeout(self, code: str) -> Tuple[str, str, bool]:
        """Reduce computation to fit within timeout."""
        repaired = code

        # Reduce n_estimators
        repaired = re.sub(r"n_estimators=\d+", "n_estimators=50", repaired)
        # Reduce n_samples
        repaired = re.sub(r"n_samples=\d+", "n_samples=200", repaired)
        # Reduce max_iter
        repaired = re.sub(r"max_iter=\d+", "max_iter=100", repaired)
        # Reduce n_iterations
        repaired = re.sub(r"n_iterations=\d+", "n_iterations=100", repaired)

        if repaired != code:
            return repaired, "Reduced computation parameters to fit within timeout.", True
        return code, "Timeout: could not identify parameters to reduce.", False

    def _repair_memory_error(self, code: str) -> Tuple[str, str, bool]:
        """Reduce data size for memory errors."""
        repaired = re.sub(r"n_samples=\d+", "n_samples=200", code)
        repaired = re.sub(r"n_features=\d+", "n_features=8", repaired)
        if repaired != code:
            return repaired, "Reduced dataset size to fit in memory.", True
        return code, "Memory error: could not identify size to reduce.", False
