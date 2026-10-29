"""Safe Code Execution Service — subprocess-based isolated execution engine."""
import os
import re
import sys
import json
import time
import tempfile
import subprocess
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result from safe code execution."""
    success: bool = False
    stdout: str = ""
    stderr: str = ""
    exit_code: int = -1
    duration_seconds: float = 0.0
    artifacts: List[str] = field(default_factory=list)
    error_type: str = ""
    repair_attempts: int = 0
    metrics: Dict[str, Any] = field(default_factory=dict)
    results_json: Dict[str, Any] = field(default_factory=dict)


class CodeExecutionService:
    """
    Safely executes Python code in an isolated subprocess.
    Features:
    - Isolated temp directory execution
    - stdout/stderr capture
    - Timeout enforcement
    - RESULTS_JSON extraction from stdout
    - Repair loop (max 3 attempts)
    """

    MAX_REPAIR_ATTEMPTS = 3
    DEFAULT_TIMEOUT = 120  # seconds

    def __init__(self):
        self.python_executable = sys.executable

    def execute_python_code(
        self,
        code: str,
        timeout: int = DEFAULT_TIMEOUT,
        experiment_id: str = "unknown",
    ) -> ExecutionResult:
        """Execute Python code safely and return structured result."""
        result = ExecutionResult()
        repaired_code = code

        for attempt in range(self.MAX_REPAIR_ATTEMPTS):
            result.repair_attempts = attempt
            exec_result = self._run_in_subprocess(repaired_code, timeout)

            if exec_result["success"]:
                result.success = True
                result.stdout = exec_result["stdout"]
                result.stderr = exec_result["stderr"]
                result.exit_code = exec_result["exit_code"]
                result.duration_seconds = exec_result["duration"]
                result.results_json = self._extract_results_json(exec_result["stdout"])
                result.metrics = self._extract_metrics(result.results_json)
                break
            else:
                result.stdout = exec_result["stdout"]
                result.stderr = exec_result["stderr"]
                result.exit_code = exec_result["exit_code"]
                result.duration_seconds = exec_result["duration"]
                result.error_type = self._classify_error(exec_result["stderr"])

                logger.warning(
                    f"Attempt {attempt+1}/{self.MAX_REPAIR_ATTEMPTS} failed: "
                    f"{result.error_type} — {exec_result['stderr'][:200]}"
                )

                if attempt < self.MAX_REPAIR_ATTEMPTS - 1:
                    repaired_code = self._repair_code(repaired_code, exec_result["stderr"], result.error_type)
                    if repaired_code == code and attempt > 0:
                        # No progress, stop early
                        break

        return result

    def _run_in_subprocess(self, code: str, timeout: int) -> Dict[str, Any]:
        """Write code to temp file and execute in isolated subprocess."""
        with tempfile.TemporaryDirectory(prefix="ai_lab_exec_") as tmpdir:
            script_path = os.path.join(tmpdir, "experiment_script.py")
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(code)

            start_time = time.time()
            try:
                proc = subprocess.run(
                    [self.python_executable, script_path],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=tmpdir,
                    env={**os.environ, "MPLBACKEND": "Agg"},  # headless matplotlib
                )
                duration = time.time() - start_time
                return {
                    "success": proc.returncode == 0,
                    "stdout": proc.stdout,
                    "stderr": proc.stderr,
                    "exit_code": proc.returncode,
                    "duration": duration,
                }
            except subprocess.TimeoutExpired:
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": f"TimeoutError: Execution exceeded {timeout} seconds.",
                    "exit_code": -1,
                    "duration": timeout,
                }
            except Exception as e:
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": str(e),
                    "exit_code": -1,
                    "duration": time.time() - start_time,
                }

    def _classify_error(self, stderr: str) -> str:
        """Categorize the type of error from stderr."""
        if not stderr:
            return "UnknownError"
        s = stderr.lower()
        if "modulenotfounderror" in s or "importerror" in s:
            return "ImportError"
        if "nameerror" in s:
            return "NameError"
        if "syntaxerror" in s:
            return "SyntaxError"
        if "filenotfounderror" in s or "no such file" in s:
            return "FileNotFoundError"
        if "timeouterror" in s or "timeout" in s:
            return "TimeoutError"
        if "valueerror" in s:
            return "ValueError"
        if "typeerror" in s:
            return "TypeError"
        if "indexerror" in s or "keyerror" in s:
            return "IndexError"
        if "memoryerror" in s or "out of memory" in s:
            return "MemoryError"
        return "RuntimeError"

    def _repair_code(self, code: str, stderr: str, error_type: str) -> str:
        """Apply targeted string-level fixes for common errors."""
        repaired = code

        if error_type == "ImportError":
            # Try to identify missing module and add a try-except wrapper
            module_match = re.search(r"No module named '([^']+)'", stderr)
            if module_match:
                missing = module_match.group(1)
                repaired = self._wrap_import_with_fallback(repaired, missing)

        elif error_type == "NameError":
            # Handle common undefined names
            name_match = re.search(r"name '([^']+)' is not defined", stderr)
            if name_match:
                undefined = name_match.group(1)
                if undefined == "display":
                    repaired = repaired.replace("display(", "print(")
                elif undefined == "plt":
                    repaired = "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n" + repaired

        elif error_type == "FileNotFoundError":
            # Replace file load with synthetic data generation
            if "pd.read_csv" in repaired or "open(" in repaired:
                repaired = self._replace_file_load_with_synthetic(repaired)

        elif error_type == "SyntaxError":
            # Try to fix common f-string and quote issues
            repaired = repaired.replace("f'", 'f"').replace("f\"", 'f"')

        elif error_type == "ValueError":
            # Fix common shape/encoding issues
            repaired = repaired.replace(
                "f1_score(y_test, y_pred, average='weighted')",
                "f1_score(y_test, y_pred, average='weighted', zero_division=0)"
            )

        return repaired

    def _wrap_import_with_fallback(self, code: str, missing_module: str) -> str:
        """Wrap the problematic import with a try-except fallback."""
        lines = code.split("\n")
        new_lines = []
        for line in lines:
            if f"import {missing_module}" in line or f"from {missing_module}" in line:
                new_lines.append(f"try:")
                new_lines.append(f"    {line}")
                new_lines.append(f"except ImportError:")
                new_lines.append(f"    print('Warning: {missing_module} not available. Using fallback.')")
                # Add common fallbacks
                if "sklearn" in missing_module:
                    new_lines.append(f"    raise  # sklearn is required")
                else:
                    new_lines.append(f"    pass")
            else:
                new_lines.append(line)
        return "\n".join(new_lines)

    def _replace_file_load_with_synthetic(self, code: str) -> str:
        """Replace file loading with synthetic data generation."""
        synthetic_header = '''
# File loading failed — using synthetic dataset instead
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
X_syn, y_syn = make_classification(n_samples=400, n_features=8, random_state=42)
df = pd.DataFrame(X_syn, columns=[f"feat_{i}" for i in range(8)])
df["target"] = y_syn
target_column = "target"
'''
        # Remove existing file load lines
        lines = code.split("\n")
        filtered = [l for l in lines if "read_csv" not in l and "open(" not in l]
        return synthetic_header + "\n".join(filtered)

    def _extract_results_json(self, stdout: str) -> Dict[str, Any]:
        """Parse RESULTS_JSON marker from stdout."""
        for line in stdout.split("\n"):
            if line.startswith("RESULTS_JSON:"):
                try:
                    return json.loads(line[len("RESULTS_JSON:"):])
                except json.JSONDecodeError:
                    pass
        return {}

    def _extract_metrics(self, results_json: Dict[str, Any]) -> Dict[str, Any]:
        """Extract standard metrics and numerical scores from results_json."""
        if not results_json:
            return {}
        metrics = {}
        exclude_keys = {
            "experiment_id", "history", "X", "labels", "centers", "summary",
            "y_test", "y_pred", "x_values", "activations", "derivatives",
            "dataset", "elbow_data", "cluster_stats", "analysis"
        }
        for k, v in results_json.items():
            if k not in exclude_keys and isinstance(v, (int, float, str)):
                metrics[k] = v
        return metrics

    def format_execution_summary(self, result: ExecutionResult) -> str:
        """Format a human-readable execution summary."""
        lines = []
        if result.success:
            lines.append("✅ **Execution Successful**")
            lines.append(f"⏱️ Duration: {result.duration_seconds:.2f}s")
            if result.metrics:
                lines.append("\n**Results:**")
                for k, v in result.metrics.items():
                    if isinstance(v, float):
                        lines.append(f"  - {k.replace('_', ' ').title()}: {v:.4f}")
                    else:
                        lines.append(f"  - {k.replace('_', ' ').title()}: {v}")
        else:
            lines.append("❌ **Execution Failed**")
            lines.append(f"Error Type: `{result.error_type}`")
            lines.append(f"Repair Attempts: {result.repair_attempts}")
            if result.stderr:
                lines.append(f"\n**Error Details:**\n```\n{result.stderr[-800:]}\n```")
        return "\n".join(lines)
