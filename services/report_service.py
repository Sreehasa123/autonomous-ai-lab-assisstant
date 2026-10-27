"""Report Generation Service for Markdown, HTML, and JSON reports."""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from config.settings import settings
from database.repository import ExperimentRepository

logger = logging.getLogger(__name__)

class ReportService:
    """Generates comprehensive experiment laboratory reports."""

    def __init__(self, repo: Optional[ExperimentRepository] = None):
        self.repo = repo or ExperimentRepository()
        settings.ensure_directories()

    def generate_report(
        self,
        experiment_id: str,
        report_format: str = "markdown"  # markdown, html, json
    ) -> Path:
        """Assembles verified stored experiment data and compiles a formal report."""
        exp = self.repo.get_experiment(experiment_id)
        if not exp:
            raise ValueError(f"Experiment {experiment_id} not found")

        # Load stored JSON structures
        prob_understanding = json.loads(exp.problem_understanding_json or "{}")
        plan = json.loads(exp.experiment_plan_json or "{}")
        dataset_meta = json.loads(exp.selected_dataset_json or "{}")
        dataset_analysis = json.loads(exp.dataset_analysis_json or "{}")
        preprocessing_config = json.loads(exp.preprocessing_config_json or "{}")
        model_config = json.loads(exp.model_config_json or "{}")

        # Get latest run
        runs = sorted(exp.runs, key=lambda r: r.run_number)
        latest_run = runs[-1] if runs else None
        metrics_data = json.loads(latest_run.metrics_json or "{}") if latest_run else {}
        error_analysis = json.loads(latest_run.error_analysis_json or "{}") if latest_run else {}

        timestamp_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        safe_name = "".join(c for c in exp.name if c.isalnum() or c in (" ", "_", "-")).rstrip()

        if report_format == "json":
            payload = {
                "metadata": {
                    "experiment_id": exp.id,
                    "name": exp.name,
                    "created_at": str(exp.created_at),
                    "updated_at": str(exp.updated_at),
                    "version": exp.current_version,
                    "status": exp.status,
                    "generated_at": timestamp_str
                },
                "problem": {
                    "statement": exp.problem_statement,
                    "understanding": prob_understanding
                },
                "plan": plan,
                "dataset": {
                    "metadata": dataset_meta,
                    "analysis": dataset_analysis
                },
                "preprocessing": preprocessing_config,
                "model": model_config,
                "evaluation": {
                    "latest_run": latest_run.run_number if latest_run else None,
                    "duration_seconds": latest_run.duration_seconds if latest_run else 0.0,
                    "metrics": metrics_data,
                    "error_analysis": error_analysis
                }
            }
            target_path = settings.reports_dir / f"{safe_name.replace(' ', '_')}_{exp.id[:8]}.json"
            with open(target_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=4)

        elif report_format == "html":
            html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>AI Lab Report - {exp.name}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; color: #1e293b; max-width: 900px; margin: 40px auto; padding: 0 20px; }}
        h1, h2, h3 {{ color: #0f172a; }}
        h1 {{ border-bottom: 2px solid #0284c7; padding-bottom: 8px; }}
        .badge {{ display: inline-block; padding: 4px 10px; border-radius: 9999px; font-size: 12px; font-weight: 600; background: #e0f2fe; color: #0369a1; }}
        .card {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; margin-bottom: 20px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 16px 0; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #e2e8f0; }}
        th {{ background: #f1f5f9; }}
    </style>
</head>
<body>
    <h1>🧪 Autonomous AI Lab Assistant: {exp.name}</h1>
    <p><span class="badge">Status: {exp.status.upper()}</span> &bull; <strong>Experiment ID:</strong> <code>{exp.id}</code> &bull; <strong>Version:</strong> v{exp.current_version}</p>
    <p><em>Generated on {timestamp_str}</em></p>

    <h2>1. Problem Statement & Formulation</h2>
    <div class="card">
        <p><strong>Problem:</strong> {exp.problem_statement}</p>
        <p><strong>Identified Task:</strong> {prob_understanding.get("problem_type", "Not specified")}</p>
        <p><strong>Objective:</strong> {prob_understanding.get("objective", "N/A")}</p>
    </div>

    <h2>2. Dataset Overview</h2>
    <div class="card">
        <p><strong>Selected Dataset:</strong> {dataset_meta.get("name", "N/A")} ({dataset_meta.get("source_provider", "local")})</p>
        <p><strong>Shape:</strong> {dataset_analysis.get("shape", ["Unknown", "Unknown"])[0]} samples, {dataset_analysis.get("shape", ["Unknown", "Unknown"])[1]} features</p>
        <p><strong>Target Variable:</strong> <code>{dataset_meta.get("target_column", "target")}</code></p>
    </div>

    <h2>3. Preprocessing & Architecture</h2>
    <div class="card">
        <p><strong>Model:</strong> {model_config.get("model_name", "RandomForestClassifier")} ({model_config.get("framework", "scikit-learn")})</p>
        <p><strong>Numeric Scaling:</strong> {preprocessing_config.get("scale_numeric", "standard")}</p>
        <p><strong>Missing Values:</strong> {preprocessing_config.get("handle_missing", "median_mode")}</p>
    </div>

    <h2>4. Unbiased Evaluation Results</h2>
    <table>
        <thead><tr><th>Metric</th><th>Measured Value</th></tr></thead>
        <tbody>
"""
            metrics_dict = metrics_data.get("metrics", {})
            if metrics_dict:
                for k, v in metrics_dict.items():
                    html_content += f"<tr><td><strong>{k.replace('_', ' ').title()}</strong></td><td>{v}</td></tr>"
            else:
                html_content += "<tr><td colspan='2'><em>No evaluation metrics available yet. Run training to generate results.</em></td></tr>"

            html_content += f"""
        </tbody>
    </table>

    <h2>5. Error Analysis & Diagnostics</h2>
    <div class="card">
        <p><strong>Overfitting Risk:</strong> {error_analysis.get("overfitting_assessment", "Not evaluated")}</p>
        <p><strong>Underfitting Risk:</strong> {error_analysis.get("underfitting_assessment", "Not evaluated")}</p>
    </div>
</body>
</html>
"""
            target_path = settings.reports_dir / f"{safe_name.replace(' ', '_')}_{exp.id[:8]}.html"
            with open(target_path, "w", encoding="utf-8") as f:
                f.write(html_content)

        else:
            # Markdown Report
            lines = [
                f"# 🧪 AI Laboratory Experiment Report: {exp.name}\n",
                f"**Experiment ID:** `{exp.id}`  ",
                f"**Status:** `{exp.status}`  ",
                f"**Current Version:** `v{exp.current_version}`  ",
                f"**Report Date:** {timestamp_str}\n",
                "---\n",
                "## 1. Problem Formulation\n",
                f"**Problem Statement:**\n> {exp.problem_statement}\n",
                f"- **Task Category:** {prob_understanding.get('problem_type', 'N/A')}",
                f"- **Objective:** {prob_understanding.get('objective', 'N/A')}",
                f"- **Expected Input:** {prob_understanding.get('expected_input', 'N/A')}",
                f"- **Expected Output:** {prob_understanding.get('expected_output', 'N/A')}\n",
                "## 2. Dataset Metadata & Quality Audit\n",
                f"- **Dataset Name:** {dataset_meta.get('name', 'N/A')}",
                f"- **Source Provider:** {dataset_meta.get('source_provider', 'N/A')}",
                f"- **License:** {dataset_meta.get('license', 'Unknown')}",
                f"- **Target Column:** `{dataset_meta.get('target_column', 'target')}`",
                f"- **Samples & Dimensions:** {dataset_analysis.get('shape', ['N/A', 'N/A'])}\n",
                "### Data Quality Findings:\n"
            ]
            findings = dataset_analysis.get("findings", [])
            if findings:
                for f in findings:
                    lines.append(f"- **[{f.get('severity', 'info').upper()}]** {f.get('description', '')} *(Suggested Action: {f.get('suggested_action', '')})*")
            else:
                lines.append("- *No critical data quality issues identified.*")

            lines.extend([
                "\n## 3. Preprocessing & Model Selection\n",
                f"- **Model Architecture:** `{model_config.get('model_name', 'RandomForestClassifier')}` ({model_config.get('framework', 'scikit-learn')})",
                f"- **Hyperparameters:** `{model_config.get('hyperparameters', {})}`",
                f"- **Feature Scaling:** `{preprocessing_config.get('scale_numeric', 'standard')}`",
                f"- **Imputation Strategy:** `{preprocessing_config.get('handle_missing', 'median_mode')}`",
                f"- **Test Split Ratio:** `{preprocessing_config.get('test_size', 0.2)}`\n",
                "## 4. Empirical Evaluation Results\n"
            ])

            metrics_dict = metrics_data.get("metrics", {})
            if metrics_dict:
                lines.append("| Metric | Measured Test Value |")
                lines.append("| :--- | :--- |")
                for k, v in metrics_dict.items():
                    lines.append(f"| **{k.replace('_', ' ').title()}** | `{v}` |")
            else:
                lines.append("*No training runs completed yet for this experiment.*")

            lines.extend([
                "\n## 5. Error Analysis & Next Steps\n",
                f"- **Overfitting Assessment:** {error_analysis.get('overfitting_assessment', 'N/A')}",
                f"- **Underfitting Assessment:** {error_analysis.get('underfitting_assessment', 'N/A')}\n",
                "### Recommendations for Next Iteration:\n"
            ])
            recs = error_analysis.get("recommendations", [])
            if recs:
                for r in recs:
                    lines.append(f"- **{r.get('title', 'Recommendation')}:** {r.get('action', '')} *(Estimated Impact: {r.get('estimated_impact', 'Moderate')})*")
            else:
                lines.append("- *No specific error analysis recommendations available.*")

            lines.append("\n---\n*Report generated by Autonomous AI Lab Assistant.*")
            target_path = settings.reports_dir / f"{safe_name.replace(' ', '_')}_{exp.id[:8]}.md"
            with open(target_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

        # Register report record in database
        self.repo.add_report(
            experiment_id=exp.id,
            title=f"Lab Report: {exp.name} ({report_format.upper()})",
            report_type=report_format,
            file_path=str(target_path)
        )
        return target_path
