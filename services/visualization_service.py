"""Visualization Service using Plotly for Interactive Charts."""
from typing import Dict, Any, List, Optional
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

class VisualizationService:
    """Builds interactive charts for results, datasets, and experiment comparisons."""

    @staticmethod
    def plot_confusion_matrix(matrix: List[List[int]], labels: Optional[List[str]] = None) -> go.Figure:
        """Interactive confusion matrix heatmap."""
        labels = labels or [f"Class {i}" for i in range(len(matrix))]
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=[f"Pred: {l}" for l in labels],
            y=[f"True: {l}" for l in labels],
            colorscale="Teal",
            text=matrix,
            texttemplate="%{text}",
            textfont={"size": 14},
            colorbar=dict(title="Count")
        ))
        fig.update_layout(
            title="Confusion Matrix (Test Evaluation)",
            xaxis_title="Predicted Class",
            yaxis_title="Ground Truth Class",
            template="plotly_white",
            margin=dict(l=40, r=40, t=50, b=40),
            height=400
        )
        return fig

    @staticmethod
    def plot_feature_importance(feature_importances: Dict[str, float], top_n: int = 15) -> go.Figure:
        """Horizontal bar chart for feature importances."""
        sorted_feats = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)[:top_n]
        features = [f[0] for f in sorted_feats][::-1]
        values = [f[1] for f in sorted_feats][::-1]

        fig = go.Figure(go.Bar(
            x=values,
            y=features,
            orientation="h",
            marker=dict(color="#1E88E5", line=dict(color="#0D47A1", width=1))
        ))
        fig.update_layout(
            title=f"Top {len(features)} Predictive Features",
            xaxis_title="Relative Importance Score",
            yaxis_title="Feature Name",
            template="plotly_white",
            margin=dict(l=40, r=40, t=50, b=40),
            height=max(350, len(features) * 25)
        )
        return fig

    @staticmethod
    def plot_actual_vs_predicted(actual_vs_pred: List[Dict[str, float]]) -> go.Figure:
        """Scatter plot of actual vs predicted targets for regression."""
        actuals = [d["actual"] for d in actual_vs_pred]
        preds = [d["predicted"] for d in actual_vs_pred]

        min_val = min(min(actuals), min(preds))
        max_val = max(max(actuals), max(preds))

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=actuals,
            y=preds,
            mode="markers",
            name="Predictions",
            marker=dict(color="#00ACC1", size=8, opacity=0.7)
        ))
        # Ideal 45-degree line
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            name="Ideal Identity (y = x)",
            line=dict(color="#E53935", dash="dash", width=2)
        ))
        fig.update_layout(
            title="Actual vs Predicted Values",
            xaxis_title="Actual Ground Truth",
            yaxis_title="Model Prediction",
            template="plotly_white",
            margin=dict(l=40, r=40, t=50, b=40),
            height=400
        )
        return fig

    @staticmethod
    def plot_residuals(actual_vs_pred: List[Dict[str, float]]) -> go.Figure:
        """Residual plot: actuals vs error (predicted - actual)."""
        actuals = [d["actual"] for d in actual_vs_pred]
        residuals = [d["predicted"] - d["actual"] for d in actual_vs_pred]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=actuals,
            y=residuals,
            mode="markers",
            marker=dict(color="#5E35B1", size=7, opacity=0.7)
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        fig.update_layout(
            title="Residual Error Analysis",
            xaxis_title="Actual Value",
            yaxis_title="Residual (Pred - Actual)",
            template="plotly_white",
            margin=dict(l=40, r=40, t=50, b=40),
            height=350
        )
        return fig

    @staticmethod
    def plot_target_distribution(target_counts: Dict[str, int]) -> go.Figure:
        """Bar chart showing target class distribution."""
        labels = list(target_counts.keys())
        counts = list(target_counts.values())

        fig = go.Figure(go.Bar(
            x=labels,
            y=counts,
            marker=dict(color="#26A69A")
        ))
        fig.update_layout(
            title="Target Class Distribution",
            xaxis_title="Class Label",
            yaxis_title="Frequency",
            template="plotly_white",
            margin=dict(l=40, r=40, t=50, b=40),
            height=320
        )
        return fig

    @staticmethod
    def plot_run_comparison(runs_data: List[Dict[str, Any]]) -> go.Figure:
        """Bar chart comparing primary metrics across experiment runs."""
        run_labels = [f"Run #{r.get('run_number', i+1)} (v{r.get('version_number', 1)})" for i, r in enumerate(runs_data)]
        metric_values = []
        metric_name = "Primary Metric"

        for r in runs_data:
            metrics = r.get("metrics", {})
            if "accuracy" in metrics:
                metric_values.append(metrics["accuracy"])
                metric_name = "Accuracy"
            elif "r2_score" in metrics:
                metric_values.append(metrics["r2_score"])
                metric_name = "R² Score"
            else:
                first_val = list(metrics.values())[0] if metrics else 0.0
                metric_values.append(first_val)

        fig = go.Figure(go.Bar(
            x=run_labels,
            y=metric_values,
            text=[f"{v:.4f}" for v in metric_values],
            textposition="auto",
            marker=dict(color="#1976D2")
        ))
        fig.update_layout(
            title=f"Run Comparison: {metric_name}",
            xaxis_title="Run / Version",
            yaxis_title=metric_name,
            template="plotly_white",
            margin=dict(l=40, r=40, t=50, b=40),
            height=380
        )
        return fig
