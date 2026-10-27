"""Experiment Comparison Page."""
import json
import streamlit as st
import pandas as pd
from database.repository import ExperimentRepository
from services.visualization_service import VisualizationService
from frontend.components import render_section_header, render_empty_state

def render_experiment_comparison(repo: ExperimentRepository, on_navigate):
    """Side-by-side empirical performance and parameter comparison across runs and experiments."""
    render_section_header("Experiment & Run Comparison", "Compare configurations, metrics, and training run durations side-by-side.")

    experiments = repo.list_experiments(limit=100)
    if not experiments:
        render_empty_state("No Experiments Available", "Create experiments and execute training runs to enable comparative analysis.")
        return

    exp_map = {f"{e.name} (ID: {e.id[:8]}, v{e.current_version})": e.id for e in experiments}
    selected_exp_names = st.multiselect("Select experiments to compare:", list(exp_map.keys()), default=list(exp_map.keys())[:2] if len(exp_map) >= 2 else list(exp_map.keys()))

    if not selected_exp_names:
        st.info("Select at least one experiment above to view comparative metrics.")
        return

    runs_data = []
    comparison_table = []

    for name in selected_exp_names:
        exp_id = exp_map[name]
        exp = repo.get_experiment(exp_id)
        if not exp:
            continue

        dataset_meta = json.loads(exp.selected_dataset_json or "{}")
        prep_config = json.loads(exp.preprocessing_config_json or "{}")
        model_config = json.loads(exp.model_config_json or "{}")

        runs = sorted(exp.runs, key=lambda r: r.run_number)
        if not runs:
            comparison_table.append({
                "Experiment": exp.name,
                "Run #": "No runs",
                "Model": model_config.get("model_name", "N/A"),
                "Dataset": dataset_meta.get("name", "N/A"),
                "Task": dataset_meta.get("task_type", "N/A"),
                "Primary Metric": "N/A",
                "Score": 0.0,
                "Duration (s)": 0.0,
                "Status": exp.status
            })
        else:
            for r in runs:
                m_payload = json.loads(r.metrics_json or "{}")
                p_name = m_payload.get("primary_metric_name", "Score")
                p_val = m_payload.get("primary_metric_value", 0.0)

                runs_data.append({
                    "experiment_name": exp.name,
                    "run_number": r.run_number,
                    "version_number": r.version_number,
                    "metrics": m_payload.get("metrics", {p_name: p_val}),
                    "primary_name": p_name,
                    "primary_val": p_val
                })

                comparison_table.append({
                    "Experiment": exp.name,
                    "Run #": f"Run #{r.run_number} (v{r.version_number})",
                    "Model": model_config.get("model_name", "N/A"),
                    "Dataset": dataset_meta.get("name", "N/A"),
                    "Task": dataset_meta.get("task_type", "N/A"),
                    "Primary Metric": p_name,
                    "Score": p_val,
                    "Duration (s)": round(r.duration_seconds, 2),
                    "Status": r.status
                })

    # Render Table
    st.markdown("#### 📊 Comparative Parameters & Performance Metrics")
    df_cmp = pd.DataFrame(comparison_table)
    st.dataframe(df_cmp, use_container_width=True)

    # Comparison Chart
    if runs_data:
        st.markdown("#### 📈 Benchmark Chart")
        fig = VisualizationService.plot_run_comparison(runs_data)
        st.plotly_chart(fig, use_container_width=True)

        # Sanity warning if task types differ
        tasks = df_cmp["Task"].unique()
        if len(tasks) > 1:
            st.warning("⚠️ Note: You are comparing runs across differing task types (e.g. classification vs regression). Metrics such as Accuracy and R² are not directly comparable on the same mathematical scale.")
