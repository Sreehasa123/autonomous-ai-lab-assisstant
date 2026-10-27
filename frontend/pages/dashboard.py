"""Dashboard Page."""
import json
import streamlit as st
import pandas as pd
from database.repository import ExperimentRepository
from frontend.components import render_metric_card, render_status_badge, render_section_header, render_empty_state

def render_dashboard(repo: ExperimentRepository, on_navigate):
    """Renders real statistics, recent activity, and quick actions from the database."""
    st.markdown("## 🧪 AI Laboratory Assistant Dashboard")
    st.markdown("<p style='color: #64748B; font-size: 15px;'>Welcome to your autonomous deep learning & ML experimentation workspace.</p>", unsafe_allow_html=True)
    st.markdown("---")

    # Fetch real counts from DB
    counts = repo.count_experiments_by_status()
    total = counts.get("total", 0)

    # 4 Key Metrics Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("Total Experiments", total, "All registered experiments")
    with c2:
        render_metric_card("Completed Runs", counts.get("completed", 0), "Successfully trained models")
    with c3:
        render_metric_card("In Progress / Draft", counts.get("draft", 0) + counts.get("running", 0), "Active workspaces")
    with c4:
        render_metric_card("Awaiting Action", counts.get("awaiting_approval", 0), "Ready for review & execution")

    st.write("")

    if total == 0:
        created = render_empty_state(
            title="No Experiments Found",
            message="Your AI Laboratory workspace is currently empty. Start by describing your machine learning or deep learning problem in natural language.",
            button_label="➕ Create First Experiment",
            button_key="dash_new_exp_btn"
        )
        if created:
            on_navigate("New Experiment")
        return

    # Recent Experiments Table
    render_section_header("Recent Experiment History", "Latest experiments loaded permanently from SQLite database.")
    recent_exps = repo.list_experiments(limit=6)

    table_data = []
    for exp in recent_exps:
        # Check latest run metrics
        last_metric = "N/A"
        if exp.runs:
            latest_run = exp.runs[-1]
            if latest_run.metrics_json:
                try:
                    m = json.loads(latest_run.metrics_json)
                    primary_name = m.get("primary_metric_name", "Score")
                    primary_val = m.get("primary_metric_value", 0.0)
                    last_metric = f"{primary_name}: {primary_val}"
                except Exception:
                    pass

        # Parse selected dataset
        ds_name = "None selected"
        if exp.selected_dataset_json:
            try:
                ds = json.loads(exp.selected_dataset_json)
                ds_name = ds.get("name", "Custom")
            except Exception:
                pass

        table_data.append({
            "Experiment Name": exp.name,
            "Status": exp.status,
            "Version": f"v{exp.current_version}",
            "Dataset": ds_name,
            "Latest Metric": last_metric,
            "Last Modified": exp.updated_at.strftime("%Y-%m-%d %H:%M"),
            "_id": exp.id
        })

    df_display = pd.DataFrame(table_data)
    
    # Render table with action buttons
    for idx, row in df_display.iterrows():
        col1, col2, col3, col4, col5 = st.columns([3, 1.5, 2, 2, 1.5])
        with col1:
            st.markdown(f"**{row['Experiment Name']}** <br><span style='color: #64748B; font-size: 12px;'>ID: {row['_id'][:8]} | {row['Version']}</span>", unsafe_allow_html=True)
        with col2:
            st.markdown(render_status_badge(row["Status"]), unsafe_allow_html=True)
        with col3:
            st.markdown(f"📊 {row['Dataset']}")
        with col4:
            st.markdown(f"🎯 `{row['Latest Metric']}`")
        with col5:
            if st.button("Open", key=f"dash_open_{row['_id']}", use_container_width=True):
                st.session_state["current_experiment_id"] = row["_id"]
                on_navigate("Experiment Workspace")
        st.markdown("<hr style='margin: 8px 0; border: none; border-top: 1px solid #F1F5F9;'>", unsafe_allow_html=True)

    st.write("")
    # Quick action bar
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("➕ Start New Experiment", use_container_width=True, type="primary"):
            on_navigate("New Experiment")
    with col_b:
        if st.button("📂 Explore All Previous Experiments", use_container_width=True):
            on_navigate("Previous Experiments")
