"""Previous Experiments Workspace Page."""
import json
import streamlit as st
import pandas as pd

from database.repository import ExperimentRepository
from services.report_service import ReportService
from services.notebook_service import NotebookService
from frontend.components import render_section_header, render_status_badge, render_empty_state

def render_previous_experiments(repo: ExperimentRepository, on_navigate):
    """Browses, filters, searches, inspects, duplicates, and manages all stored experiments."""
    render_section_header("Previous Experiments Archive", "All experiments are permanently persisted in the SQLite database and remain accessible across restarts.")

    # Search & Filter Controls
    c_s1, c_s2, c_s3 = st.columns([2, 1.5, 1.5])
    with c_s1:
        search_query = st.text_input("🔍 Search by title or problem keywords:", "")
    with c_s2:
        status_filter = st.selectbox("Filter by Status:", ["all", "completed", "draft", "running", "failed", "awaiting_approval"])
    with c_s3:
        sort_by = st.selectbox("Sort Order:", ["Newest First", "Oldest First", "Name (A-Z)"])

    experiments = repo.list_experiments(
        status=status_filter if status_filter != "all" else None,
        search_query=search_query if search_query.strip() else None,
        limit=100
    )

    if sort_by == "Oldest First":
        experiments = sorted(experiments, key=lambda x: x.created_at)
    elif sort_by == "Name (A-Z)":
        experiments = sorted(experiments, key=lambda x: x.name.lower())

    if not experiments:
        created = render_empty_state(
            title="No Matching Experiments Found",
            message="No experiments match your search criteria. Create a new experiment to begin.",
            button_label="➕ Create New Experiment",
            button_key="prev_new_btn"
        )
        if created:
            on_navigate("New Experiment")
        return

    st.markdown(f"**Found {len(experiments)} experiment records in persistent storage:**")

    for exp in experiments:
        with st.container():
            c_main, c_stat, c_act = st.columns([3, 1.5, 2.5])
            with c_main:
                st.markdown(f"#### 🧪 {exp.name}")
                st.markdown(f"<span style='color: #64748B; font-size: 13px;'>ID: <code>{exp.id[:8]}</code> | Version: <strong>v{exp.current_version}</strong> | Modified: {exp.updated_at.strftime('%Y-%m-%d %H:%M')}</span>", unsafe_allow_html=True)
                st.markdown(f"> *{exp.problem_statement[:140]}...*" if len(exp.problem_statement) > 140 else f"> *{exp.problem_statement}*")

            with c_stat:
                st.markdown(render_status_badge(exp.status), unsafe_allow_html=True)
                
                # Show latest metric
                runs = sorted(exp.runs, key=lambda r: r.run_number)
                if runs:
                    latest = runs[-1]
                    m_dict = json.loads(latest.metrics_json or "{}")
                    p_name = m_dict.get("primary_metric_name", "Metric")
                    p_val = m_dict.get("primary_metric_value", 0.0)
                    st.markdown(f"**Latest Score:**<br>`{p_name}: {p_val}`", unsafe_allow_html=True)
                else:
                    st.markdown("<span style='color: #94A3B8;'>No runs yet</span>", unsafe_allow_html=True)

            with c_act:
                btn_c1, btn_c2 = st.columns(2)
                with btn_c1:
                    if st.button("📂 Open Workspace", key=f"open_exp_{exp.id}", use_container_width=True, type="primary"):
                        st.session_state["current_experiment_id"] = exp.id
                        on_navigate("Experiment Workspace")
                    if st.button("📋 Duplicate", key=f"dup_exp_{exp.id}", use_container_width=True):
                        dup = repo.duplicate_experiment(exp.id)
                        st.success(f"Duplicated as '{dup.name}'!")
                        st.rerun()

                with btn_c2:
                    if st.button("📄 Generate Report", key=f"rep_exp_{exp.id}", use_container_width=True):
                        rep_service = ReportService(repo=repo)
                        r_path = rep_service.generate_report(exp.id, report_format="markdown")
                        st.session_state["current_experiment_id"] = exp.id
                        st.success(f"Report compiled: {r_path.name}")
                        on_navigate("Reports")
                    if st.button("🗑️ Delete", key=f"del_exp_{exp.id}", use_container_width=True):
                        repo.delete_experiment(exp.id)
                        if st.session_state.get("current_experiment_id") == exp.id:
                            st.session_state["current_experiment_id"] = None
                        st.warning(f"Experiment '{exp.name}' deleted.")
                        st.rerun()

            st.markdown("<hr style='margin: 12px 0; border: none; border-top: 1px solid #E2E8F0;'>", unsafe_allow_html=True)
