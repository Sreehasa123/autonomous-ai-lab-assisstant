"""Dataset Explorer Page."""
import json
import streamlit as st
import pandas as pd
from database.repository import ExperimentRepository
from orchestration.workflow_orchestrator import WorkflowOrchestrator
from frontend.components import render_section_header, render_empty_state

def render_dataset_explorer(repo: ExperimentRepository, orchestrator: WorkflowOrchestrator, on_navigate):
    """Multi-source dataset search, ranking, comparison, and selection."""
    render_section_header("Multi-Source Dataset Explorer", "Search, compare, and select verified benchmark datasets from UCI, OpenML, Hugging Face, Kaggle, or uploaded files.")

    curr_exp_id = st.session_state.get("current_experiment_id")
    exp = repo.get_experiment(curr_exp_id) if curr_exp_id else None

    # Search bar & provider filters
    default_q = ""
    if exp:
        default_q = exp.problem_statement.split(".")[0]

    col_s1, col_s2 = st.columns([3, 1])
    with col_s1:
        search_query = st.text_input("Enter search keywords or domain (e.g., 'breast cancer', 'housing', 'wine', 'iris'):", value=default_q)
    with col_s2:
        search_btn = st.button("🔍 Search Datasets", type="primary", use_container_width=True)

    # Provider filters
    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        filter_uci = st.checkbox("UCI ML Repository", value=True)
    with f_col2:
        filter_openml = st.checkbox("OpenML Benchmarks", value=True)
    with f_col3:
        filter_hf = st.checkbox("Hugging Face Datasets", value=True)

    # Cache search results in session state
    if search_btn or "cached_search_results" not in st.session_state:
        if curr_exp_id:
            with st.spinner("Searching configured repositories..."):
                results = orchestrator.search_and_rank_datasets(curr_exp_id, search_query=search_query)
                st.session_state["cached_search_results"] = results
        else:
            from services.dataset_service import DatasetService
            ds_service = DatasetService()
            search_out = ds_service.search_all_providers(search_query or "wine")
            from agents.dataset_comparison_agent import DatasetComparisonAgent
            agent = DatasetComparisonAgent()
            ranked = agent.run({"datasets": [d.dict() for d in search_out["datasets"]]})
            st.session_state["cached_search_results"] = {
                "datasets": ranked.get("ranked_datasets", []),
                "providers": search_out["providers"],
                "total_found": search_out["total_found"]
            }

    search_data = st.session_state.get("cached_search_results", {})
    datasets = search_data.get("datasets", [])
    providers = search_data.get("providers", {})

    # Provider status indicators
    with st.expander("📡 Dataset Repository Connectivity Status", expanded=False):
        for p_name, p_info in providers.items():
            status_text = p_info.get("status", "unknown").upper()
            if status_text == "AVAILABLE":
                st.markdown(f"✅ **{p_name.upper()}**: Online & Accessible ({p_info.get('count', 0)} matching datasets)")
            elif status_text == "UNCONFIGURED":
                st.markdown(f"ℹ️ **{p_name.upper()}**: Unconfigured - {p_info.get('message', '')}")
            else:
                st.markdown(f"⚠️ **{p_name.upper()}**: Offline or timed out - {p_info.get('error', '')}")

    if not datasets:
        render_empty_state("No Matching Datasets", "No datasets found for the query. Try broader keywords like 'tabular', 'iris', 'wine', 'housing'.")
        return

    # Filter according to checkboxes
    filtered_datasets = []
    for d in datasets:
        provider = d.get("source_provider", "")
        if provider == "uci" and not filter_uci:
            continue
        if provider == "openml" and not filter_openml:
            continue
        if provider == "huggingface" and not filter_hf:
            continue
        filtered_datasets.append(d)

    st.markdown(f"#### Available Datasets ({len(filtered_datasets)} matches)")

    # Comparison Selector
    st.markdown("##### ⚖️ Dataset Comparison Tool")
    ds_names = [d.get("name") for d in filtered_datasets]
    compare_selection = st.multiselect("Select at least 2 datasets to compare side-by-side:", ds_names, default=ds_names[:2] if len(ds_names) >= 2 else ds_names)

    if len(compare_selection) >= 2:
        compare_records = [d for d in filtered_datasets if d.get("name") in compare_selection]
        cmp_df = pd.DataFrame([
            {
                "Dataset Name": c.get("name"),
                "Provider": c.get("source_provider"),
                "Suitability Score": f"{c.get('suitability_score', 0)}/100",
                "Task Type": c.get("task_type"),
                "Samples (Rows)": c.get("num_rows", "Unknown"),
                "Features": c.get("num_features", "Unknown"),
                "Format": c.get("data_format"),
                "License": c.get("license", "Unknown"),
                "Why Recommended": c.get("suitability_reason", "Candidate dataset")
            }
            for c in compare_records
        ])
        st.dataframe(cmp_df, use_container_width=True)

    st.markdown("---")
    st.markdown("##### Detailed Dataset Cards & Selection")

    for idx, ds in enumerate(filtered_datasets):
        with st.container():
            c_header, c_score = st.columns([4, 1])
            with c_header:
                st.markdown(f"### {idx+1}. {ds.get('name')}")
                st.markdown(f"<span style='color: #64748B;'>Provider: <strong>{ds.get('source_provider', '').upper()}</strong> | Format: <code>{ds.get('data_format')}</code> | License: <code>{ds.get('license', 'Unknown')}</code></span>", unsafe_allow_html=True)
            with c_score:
                suit_score = ds.get("suitability_score", 50.0)
                st.metric("Suitability", f"{suit_score}/100")

            st.markdown(f"> {ds.get('description')}")
            st.markdown(f"**Recommendation Rationale:** {ds.get('suitability_reason', 'Compatible benchmark dataset.')}")

            # Details & Actions
            c_det1, c_det2, c_btn = st.columns([2, 2, 2])
            with c_det1:
                st.markdown(f"- **Instances (Rows):** `{ds.get('num_rows', 'Unknown')}`")
                st.markdown(f"- **Features:** `{ds.get('num_features', 'Unknown')}`")
            with c_det2:
                st.markdown(f"- **Target Column:** `{ds.get('target_column', 'target')}`")
                if ds.get("url"):
                    st.markdown(f"- [View Source Repository]({ds.get('url')})")

            with c_btn:
                if curr_exp_id:
                    if st.button(f"✅ Select Dataset for Experiment", key=f"sel_ds_{ds.get('dataset_id')}_{idx}", type="primary", use_container_width=True):
                        with st.spinner("Analyzing dataset structure and auditing data quality..."):
                            orchestrator.select_and_analyze_dataset(curr_exp_id, ds)
                            st.success(f"Selected '{ds.get('name')}' for current experiment!")
                            on_navigate("Experiment Workspace")
                else:
                    st.info("Start or select an experiment to attach this dataset.")

            st.markdown("<hr style='margin: 16px 0; border: none; border-top: 1px solid #E2E8F0;'>", unsafe_allow_html=True)
