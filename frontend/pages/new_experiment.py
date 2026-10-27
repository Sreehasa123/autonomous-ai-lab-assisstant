"""New Experiment Workflow Page."""
import json
import streamlit as st
from database.repository import ExperimentRepository
from orchestration.workflow_orchestrator import WorkflowOrchestrator
from services.dataset_service import DatasetService
from frontend.components import render_section_header

def render_new_experiment(repo: ExperimentRepository, orchestrator: WorkflowOrchestrator, on_navigate):
    """Guided workflow for starting new experiments with Problem Understanding & Planning."""
    render_section_header("Start a New Experiment", "Describe your problem in plain natural language. The autonomous agents will analyze requirements, create an experiment plan, and guide you through dataset discovery and model training.")

    # Preset templates for convenience
    presets = {
        "Custom Problem": "",
        "Medical Diagnostics": "I want to build a deep learning model that identifies whether breast cancer tumors are malignant or benign from digitized fine needle aspirate biopsy features.",
        "Real Estate Valuation": "I want to predict continuous median house prices across districts using geographic, demographic, and housing attributes.",
        "Agricultural Plant Health": "I want to build a machine learning model that identifies plant and crop diseases from physical soil and leaf characteristics.",
        "Customer Churn / Sentiment": "I want to classify customer reviews into positive and negative sentiment categories to alert customer support."
    }

    selected_preset = st.selectbox("Quick-fill with an example scenario (optional):", list(presets.keys()))
    default_text = presets[selected_preset]

    with st.form("new_experiment_form"):
        st.markdown("#### Step 1: Experiment Definition")
        exp_name = st.text_input("Experiment Title / Name *", value="Cancer Diagnostic Classification" if "cancer" in default_text.lower() else ("Housing Price Forecasting" if "price" in default_text.lower() else "Autonomous ML Experiment"))
        problem_statement = st.text_area(
            "Problem Statement (Natural Language) *",
            value=default_text or "I want to build a classification model to categorize plant leaf health into discrete categories.",
            height=120,
            help="Describe your objective, what data you expect to have, and what the model should predict."
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            ml_dl_pref = st.selectbox("Methodology Preference", ["Auto (Agent Selects)", "Machine Learning (Tabular / Ensembles)", "Deep Learning (Neural Networks)"])
        with col2:
            framework = st.selectbox("Preferred Framework", ["scikit-learn", "PyTorch", "TensorFlow"])
        with col3:
            compute_budget = st.selectbox("Target Compute Budget", ["Local CPU (Fast, Free)", "Colab GPU (Free / T4)", "High Performance Cluster"])

        st.markdown("#### Optional Custom Dataset Upload")
        uploaded_file = st.file_uploader("Upload local CSV dataset (optional, up to 200MB):", type=["csv"])

        submitted = st.form_submit_button("🤖 Analyze Problem & Generate Plan", type="primary", use_container_width=True)

    if submitted:
        if not exp_name.strip() or not problem_statement.strip():
            st.error("Please provide both an Experiment Name and a Problem Statement.")
            return

        with st.spinner("Problem Understanding & Planning Agents are analyzing your requirements..."):
            # 1. Create experiment record in DB
            exp = repo.create_experiment(name=exp_name.strip(), problem_statement=problem_statement.strip())
            st.session_state["current_experiment_id"] = exp.id

            # 2. Handle optional dataset upload
            if uploaded_file is not None:
                ds_service = DatasetService()
                upload_meta = ds_service.save_uploaded_dataset(uploaded_file.name, uploaded_file.getvalue())
                repo.update_experiment(exp.id, selected_dataset_json=json.dumps(upload_meta.dict()))

            # 3. Process Problem Formulation
            orchestrator.process_problem_formulation(exp.id)
            st.success(f"Experiment '{exp_name}' created successfully (ID: {exp.id[:8]}).")
            st.rerun()

    # If an active experiment is currently loaded in creation phase, display the interactive review
    curr_id = st.session_state.get("current_experiment_id")
    if curr_id:
        exp = repo.get_experiment(curr_id)
        if exp and exp.problem_understanding_json and exp.problem_understanding_json != "{}":
            st.markdown("---")
            st.markdown("### Step 2: Agent Problem Understanding & Proposed Plan")
            
            prob_dict = json.loads(exp.problem_understanding_json)
            plan_dict = json.loads(exp.experiment_plan_json)

            # Display Problem Understanding Card
            with st.expander("🔍 Problem Understanding Interpretation (Review & Edit)", expanded=True):
                col_u1, col_u2 = st.columns(2)
                with col_u1:
                    st.markdown(f"**Identified Task Type:** `{prob_dict.get('problem_type')}`")
                    st.markdown(f"**Primary Objective:** {prob_dict.get('objective')}")
                    st.markdown(f"**Expected Inputs:** {prob_dict.get('expected_input')}")
                with col_u2:
                    st.markdown(f"**Expected Output:** {prob_dict.get('expected_output')}")
                    st.markdown(f"**Candidate Evaluation Metrics:** {', '.join(prob_dict.get('candidate_metrics', []))}")
                    st.markdown(f"**Recommended Model Families:** {', '.join(prob_dict.get('possible_model_families', []))}")

                st.markdown("**Potential Risks Identified:**")
                for r in prob_dict.get("potential_risks", []):
                    st.markdown(f"- ⚠️ {r}")

            # Display Experiment Plan Card
            with st.expander("📋 Strategic Experiment Plan", expanded=True):
                st.markdown("**Core Objectives:**")
                for obj in plan_dict.get("objectives", []):
                    st.markdown(f"- 🎯 {obj}")

                st.markdown(f"**Evaluation Methodology:** {plan_dict.get('evaluation_methodology', 'N/A')}")
                st.markdown(f"**Resource Considerations:** {plan_dict.get('resource_considerations', 'N/A')}")

            # Action to proceed to Dataset Explorer
            c_left, c_right = st.columns([1, 1])
            with c_left:
                if st.button("🔎 Proceed to Dataset Discovery & Selection", type="primary", use_container_width=True):
                    on_navigate("Dataset Explorer")
            with c_right:
                if st.button("🛠️ Open Experiment Workspace", use_container_width=True):
                    on_navigate("Experiment Workspace")
