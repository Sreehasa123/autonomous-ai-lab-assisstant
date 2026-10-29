"""Experiment Workspace Page."""
import json
import streamlit as st
import pandas as pd
from pathlib import Path

from database.repository import ExperimentRepository
from orchestration.workflow_orchestrator import WorkflowOrchestrator
from services.visualization_service import VisualizationService
from services.notebook_service import NotebookService
from frontend.components import render_section_header, render_status_badge, render_empty_state

def render_experiment_workspace(repo: ExperimentRepository, orchestrator: WorkflowOrchestrator, on_navigate):
    """Central interactive laboratory workspace for the active experiment."""
    curr_id = st.session_state.get("current_experiment_id")
    if not curr_id:
        created = render_empty_state(
            title="No Active Experiment Selected",
            message="Select an existing experiment from Previous Experiments or start a new experiment to begin your laboratory session.",
            button_label="➕ Start New Experiment",
            button_key="ws_new_exp_btn"
        )
        if created:
            on_navigate("New Experiment")
        return

    exp = repo.get_experiment(curr_id)
    if not exp:
        st.error(f"Experiment with ID `{curr_id}` was not found in the database.")
        return

    # Header & Meta Bar
    c_title, c_badge = st.columns([4, 1.5])
    with c_title:
        st.markdown(f"## 🔬 {exp.name}")
        st.markdown(f"<span style='color: #64748B; font-size: 13px;'>ID: <code>{exp.id}</code> | Version: <strong>v{exp.current_version}</strong> | Last Updated: {exp.updated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}</span>", unsafe_allow_html=True)
    with c_badge:
        st.markdown(f"<div style='text-align: right; padding-top: 10px;'>{render_status_badge(exp.status)}</div>", unsafe_allow_html=True)

    st.markdown("---")

    # Workflow Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📋 1. Formulation & Plan",
        "📊 2. Data & Quality Audit",
        "⚙️ 3. Preprocessing & Model",
        "💻 4. Code & Validation",
        "🚀 5. Execution & Colab",
        "📈 6. Results & Visualizations"
    ])

    # TAB 1: FORMULATION & PLAN
    with tab1:
        st.markdown("#### Problem Definition")
        st.info(exp.problem_statement)

        prob_data = json.loads(exp.problem_understanding_json or "{}")
        plan_data = json.loads(exp.experiment_plan_json or "{}")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Identified Task Type:** `{prob_data.get('problem_type', 'N/A')}`")
            st.markdown(f"**Primary Objective:** {prob_data.get('objective', 'N/A')}")
            st.markdown(f"**Expected Inputs:** {prob_data.get('expected_input', 'N/A')}")
        with col2:
            st.markdown(f"**Expected Outputs:** {prob_data.get('expected_output', 'N/A')}")
            st.markdown(f"**Target Evaluation Metrics:** {', '.join(prob_data.get('candidate_metrics', []))}")
            st.markdown(f"**Model Families:** {', '.join(prob_data.get('possible_model_families', []))}")

        if plan_data.get("objectives"):
            st.markdown("##### Strategic Experiment Plan")
            for obj in plan_data["objectives"]:
                st.markdown(f"- 🎯 {obj}")

    # TAB 2: DATA & QUALITY AUDIT
    with tab2:
        understanding = json.loads(exp.problem_understanding_json or "{}")
        if not understanding.get("dataset_required", True):
            st.info("⚡ **Dataset-Free Experiment**: This experiment (e.g. mathematical optimization, algorithm comparison, or synthetic demonstration) is self-contained and does not require an external dataset.")
        else:
            dataset_meta = json.loads(exp.selected_dataset_json or "{}")
            if not dataset_meta:
                st.warning("No dataset selected for this experiment yet.")
                if st.button("🔎 Go to Dataset Explorer to Select Dataset", type="primary"):
                    on_navigate("Dataset Explorer")
            else:
            st.markdown(f"#### Selected Dataset: {dataset_meta.get('name')}")
            st.markdown(f"<span style='color: #64748B;'>Provider: <strong>{dataset_meta.get('source_provider', '').upper()}</strong> | Format: <code>{dataset_meta.get('data_format')}</code></span>", unsafe_allow_html=True)
            
            analysis = json.loads(exp.dataset_analysis_json or "{}")
            shape = analysis.get("shape", [0, 0])
            st.markdown(f"- **Dimensions:** `{shape[0]} rows × {shape[1]} columns`")
            st.markdown(f"- **Target Column:** `{dataset_meta.get('target_column', 'target')}`")

            # Sample records
            samples = analysis.get("sample_records", [])
            if samples:
                st.markdown("##### Sample Records Preview (First 10 rows)")
                st.dataframe(pd.DataFrame(samples), use_container_width=True)

            # Data Quality Audit
            findings = analysis.get("findings", [])
            st.markdown("##### 🛡️ Data Quality & Leakage Audit Findings")
            if not findings:
                st.success("✅ No critical data leakage or quality risks detected in sample audit.")
            else:
                for f in findings:
                    sev = f.get("severity", "info").lower()
                    icon = "🚨" if sev in ("critical", "high_risk") else "⚠️"
                    color = "#EF4444" if sev in ("critical", "high_risk") else "#F59E0B"
                    st.markdown(f"""
                    <div style="border-left: 4px solid {color}; padding: 8px 12px; background: #F8FAFC; margin-bottom: 8px; border-radius: 4px;">
                        <strong>{icon} [{sev.upper()}] {f.get('description')}</strong><br>
                        <span style="font-size: 13px; color: #475569;">Impact: {f.get('impact')}</span><br>
                        <span style="font-size: 13px; color: #0284C7;">Action: {f.get('suggested_action')}</span>
                    </div>
                    """, unsafe_allow_html=True)

    # TAB 3: PREPROCESSING & MODEL CONFIGURATION
    with tab3:
        st.markdown("#### Configure Preprocessing & Model Selection")
        prep_config = json.loads(exp.preprocessing_config_json or "{}")
        model_config = json.loads(exp.model_config_json or "{}")

        with st.form("config_edit_form"):
            c_p1, c_p2 = st.columns(2)
            with c_p1:
                st.markdown("##### Preprocessing Settings")
                scale_sel = st.selectbox("Numerical Feature Scaling", ["standard", "minmax", "robust", "none"], index=["standard", "minmax", "robust", "none"].index(prep_config.get("scale_numeric", "standard")))
                impute_sel = st.selectbox("Missing Value Strategy", ["median_mode", "mean", "none"], index=["median_mode", "mean", "none"].index(prep_config.get("handle_missing", "median_mode")))
                test_size_sel = st.slider("Test Split Proportion", min_value=0.1, max_value=0.4, value=float(prep_config.get("test_size", 0.2)), step=0.05)
                random_state_sel = st.number_input("Reproducibility Random State (Seed)", value=int(prep_config.get("random_state", 42)))

            with c_p2:
                st.markdown("##### Model Selection")
                model_sel = st.selectbox(
                    "Model Architecture",
                    ["RandomForestClassifier", "GradientBoostingClassifier", "LogisticRegression", "MLPClassifier", "RandomForestRegressor", "GradientBoostingRegressor", "RidgeRegression", "MLPRegressor"],
                    index=0 if model_config.get("model_name") not in ["RandomForestClassifier", "GradientBoostingClassifier", "LogisticRegression", "MLPClassifier", "RandomForestRegressor", "GradientBoostingRegressor", "RidgeRegression", "MLPRegressor"] else ["RandomForestClassifier", "GradientBoostingClassifier", "LogisticRegression", "MLPClassifier", "RandomForestRegressor", "GradientBoostingRegressor", "RidgeRegression", "MLPRegressor"].index(model_config.get("model_name"))
                )
                st.markdown(f"**Explanation:** {model_config.get('explanation', 'Configured model architecture.')}")

            save_configs = st.form_submit_button("💾 Save Configurations & Regenerate Code", type="primary", use_container_width=True)

        if save_configs:
            prep_config["scale_numeric"] = scale_sel
            prep_config["handle_missing"] = impute_sel
            prep_config["test_size"] = test_size_sel
            prep_config["random_state"] = int(random_state_sel)
            model_config["model_name"] = model_sel
            
            repo.update_experiment(
                exp.id,
                preprocessing_config_json=json.dumps(prep_config),
                model_config_json=json.dumps(model_config)
            )
            # Re-generate code
            orchestrator.generate_and_validate_code(exp.id)
            st.success("Configuration updated and verified code regenerated successfully!")
            st.rerun()

    # TAB 4: CODE & VALIDATION
    with tab4:
        st.markdown("#### Generated Python Experiment Code")
        val_data = json.loads(exp.last_validation_json or "{}")

        # Validation status banner
        if val_data.get("is_valid"):
            st.success("✅ **Code Validation Status: PASSED** (AST syntax, static dependencies, and data interfaces verified).")
        else:
            st.warning("⚠️ Code not yet validated or validation pending.")
            if st.button("🔨 Generate & Validate Code Now", type="primary"):
                with st.spinner("Generating reproducible Python experiment code..."):
                    res = orchestrator.generate_and_validate_code(exp.id)
                    st.rerun()

        # Display Code
        gen_res = orchestrator.generate_and_validate_code(exp.id)
        code_str = gen_res.get("generated_code", "")
        st.code(code_str, language="python")

    # TAB 5: EXECUTION & COLAB
    with tab5:
        st.markdown("#### Experiment Execution Controls")
        col_exec_local, col_exec_colab = st.columns(2)

        with col_exec_local:
            st.markdown("##### 💻 Local Execution")
            st.markdown("Execute the verified machine learning pipeline locally with live diagnostics and immediate evaluation metrics.")
            
            run_btn = st.button("🚀 Start Local Training Run", type="primary", use_container_width=True)
            if run_btn:
                with st.spinner("Executing pipeline locally..."):
                    understanding = json.loads(exp.problem_understanding_json or "{}")
                    if not understanding.get("dataset_required", True):
                        code_res = orchestrator.generate_code_without_dataset(exp.id)
                        res = orchestrator.execute_experiment_code(exp.id, code_res.get("generated_code", ""))
                        if res.get("success"):
                            st.success("Execution completed successfully! View results in the Results tab.")
                            st.rerun()
                        else:
                            st.error(f"Execution failed: {res.get('execution_result', {}).get('stderr', '')}")
                    else:
                        success, msg, run_rec = orchestrator.execution_agent.experiment_service.execute_local_training(exp.id)
                        if success:
                            st.success("Training run finished successfully! View results in the Results tab.")
                            st.rerun()
                        else:
                            st.error(f"Execution failed: {msg}")

        with col_exec_colab:
            st.markdown("##### ☁️ Google Colab Export")
            st.markdown("Export a fully executable Jupyter Notebook (.ipynb) customized with GPU checks, data loading, training, and evaluation.")
            
            nb_service = NotebookService(repo=repo)
            try:
                nb_path = nb_service.export_experiment_notebook(exp.id, version_number=exp.current_version)
                with open(nb_path, "rb") as f:
                    nb_bytes = f.read()

                st.download_button(
                    label="📥 Download Colab Notebook (.ipynb)",
                    data=nb_bytes,
                    file_name=nb_path.name,
                    mime="application/x-ipynb+json",
                    use_container_width=True
                )
                st.caption(f"Generated file: `{nb_path.name}`")
            except Exception as e:
                st.warning(f"Select a dataset and generate code before exporting: {e}")

        st.markdown("---")
        st.markdown("##### 📥 Import External Results from Colab")
        st.markdown("If you ran your model in Google Colab, upload the exported `experiment_results.json` to link the empirical metrics back to this experiment.")
        colab_file = st.file_uploader("Upload Colab experiment_results.json:", type=["json"])
        if colab_file is not None:
            if st.button("Associate Colab Results", use_container_width=True):
                try:
                    payload = json.loads(colab_file.getvalue().decode("utf-8"))
                    success, msg = orchestrator.execution_agent.experiment_service.import_colab_results(exp.id, payload)
                    if success:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)
                except Exception as ex:
                    st.error(f"Error parsing results JSON: {ex}")

    # TAB 6: RESULTS & VISUALIZATIONS
    with tab6:
        st.markdown("#### Empirical Evaluation Results & Visualizations")
        runs = sorted(exp.runs, key=lambda r: r.run_number)
        exec_results_data = json.loads(getattr(exp, "execution_results_json", None) or "{}")

        if not runs and not exec_results_data:
            st.info("No training runs recorded yet. Execute training in Tab 5 to view empirical test results.")
        elif not runs and exec_results_data:
            metrics_dict = exec_results_data.get("metrics", {})
            st.markdown(f"**Autonomous Execution Results** | **Duration:** `{exec_results_data.get('duration_seconds', 0):.2f}s`")
            if metrics_dict:
                m_cols = st.columns(min(4, max(1, len(metrics_dict))))
                for idx, (m_name, m_val) in enumerate(metrics_dict.items()):
                    with m_cols[idx % len(m_cols)]:
                        st.metric(m_name.replace("_", " ").upper(), f"{m_val:.4f}" if isinstance(m_val, float) else f"{m_val}")

            results_json = exec_results_data.get("results_json", {})
            if "history" in results_json and isinstance(results_json["history"], list):
                losses = [h.get("loss") for h in results_json["history"] if "loss" in h]
                if losses:
                    st.plotly_chart(VisualizationService.plot_convergence_curve(losses), use_container_width=True)
            elif "activations" in results_json and "x_values" in results_json:
                st.plotly_chart(VisualizationService.plot_activation_functions(results_json["x_values"], results_json["activations"]), use_container_width=True)
            elif "X" in results_json and "labels" in results_json:
                st.plotly_chart(VisualizationService.plot_clustering_2d(results_json["X"], results_json["labels"], results_json.get("centers")), use_container_width=True)

            with st.expander("📄 View Full Execution Output", expanded=False):
                st.text(exec_results_data.get("stdout", "No output recorded."))
        else:
            latest_run = runs[-1]
            st.markdown(f"**Latest Run:** `Run #{latest_run.run_number}` | **Mode:** `{latest_run.execution_mode}` | **Duration:** `{latest_run.duration_seconds:.2f}s`")
            
            metrics_payload = json.loads(latest_run.metrics_json or "{}")
            metrics_dict = metrics_payload.get("metrics", {})

            # Metric Cards
            m_cols = st.columns(min(4, max(1, len(metrics_dict))))
            for idx, (m_name, m_val) in enumerate(metrics_dict.items()):
                with m_cols[idx % len(m_cols)]:
                    st.metric(m_name.replace("_", " ").upper(), f"{m_val}")

            st.markdown("---")
            viz_col1, viz_col2 = st.columns(2)

            # Confusion Matrix Plot
            cm = metrics_payload.get("confusion_matrix")
            if cm:
                with viz_col1:
                    cm_fig = VisualizationService.plot_confusion_matrix(cm, labels=metrics_payload.get("class_labels"))
                    st.plotly_chart(cm_fig, use_container_width=True)

            # Feature Importance Plot
            fi = metrics_payload.get("feature_importances")
            if fi:
                with viz_col2:
                    fi_fig = VisualizationService.plot_feature_importance(fi)
                    st.plotly_chart(fi_fig, use_container_width=True)

            # Actual vs Predicted (for Regression)
            avp = metrics_payload.get("actual_vs_predicted")
            if avp:
                with viz_col1:
                    avp_fig = VisualizationService.plot_actual_vs_predicted(avp)
                    st.plotly_chart(avp_fig, use_container_width=True)
                with viz_col2:
                    res_fig = VisualizationService.plot_residuals(avp)
                    st.plotly_chart(res_fig, use_container_width=True)

            # Execution Logs
            with st.expander("📜 View Full Execution Logs", expanded=False):
                st.text(latest_run.execution_logs or "No logs recorded.")

            # Error Analysis & Recommendations
            error_analysis = json.loads(latest_run.error_analysis_json or "{}")
            if error_analysis:
                st.markdown("##### 💡 Error Diagnostics & Improvement Recommendations")
                recs = error_analysis.get("recommendations", [])
                for r in recs:
                    st.markdown(f"""
                    <div style="background: #F0FDF4; border-left: 4px solid #22C55E; padding: 12px; margin-bottom: 8px; border-radius: 4px;">
                        <strong>🚀 {r.get('title')}</strong> ({r.get('category').upper()})<br>
                        <span>Action: {r.get('action')}</span><br>
                        <span style="font-size: 13px; color: #475569;">Rationale: {r.get('rationale')}</span>
                    </div>
                    """, unsafe_allow_html=True)

                if st.button("✨ Apply Recommended Improvement as New Version", type="primary"):
                    new_ver = repo.create_new_version(
                        experiment_id=exp.id,
                        change_summary=f"Applied improvement: {recs[0].get('title', 'Tuning') if recs else 'Iteration'}",
                        config_snapshot={"previous_version": exp.current_version}
                    )
                    st.success(f"Created Version v{new_ver.version_number}! You can now modify parameters and train again.")
                    st.rerun()
