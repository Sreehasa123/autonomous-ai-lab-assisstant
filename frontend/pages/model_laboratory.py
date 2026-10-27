"""Model Laboratory Page."""
import streamlit as st
import pandas as pd
from services.model_service import ModelService
from frontend.components import render_section_header

def render_model_laboratory(on_navigate):
    """Explores candidate machine learning & deep learning architectures and hyperparameters."""
    render_section_header("Model Architecture Laboratory", "Compare model families, inspect algorithm trade-offs, and fine-tune hyperparameter strategies.")

    c_task, c_dl = st.columns([2, 1])
    with c_task:
        task_category = st.selectbox("Task Category:", ["tabular_classification", "tabular_regression", "image_classification", "text_classification"])
    with c_dl:
        include_dl = st.checkbox("Include Deep Learning Architectures", value=True)

    model_service = ModelService()
    recommendations = model_service.recommend_models(task_type=task_category, is_deep_learning=include_dl)

    st.markdown("#### Candidate Architectures Catalog")

    for idx, model_cfg in enumerate(recommendations):
        with st.expander(f"{'⭐ [Primary Baseline] ' if model_cfg.is_baseline else ''}{model_cfg.model_name} ({model_cfg.model_family.title()})", expanded=(idx == 0)):
            st.markdown(f"**Framework:** `{model_cfg.framework}`")
            st.markdown(f"**Architectural Rationale:** {model_cfg.explanation}")
            st.markdown("**Default Hyperparameters:**")
            st.json(model_cfg.hyperparameters)

    st.markdown("---")
    st.markdown("#### Model Family Comparison Matrix")
    comparison_data = [
        {"Model Family": "Ensemble (Random Forest)", "Training Speed": "Fast (Multi-threaded)", "Overfitting Risk": "Low (Bagging)", "Interpretability": "High (Feature Importances)", "Data Scale": "Small - Medium"},
        {"Model Family": "Boosting (Gradient Boosting)", "Training Speed": "Moderate", "Overfitting Risk": "Moderate (Sequential)", "Interpretability": "High", "Data Scale": "Small - Large"},
        {"Model Family": "Linear / Regularized", "Training Speed": "Ultra-Fast", "Overfitting Risk": "Very Low (L1/L2)", "Interpretability": "Maximum (Weights)", "Data Scale": "Any"},
        {"Model Family": "Neural Networks (MLP/Deep)", "Training Speed": "Moderate (CPU) / Fast (GPU)", "Overfitting Risk": "High without Regularization", "Interpretability": "Black-Box (SHAP needed)", "Data Scale": "Large"}
    ]
    st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)
