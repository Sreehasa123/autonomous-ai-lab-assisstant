"""Application Settings Page."""
import os
import streamlit as st
from config.settings import settings
from frontend.components import render_section_header

def render_settings(on_navigate):
    """User preferences, default parameters, and runtime path configurations."""
    render_section_header("Assistant Settings & Preferences", "Configure execution defaults, storage paths, and optional API credentials.")

    with st.form("settings_form"):
        st.markdown("#### General Preferences")
        c1, c2 = st.columns(2)
        with c1:
            fw = st.selectbox("Default Modeling Framework", ["scikit-learn", "PyTorch"], index=0 if settings.default_framework == "scikit-learn" else 1)
            exec_mode = st.selectbox("Default Execution Mode", ["local", "colab"], index=0 if settings.default_execution_mode == "local" else 1)
        with c2:
            seed = st.number_input("Global Random Seed", value=int(settings.random_seed))
            timeout = st.number_input("Execution Timeout (seconds)", value=int(settings.execution_timeout_seconds), min_value=30, max_value=3600)

        st.markdown("#### Optional LLM Reasoning Provider")
        llm_p = st.selectbox("LLM Provider", ["none", "gemini", "openai", "anthropic"], index=["none", "gemini", "openai", "anthropic"].index(settings.llm_provider.lower() if settings.llm_provider.lower() in ["none", "gemini", "openai", "anthropic"] else "none"))
        llm_key = st.text_input("LLM API Key (Masked)", value="" if not settings.llm_api_key else ("•" * 12 + settings.llm_api_key[-4:] if len(settings.llm_api_key) > 4 else "••••"), type="password", help="Leave blank to maintain current value.")

        st.markdown("#### Storage & Database Paths")
        st.code(f"Database URL: {settings.database_url}\nBase Directory: {settings.base_dir}\nArtifacts Path: {settings.artifacts_dir}", language="bash")

        saved = st.form_submit_button("💾 Save Settings", type="primary", use_container_width=True)

    if saved:
        settings.default_framework = fw
        settings.default_execution_mode = exec_mode
        settings.random_seed = int(seed)
        settings.execution_timeout_seconds = int(timeout)
        if llm_p != settings.llm_provider:
            settings.llm_provider = llm_p
        if llm_key and not llm_key.startswith("•"):
            settings.llm_api_key = llm_key
            os.environ["LLM_API_KEY"] = llm_key

        st.success("Settings updated successfully for this session!")
