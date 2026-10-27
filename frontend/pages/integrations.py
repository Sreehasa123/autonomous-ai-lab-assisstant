"""Integrations Status and Configuration Page."""
import os
import streamlit as st
from config.settings import settings
from integrations.kaggle_provider import KaggleProvider
from services.llm_service import LLMService
from frontend.components import render_section_header

def render_integrations(on_navigate):
    """Monitors connectivity status and setup instructions for external providers."""
    render_section_header("External Services & Integrations", "Configure optional third-party providers for multi-source dataset discovery, GPU execution, and LLM reasoning.")

    # 1. UCI ML Repository
    with st.expander("🏛️ UCI Machine Learning Repository (Active by Default)", expanded=True):
        st.markdown("**Status:** ✅ Available & Connected")
        st.markdown("Accesses standard benchmark datasets (Iris, Wine, Breast Cancer, California Housing, Diabetes, Heart Disease) directly without authentication.")

    # 2. OpenML
    with st.expander("🌐 OpenML Benchmark Platform", expanded=True):
        openml_configured = bool(settings.openml_api_key)
        st.markdown(f"**Status:** {'✅ API Key Configured' if openml_configured else 'ℹ️ Public Access Mode (No key required for standard queries)'}")
        st.markdown("Provides programmatic access to thousands of machine learning datasets and benchmark task definitions.")
        st.caption("Optional: Set `OPENML_API_KEY` in `.env` to authenticate private datasets.")

    # 3. Hugging Face
    with st.expander("🤗 Hugging Face Datasets Hub", expanded=True):
        hf_configured = bool(settings.hf_token)
        st.markdown(f"**Status:** {'✅ Authenticated Token Configured' if hf_configured else 'ℹ️ Public Access Mode (Rate-limited)'}")
        st.markdown("Enables natural language and computer vision dataset queries across the Hugging Face hub.")
        st.caption("Optional: Set `HF_TOKEN` in `.env` to access gated datasets.")

    # 4. Kaggle
    with st.expander("🏆 Kaggle Datasets", expanded=True):
        kaggle_provider = KaggleProvider()
        has_k, msg = kaggle_provider.check_credentials()
        st.markdown(f"**Status:** {'✅ ' + msg if has_k else '⚠️ Unconfigured'}")
        if not has_k:
            st.info("To enable Kaggle search, download `kaggle.json` from your Kaggle Account Settings and place it in `~/.kaggle/kaggle.json`, or set `KAGGLE_USERNAME` and `KAGGLE_KEY` in `.env`.")

    # 5. Google Colab
    with st.expander("☁️ Google Colab Integration", expanded=True):
        st.markdown("**Status:** ✅ Native Notebook (.ipynb) Generator Active")
        st.markdown("The application automatically exports syntactically valid `.ipynb` notebooks customized to the experiment dataset, model, and metrics serialization for import.")

    # 6. LLM Reasoning Integration
    with st.expander("🧠 Large Language Model (LLM) Integration", expanded=True):
        llm = LLMService()
        is_llm = llm.is_configured()
        st.markdown(f"**Status:** {'✅ Active Provider: ' + llm.provider.upper() if is_llm else 'ℹ️ Rule-Based Reasoning Active (Default fallback)'}")
        st.markdown("When an API key is configured, the assistant leverages LLM reasoning for problem understanding and customized code generation. When unconfigured, deterministic rule-based algorithms are used.")
        st.caption("Supported: Gemini, OpenAI, Anthropic. Set `LLM_PROVIDER` and `LLM_API_KEY` (or `GEMINI_API_KEY`) in `.env`.")
