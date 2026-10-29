"""Autonomous AI Lab Assistant - Streamlit Application Entrypoint."""
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import streamlit as st
from config.settings import settings
from database.initialization import init_db
from database.repository import ExperimentRepository
from orchestration.workflow_orchestrator import WorkflowOrchestrator
from frontend.styles import apply_custom_styles
from frontend.components import render_status_badge
from frontend.pages import (
    render_ai_assistant,
    render_dashboard,
    render_new_experiment,
    render_dataset_explorer,
    render_experiment_workspace,
    render_model_laboratory,
    render_previous_experiments,
    render_experiment_comparison,
    render_reports,
    render_integrations,
    render_settings,
    render_help_documentation,
)

# Page configuration
st.set_page_config(
    page_title="Autonomous AI Lab Assistant",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom UI styling (Dark navy sidebar, clean cards, blue/teal accents)
apply_custom_styles()

# Initialize persistent database schema safely
@st.cache_resource
def get_initialized_services():
    init_db()
    repo = ExperimentRepository()
    orchestrator = WorkflowOrchestrator(repo=repo)
    return repo, orchestrator

repo, orchestrator = get_initialized_services()

# Session State defaults
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "AI Assistant"

if "current_experiment_id" not in st.session_state:
    # Pick latest experiment if available
    recent = repo.list_experiments(limit=1)
    st.session_state["current_experiment_id"] = recent[0].id if recent else None

def navigate_to(page_name: str):
    """Callback to switch pages smoothly."""
    st.session_state["current_page"] = page_name
    st.rerun()

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.markdown("## 🧪 AI Lab Assistant")
    st.markdown("<p style='color: #94A3B8; font-size: 13px; margin-top: -10px;'>Autonomous Deep Learning Research Platform</p>", unsafe_allow_html=True)
    st.markdown("---")

    # Current Experiment Workspace Status in Sidebar
    curr_id = st.session_state.get("current_experiment_id")
    curr_exp = repo.get_experiment(curr_id) if curr_id else None

    if curr_exp:
        st.markdown(f"**Active Workspace:**<br>`{curr_exp.name[:24]}`", unsafe_allow_html=True)
        st.markdown(render_status_badge(curr_exp.status), unsafe_allow_html=True)
        st.markdown(f"<span style='color: #94A3B8; font-size: 12px;'>Version v{curr_exp.current_version}</span>", unsafe_allow_html=True)
        st.markdown("---")

    # Navigation menu
    nav_options = [
        "AI Assistant",
        "Dashboard",
        "New Experiment",
        "Dataset Explorer",
        "Experiment Workspace",
        "Model Laboratory",
        "Previous Experiments",
        "Compare Experiments",
        "Reports",
        "Integrations",
        "Settings",
        "Help and Documentation"
    ]

    selected_page = st.radio(
        "Navigation",
        nav_options,
        index=nav_options.index(st.session_state["current_page"]) if st.session_state["current_page"] in nav_options else 0,
        label_visibility="collapsed"
    )

    if selected_page != st.session_state["current_page"]:
        st.session_state["current_page"] = selected_page
        st.rerun()

    st.markdown("---")
    # Quick Switcher for Previous Experiments in Sidebar
    all_exps = repo.list_experiments(limit=15)
    if all_exps:
        exp_options = {f"{e.name[:20]} ({e.id[:6]})": e.id for e in all_exps}
        selected_switch = st.selectbox(
            "Switch Experiment:",
            list(exp_options.keys()),
            index=list(exp_options.values()).index(curr_id) if curr_id in exp_options.values() else 0
        )
        new_switched_id = exp_options[selected_switch]
        if new_switched_id != curr_id:
            st.session_state["current_experiment_id"] = new_switched_id
            st.rerun()

    st.markdown(f"<div style='color: #64748B; font-size: 11px; margin-top: 20px;'>Persistent SQLite Storage: Active<br>Version: {settings.version}</div>", unsafe_allow_html=True)

# --- PAGE ROUTING ---
page = st.session_state["current_page"]

if page == "AI Assistant":
    render_ai_assistant(repo, orchestrator, on_navigate=navigate_to)
elif page == "Dashboard":
    render_dashboard(repo, on_navigate=navigate_to)
elif page == "New Experiment":
    render_new_experiment(repo, orchestrator, on_navigate=navigate_to)
elif page == "Dataset Explorer":
    render_dataset_explorer(repo, orchestrator, on_navigate=navigate_to)
elif page == "Experiment Workspace":
    render_experiment_workspace(repo, orchestrator, on_navigate=navigate_to)
elif page == "Model Laboratory":
    render_model_laboratory(on_navigate=navigate_to)
elif page == "Previous Experiments":
    render_previous_experiments(repo, on_navigate=navigate_to)
elif page == "Compare Experiments":
    render_experiment_comparison(repo, on_navigate=navigate_to)
elif page == "Reports":
    render_reports(repo, on_navigate=navigate_to)
elif page == "Integrations":
    render_integrations(on_navigate=navigate_to)
elif page == "Settings":
    render_settings(on_navigate=navigate_to)
elif page == "Help and Documentation":
    render_help_documentation(on_navigate=navigate_to)
