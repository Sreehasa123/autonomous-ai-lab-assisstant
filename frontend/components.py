"""Reusable UI Components for Streamlit Dashboard."""
import streamlit as st
from typing import Optional, List, Dict, Any

def render_metric_card(title: str, value: Any, subtitle: str = "", delta: Optional[str] = None):
    """Renders a styled metric card."""
    delta_html = f"<span style='color: #059669; font-size: 12px; margin-left: 6px;'>{delta}</span>" if delta else ""
    html = f"""
    <div class="metric-card">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}{delta_html}</div>
        <div class="metric-subtitle">{subtitle}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def render_status_badge(status: str) -> str:
    """Returns an HTML badge formatted according to the experiment status."""
    st_clean = status.lower()
    if st_clean in ("completed", "success"):
        cls = "badge-completed"
    elif st_clean in ("running", "executing"):
        cls = "badge-running"
    elif st_clean in ("failed", "error"):
        cls = "badge-failed"
    elif st_clean in ("awaiting_approval", "ready_to_run", "planning"):
        cls = "badge-awaiting"
    else:
        cls = "badge-draft"

    label = status.replace("_", " ").upper()
    return f'<span class="badge {cls}">{label}</span>'

def render_section_header(title: str, subtitle: str = "", badge: Optional[str] = None):
    """Renders a clean section header with optional status badge."""
    badge_html = f" {badge}" if badge else ""
    st.markdown(f"### {title}{badge_html}")
    if subtitle:
        st.markdown(f"<p style='color: #64748B; font-size: 14px; margin-top: -8px; margin-bottom: 16px;'>{subtitle}</p>", unsafe_allow_html=True)

def render_empty_state(title: str, message: str, button_label: Optional[str] = None, button_key: Optional[str] = None) -> bool:
    """Renders an empty state placeholder with an optional action button."""
    st.markdown(f"""
    <div style="text-align: center; padding: 40px 20px; background: #F8FAFC; border: 1px dashed #CBD5E1; border-radius: 12px; margin: 20px 0;">
        <div style="font-size: 36px; margin-bottom: 10px;">🧪</div>
        <h4 style="color: #1E293B; margin-bottom: 6px;">{title}</h4>
        <p style="color: #64748B; max-width: 500px; margin: 0 auto 16px auto; font-size: 14px;">{message}</p>
    </div>
    """, unsafe_allow_html=True)
    if button_label:
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            return st.button(button_label, key=button_key or "empty_state_btn", use_container_width=True)
    return False

def render_step_progress(current_step_index: int, total_steps: int = 6):
    """Renders a horizontal progress bar for multi-step workflows."""
    progress = min(1.0, max(0.0, current_step_index / total_steps))
    st.progress(progress)
