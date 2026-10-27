"""Professional UI Styles and Custom CSS."""

CUSTOM_CSS = """
<style>
/* Main typography and container spacing */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Sidebar styling: Dark navy / charcoal */
section[data-testid="stSidebar"] {
    background-color: #0F172A !important;
    color: #F8FAFC !important;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span {
    color: #E2E8F0 !important;
}

section[data-testid="stSidebar"] .stRadio > div {
    background-color: transparent !important;
}

/* Reusable Lab Metric Card */
.metric-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 18px 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    margin-bottom: 12px;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}

.metric-card:hover {
    box-shadow: 0 4px 10px rgba(0,0,0,0.08);
}

.metric-title {
    font-size: 13px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #64748B;
    margin-bottom: 6px;
}

.metric-value {
    font-size: 26px;
    font-weight: 700;
    color: #0F172A;
}

.metric-subtitle {
    font-size: 12px;
    color: #94A3B8;
    margin-top: 4px;
}

/* Status Badges */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 9999px;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}

.badge-completed {
    background-color: #DCFCE7;
    color: #15803D;
    border: 1px solid #BBF7D0;
}

.badge-running {
    background-color: #E0F2FE;
    color: #0369A1;
    border: 1px solid #BAE6FD;
}

.badge-draft {
    background-color: #F1F5F9;
    color: #475569;
    border: 1px solid #E2E8F0;
}

.badge-failed {
    background-color: #FEE2E2;
    color: #B91C1C;
    border: 1px solid #FECACA;
}

.badge-awaiting {
    background-color: #FEF3C7;
    color: #B45309;
    border: 1px solid #FDE68A;
}

/* Content Cards */
.lab-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 16px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04);
}

.lab-card-header {
    font-size: 16px;
    font-weight: 600;
    color: #0F172A;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

/* Table styling adjustments */
div[data-testid="stDataFrame"] {
    border-radius: 8px;
    border: 1px solid #E2E8F0;
}

/* Hide streamlit default hamburger menu if desired */
#MainMenu {visibility: visible;}
footer {visibility: hidden;}
</style>
"""

def apply_custom_styles():
    """Injects custom CSS into the Streamlit session."""
    import streamlit as st
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
