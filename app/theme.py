"""
theme.py

Shared visual theme for the M-MES Streamlit app. Import apply_theme() at
the top of every page for a consistent, non-default look (Streamlit's
default theme is what every other hackathon dashboard looks like).

This module only changes appearance (CSS injection + page config
defaults). It contains no data logic and fabricates nothing.
"""

import streamlit as st

PALETTE = {
    "ink": "#14170F",
    "ink_soft": "#3A3F31",
    "basalt": "#1A1D17",
    "cream": "#F3EFE2",
    "cream_dim": "#E7E0CC",
    "rust": "#8B4A2B",
    "rust_bright": "#B15C33",
    "gold": "#C9A961",
    "lichen": "#57694D",
}

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"]  {
    font-family: 'Inter', -apple-system, sans-serif;
}

/* Headings use the display serif */
h1, h2, h3 {
    font-family: 'Fraunces', serif !important;
    font-weight: 600 !important;
    color: #14170F;
}

/* App background: warm cream instead of Streamlit default white/grey */
.stApp {
    background-color: #F8F5EB;
}

/* Sidebar: basalt dark, cream text -- reads as a control console */
section[data-testid="stSidebar"] {
    background-color: #1A1D17;
}
section[data-testid="stSidebar"] * {
    color: #F3EFE2 !important;
}
section[data-testid="stSidebar"] hr {
    border-color: rgba(243,239,226,0.15);
}

/* Metric cards -- give them a defined edge instead of floating text */
div[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid rgba(20,23,15,0.1);
    border-radius: 8px;
    padding: 14px 18px;
}
div[data-testid="stMetricLabel"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem !important;
    color: #57694D !important;
    text-transform: none;
}
div[data-testid="stMetricValue"] {
    color: #8B4A2B !important;
    font-family: 'Fraunces', serif !important;
}

/* Primary buttons -> rust accent */
.stButton > button[kind="primary"], .stDownloadButton > button {
    background-color: #8B4A2B;
    border-color: #8B4A2B;
    color: #F3EFE2;
}
.stButton > button[kind="primary"]:hover, .stDownloadButton > button:hover {
    background-color: #B15C33;
    border-color: #B15C33;
}

/* Tabs */
button[data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
}

/* Expander headers */
.streamlit-expanderHeader {
    font-family: 'Fraunces', serif;
}

/* Top banner strip used on every page */
.mmes-topbar {
    display:flex; align-items:center; justify-content:space-between;
    padding: 10px 0 18px; border-bottom: 1px solid rgba(20,23,15,0.12);
    margin-bottom: 22px;
}
.mmes-brand {
    font-family: 'Fraunces', serif; font-weight: 700; font-size: 1.4rem; color:#14170F;
}
.mmes-brand span { color:#8B4A2B; }
.mmes-tag {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; color:#57694D;
}

/* Scope banner -- always visible reminder of what this tool is / isn't */
.mmes-scope-banner {
    background: rgba(139,74,43,0.08);
    border-left: 3px solid #8B4A2B;
    padding: 10px 16px;
    border-radius: 0 6px 6px 0;
    font-size: 0.85rem;
    color: #3A3F31;
    margin-bottom: 18px;
}

/* Data status pills */
.mmes-pill {
    display:inline-block; padding:2px 10px; border-radius:999px;
    font-family:'IBM Plex Mono', monospace; font-size:0.72rem; font-weight:500;
    margin-right:6px;
}
.mmes-pill-real { background:rgba(87,105,77,0.15); color:#3E4C33; }
.mmes-pill-sim { background:rgba(139,74,43,0.15); color:#8B4A2B; }
.mmes-pill-roadmap { background:rgba(201,169,97,0.2); color:#8C6F2F; }
</style>
"""


def apply_theme(page_title: str, page_icon: str = "\u26f0\ufe0f"):
    """Call once, first, on every page. Sets page config + injects CSS."""
    st.set_page_config(page_title=page_title, page_icon=page_icon, layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def top_bar(current_tag: str):
    """Render the consistent brand strip at the top of a page."""
    st.markdown(
        f"""
        <div class="mmes-topbar">
            <div class="mmes-brand">M&#8209;MES <span>&#9679;</span> Manganese Mineral Exploration System</div>
            <div class="mmes-tag">{current_tag}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def scope_banner(text: str):
    st.markdown(f'<div class="mmes-scope-banner">{text}</div>', unsafe_allow_html=True)


def status_pill(label: str, kind: str = "real"):
    css_class = {"real": "mmes-pill-real", "sim": "mmes-pill-sim", "roadmap": "mmes-pill-roadmap"}.get(kind, "mmes-pill-real")
    return f'<span class="mmes-pill {css_class}">{label}</span>'
