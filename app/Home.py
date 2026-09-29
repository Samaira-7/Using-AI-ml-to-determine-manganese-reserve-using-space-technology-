#!/usr/bin/env python3
"""
Home.py

M-MES: Manganese Mineral Exploration System
Multi-page Streamlit app entry point.

Run with:
    streamlit run app/Home.py

Pages (in the sidebar, auto-discovered from app/pages/):
    1_Prospectivity_Map.py   -- Track A: the upgraded app.py map/dashboard
    2_Live_AOI_Scanner.py    -- NEW: on-demand Earth Engine band-ratio tool
    3_Production_Shortfall.py-- Track B: shortfall_model.py dashboard
    4_Band_Ratio_Lab.py      -- NEW: interactive spectral-index explainer

SCOPE (read first): this app screens satellite-visible SURFACE
reflectance for spectral patterns statistically associated with known
manganese occurrences. It does not detect underground ore, and no page
in this app claims otherwise.
"""

import streamlit as st
from theme import apply_theme, top_bar, scope_banner, status_pill

apply_theme("M-MES | Manganese Mineral Exploration System")
top_bar("SIH26009 &middot; Ministry of Steel &middot; MOIL Limited")

scope_banner(
    "This system screens <b>surface</b> spectral reflectance for patterns "
    "associated with known manganese occurrences. It does not detect "
    "underground reserves, grade, or tonnage &mdash; every output is a "
    "field-validation priority signal, not a verdict."
)

st.title("Manganese Mineral Exploration System")
st.caption("Surface prospectivity mapping and production-shortfall forecasting for SIH26009.")

col1, col2 = st.columns([2, 1])
with col1:
    st.markdown(
        """
        This dashboard covers both halves of the problem statement:

        - **Track A &mdash; Identify reserves.** Sentinel-2 / Landsat band
          ratios (MORI, HAI, FMI) plus SRTM terrain context, fed into a
          spatially cross-validated model, ranked as field-validation
          targets.
        - **Track B &mdash; Predict shortfalls.** Real rainfall (CHIRPS) and
          real MOIL production history, combined with a clearly-labelled
          simulated equipment-downtime series, into a shortfall-risk flag.

        Use the sidebar to move between pages. Every page states plainly
        which of its numbers are real, satellite-derived data and which
        are simulated placeholders &mdash; look for the pills below.
        """
    )
    st.markdown(
        status_pill("REAL DATA", "real") + status_pill("SIMULATED", "sim") + status_pill("ROADMAP", "roadmap"),
        unsafe_allow_html=True,
    )

with col2:
    st.markdown("#### Quick links")
    st.page_link("pages/1_Prospectivity_Map.py", label="Open the prospectivity map", icon="\U0001F5FA\uFE0F")
    st.page_link("pages/2_Live_AOI_Scanner.py", label="Run a live AOI scan", icon="\U0001F6F0\uFE0F")
    st.page_link("pages/3_Production_Shortfall.py", label="Shortfall forecast", icon="\U0001F4C9")
    st.page_link("pages/4_Band_Ratio_Lab.py", label="Band-ratio lab", icon="\U0001F52C")

st.divider()

st.subheader("What's real vs. simulated in this build")
st.markdown(
    """
| Data | Status | Source |
|---|---|---|
| Landsat / Sentinel-2 / SRTM DEM | **Real, live** | Google Earth Engine |
| Rainfall, soil moisture, land surface temperature | **Real, live** | CHIRPS / MODIS LST via Earth Engine |
| GSI lithology & occurrence points | **Real, manual** | GSI Bhukosh |
| MOIL production figures | **Real, public** | IBM Yearbook / MOIL annual reports |
| PRISMA hyperspectral | **Roadmap** | Registration pending, ASI |
| Equipment downtime / maintenance logs | **Simulated** | Poisson-process stand-in, clearly labelled |
"""
)

st.caption(
    "M-MES is a hackathon prototype demonstrating an AI/ML + remote sensing + "
    "production-forecasting workflow. It is not a certified reserve estimate "
    "or an operational planning tool."
)
