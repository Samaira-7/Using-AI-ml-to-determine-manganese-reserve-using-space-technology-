#!/usr/bin/env python3
"""
4_Band_Ratio_Lab.py

NEW page. Interactive, educational explainer for the band-ratio math
used throughout the pipeline -- lets a user (a judge, a new teammate)
build intuition for MORI/HAI/FMI/NDVI before looking at the live map.
Pure client-side math mirroring spectral_indices.py; no network calls.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from theme import apply_theme, top_bar, scope_banner

apply_theme("M-MES | Band Ratio Lab")
top_bar("Explainer &middot; No live data used on this page")
scope_banner(
    "Everything on this page is illustrative single-pixel math, mirroring "
    "<code>spectral_indices.py</code> exactly. It is not derived from satellite imagery."
)

st.title("Band-ratio lab")
st.caption("Build intuition for how reflectance numbers become a mineral-screening score.")

EPS = 1e-6


def safe_divide(n, d):
    return n / (d + EPS)


st.subheader("Why ratios instead of raw reflectance?")
st.markdown(
    """
    A raw reflectance value mixes two things together: how much light a
    surface actually reflects (its **composition**) and how much light
    hit it in the first place (**illumination** — time of day, slope
    angle, shadow). Two identical rocks on a sunlit slope and a shaded
    slope will show very different raw brightness values despite being
    the same mineral.

    Dividing one band by another cancels out most of that illumination
    effect, because it affects both bands roughly proportionally, leaving
    a number that tracks composition instead. That's the entire idea
    behind every ratio below.
    """
)

st.divider()
st.subheader("Interactive: move the sliders, watch the ratios respond")

colL, colR = st.columns([1, 1.3])

with colL:
    st.markdown("**Reflectance inputs** (0&ndash;1 scale, typical surface values)")
    blue = st.slider("Blue", 0.0, 0.5, 0.09, 0.005)
    red = st.slider("Red", 0.0, 0.5, 0.14, 0.005)
    nir = st.slider("NIR", 0.0, 0.6, 0.22, 0.005)
    swir1 = st.slider("SWIR1", 0.0, 0.6, 0.28, 0.005)
    swir2 = st.slider("SWIR2", 0.0, 0.6, 0.19, 0.005)

mori = safe_divide(red, blue)
hai = safe_divide(swir1, swir2)
fmi = safe_divide(swir1, nir)
ndvi = safe_divide(nir - red, nir + red)

with colR:
    st.markdown("**Resulting indices**")
    m1, m2 = st.columns(2)
    m1.metric("MORI (Red/Blue)", f"{mori:.2f}")
    m2.metric("HAI (SWIR1/SWIR2)", f"{hai:.2f}")
    m3, m4 = st.columns(2)
    m3.metric("FMI (SWIR1/NIR)", f"{fmi:.2f}")
    m4.metric("NDVI", f"{ndvi:.2f}")

    fig = go.Figure(go.Bar(
        x=["MORI", "HAI", "FMI"], y=[mori, hai, fmi],
        marker_color=["#8B4A2B", "#C9A961", "#57694D"],
    ))
    fig.update_layout(
        title="Index comparison (higher = stronger signal)",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        yaxis_title="ratio value", height=320,
    )
    st.plotly_chart(fig, width="stretch")

    if ndvi > 0.3:
        st.warning(
            "NDVI above ~0.3 typically means the pixel is vegetated. In the real pipeline, "
            "this pixel would likely be masked out — the sensor can't see the ground surface "
            "underneath dense vegetation, so the ratios above become unreliable here."
        )

st.divider()
st.subheader("Reference: known surface material signatures")
st.caption("Illustrative ranges from the remote-sensing literature (Sabins, 1999) — not measurements from this project's AOI.")

reference = pd.DataFrame({
    "Surface type": ["Fresh basalt", "Lateritic soil (Fe/Mn-rich)", "Dense vegetation", "Bare quartzite", "Clay alteration zone"],
    "Typical MORI": [1.1, 1.9, 1.0, 1.2, 1.3],
    "Typical HAI": [1.0, 1.3, 1.0, 0.9, 1.8],
    "Typical NDVI": [0.05, 0.10, 0.65, 0.05, 0.15],
})
st.dataframe(reference, width="stretch", hide_index=True)

st.info(
    "These reference numbers are for building intuition only. Real classification always "
    "requires calibration against known occurrence points in the actual AOI — a ratio that "
    "means 'manganese-associated' in one geological setting can mean something else entirely "
    "in another."
)
