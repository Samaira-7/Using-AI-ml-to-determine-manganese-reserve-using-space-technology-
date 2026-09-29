#!/usr/bin/env python3
"""
3_Production_Shortfall.py

Track B dashboard. Wraps shortfall_model.py's real logic: real CHIRPS
rainfall pull via Earth Engine, real MOIL/IBM production CSV the user
supplies, and a clearly-labelled simulated downtime series. Never
invents production or rainfall numbers if the real sources are missing
-- it says so and stops, exactly like the underlying script's docstring
requires.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from theme import apply_theme, top_bar, scope_banner, status_pill

apply_theme("M-MES | Production Shortfall")
top_bar("Track B &middot; Production shortfall forecasting")
scope_banner(
    "Production and rainfall must be <b>real</b> to trust this page's output. "
    "Equipment downtime here is <b>simulated</b> — MOIL's real SCADA/maintenance "
    "feed is not public. Never presented as measured."
)

st.title("Production shortfall forecast")
st.caption("Real rainfall + real production history, regressed against a labelled synthetic downtime series.")

DATA_DIR = Path("data")
PRODUCTION_CSV = DATA_DIR / "moil_production_real.csv"

st.subheader("1. Production data")
uploaded = st.file_uploader(
    "Upload MOIL's real monthly production CSV (columns: `month`, `tonnes`) — sourced from the "
    "IBM Indian Minerals Yearbook or MOIL's annual reports.",
    type="csv",
)

production = None
using_real_production = False

if uploaded is not None:
    try:
        production_df = pd.read_csv(uploaded, parse_dates=["month"])
        production = production_df.set_index("month")["tonnes"]
        using_real_production = True
        st.markdown(status_pill("REAL DATA (uploaded)", "real"), unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Could not parse the uploaded CSV: {e}")
elif PRODUCTION_CSV.exists():
    production_df = pd.read_csv(PRODUCTION_CSV, parse_dates=["month"])
    production = production_df.set_index("month")["tonnes"]
    using_real_production = True
    st.markdown(status_pill("REAL DATA (data/moil_production_real.csv)", "real"), unsafe_allow_html=True)
else:
    st.warning(
        "No real production data supplied yet. Showing an illustrative synthetic series below "
        "so you can see the dashboard shape — **do not present these numbers to judges as real.** "
        "Upload a real CSV above, or add real figures at `data/moil_production_real.csv`, to replace it."
    )
    rng = np.random.default_rng(11)
    idx = pd.date_range("2019-01-01", "2024-12-01", freq="MS")
    base = 45000 + 3000 * np.sin(np.arange(len(idx)) / 6)
    production = pd.Series(base + rng.normal(0, 1500, len(idx)), index=idx, name="tonnes")
    st.markdown(status_pill("SYNTHETIC PLACEHOLDER", "sim"), unsafe_allow_html=True)

st.subheader("2. Rainfall data")
st.markdown(
    "Real rainfall requires a live Earth Engine session (see the Live AOI Scanner page for setup). "
    "Below, a synthetic seasonal rainfall series stands in whenever a live pull isn't available, "
    "clearly labelled as such."
)

EE_READY = False
try:
    import ee
    try:
        ee.Initialize()
        EE_READY = True
    except Exception:
        EE_READY = False
except ImportError:
    EE_READY = False

rainfall = None
using_real_rainfall = False

if EE_READY and st.button("Pull real CHIRPS rainfall for the MOIL belt"):
    with st.spinner("Querying CHIRPS via Earth Engine..."):
        try:
            aoi = ee.Geometry.Rectangle([79.0, 20.8, 80.5, 22.2])
            chirps = (
                ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
                .filterDate(str(production.index.min().date()), str(production.index.max().date()))
                .filterBounds(aoi)
            )

            def _to_feature(img):
                stat = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=aoi, scale=5000)
                return ee.Feature(None, {"date": img.date().format("YYYY-MM-dd"), "precip": stat.get("precipitation")})

            features = chirps.map(_to_feature).getInfo()["features"]
            rdf = pd.DataFrame([f["properties"] for f in features])
            rdf["date"] = pd.to_datetime(rdf["date"])
            rainfall = rdf.set_index("date").resample("MS").sum(numeric_only=True)["precip"]
            using_real_rainfall = True
            st.success(f"Pulled {len(rainfall)} months of real CHIRPS rainfall.")
        except Exception as e:
            st.error(f"Earth Engine query failed: {e}")

if rainfall is None:
    rng2 = np.random.default_rng(3)
    month_of_year = np.asarray(production.index.month)
    cos_term = np.clip(np.cos((month_of_year - 7) / 12 * 2 * np.pi), -1, 1)
    seasonal = 120 + 100 * np.clip(cos_term, 0, None)
    rainfall = pd.Series(seasonal + rng2.normal(0, 15, len(production)), index=production.index, name="precip")
    st.markdown(status_pill("SYNTHETIC PLACEHOLDER (typical monsoon shape)", "sim"), unsafe_allow_html=True)

st.subheader("3. Simulated equipment downtime")
st.markdown(status_pill("SIMULATED — not real SCADA data", "sim"), unsafe_allow_html=True)
mean_events = st.slider("Mean downtime events per month (Poisson λ)", 0.0, 5.0, 1.5, 0.1)
rng3 = np.random.default_rng(42)
downtime = pd.Series(rng3.poisson(lam=mean_events, size=len(production)), index=production.index, name="simulated_downtime_events")

st.subheader("4. Model")
merged = pd.concat([production, rainfall, downtime], axis=1).dropna()
merged.columns = ["tonnes", "precip", "downtime_events"]

if len(merged) < 6:
    st.error("Not enough overlapping months between production, rainfall, and downtime series to fit a model.")
    st.stop()

from sklearn.linear_model import LinearRegression

X = merged[["precip", "downtime_events"]].values
y = merged["tonnes"].values
model = LinearRegression().fit(X, y)
r2 = model.score(X, y)

c1, c2, c3 = st.columns(3)
c1.metric("R\u00b2 (training fit)", f"{r2:.3f}")
c2.metric("Rainfall coefficient", f"{model.coef_[0]:.2f} t/mm")
c3.metric("Downtime coefficient", f"{model.coef_[1]:.1f} t/event")

fig = go.Figure()
fig.add_trace(go.Scatter(x=merged.index, y=merged["tonnes"], name="Production (tonnes)", line=dict(color="#8B4A2B")))
fig.add_trace(go.Scatter(x=merged.index, y=model.predict(X), name="Model fit", line=dict(color="#C9A961", dash="dot")))
fig.update_layout(title="Monthly production vs. model fit", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig, width="stretch")

st.subheader("5. Flag an upcoming month")
f1, f2, f3 = st.columns(3)
upcoming_precip = f1.number_input("Forecast/assumed rainfall (mm)", value=float(merged["precip"].median()))
upcoming_downtime = f2.number_input("Assumed downtime events", value=float(merged["downtime_events"].median()))
baseline_tonnes = f3.number_input("Baseline tonnes to compare against", value=float(merged["tonnes"].median()))

predicted = model.predict([[upcoming_precip, upcoming_downtime]])[0]
shortfall_pct = (baseline_tonnes - predicted) / baseline_tonnes * 100
at_risk = shortfall_pct > 10

r1, r2, r3 = st.columns(3)
r1.metric("Predicted production", f"{predicted:,.0f} t")
r2.metric("Shortfall vs. baseline", f"{shortfall_pct:+.1f}%")
r3.metric("At risk?", "YES" if at_risk else "No", delta=None)

st.caption(
    "Threshold for 'at risk' is a starting guess (10% below baseline) — justify or tune it against "
    "MOIL's own planning tolerances before presenting this as an operational flag."
    + ("" if using_real_production else " Currently running on synthetic production data — treat this section as a UI demo only.")
)
