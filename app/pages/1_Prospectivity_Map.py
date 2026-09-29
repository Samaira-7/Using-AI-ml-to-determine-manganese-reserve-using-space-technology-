#!/usr/bin/env python3
"""
1_Prospectivity_Map.py

Track A dashboard page. Same computation and honesty rules as the
original app.py (data validation, priority classification, exports),
restyled with the shared M-MES theme and with a clearly-labelled demo
dataset offered when no real prediction CSV is present, so the page is
explorable immediately rather than dead on first load.
"""

import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

import streamlit as st
import folium
from folium.plugins import MarkerCluster, Fullscreen, MiniMap, HeatMap
from streamlit_folium import st_folium
import plotly.express as px

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from theme import apply_theme, top_bar, scope_banner

try:
    import geopandas as gpd
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False

apply_theme("M-MES | Prospectivity Map")
top_bar("Track A &middot; Reserve identification")
scope_banner(
    "Scores indicate <b>relative surface prospectivity</b> from remote-sensing "
    "and terrain features. They do not confirm underground manganese "
    "reserves, ore grade, or economic viability."
)

# ============================================================================
# CONFIGURATION (unchanged from the original app.py contract)
# ============================================================================
DATA_DIRECTORY = Path("data")
PREDICTION_FILE_PATH = DATA_DIRECTORY / "mmes_predictions.csv"
LINEAMENTS_GEOJSON_PATH = DATA_DIRECTORY / "lineaments.geojson"
GEOLOGY_BOUNDARIES_PATH = DATA_DIRECTORY / "geology_boundaries.geojson"
MORI_OVERLAY_GEOJSON_PATH = DATA_DIRECTORY / "mori_overlay.geojson"
MODEL_METADATA_PATH = DATA_DIRECTORY / "model_metadata.json"

DEFAULT_ZOOM = 9
DEFAULT_THRESHOLD = 0.70
MIN_THRESHOLD = 0.50
MAX_THRESHOLD = 0.99
THRESHOLD_STEP = 0.01

MAP_TILE_OPTIONS = ["OpenStreetMap", "Esri World Imagery", "CartoDB Positron"]

REGION_CENTERS = {
    "Balaghat Belt (MP)": {"lat": 21.90, "lon": 80.05, "zoom": 9},
    "Keonjhar Belt (Odisha)": {"lat": 21.63, "lon": 85.58, "zoom": 9},
    "Shivamogga (Karnataka)": {"lat": 13.93, "lon": 75.57, "zoom": 9},
}

HIGH_PRIORITY_THRESHOLD = 0.85
MEDIUM_PRIORITY_THRESHOLD = 0.70
PRIORITY_COLORS = {"High Priority": "red", "Medium Priority": "orange", "Other Threshold-Qualified": "blue"}
CLUSTER_THRESHOLD = 150
HEATMAP_MIN_POINTS = 30
MAX_MAP_MARKERS = 3000

POPUP_RECOMMENDED_ACTION = "Desktop geological review and field validation required."
EXPORT_RECOMMENDED_ACTION = "Ground validation required; this output is not confirmation of a mineral reserve."

REQUIRED_PREDICTION_COLUMNS = ["longitude", "latitude", "prospectivity_score"]
OPTIONAL_NUMERIC_FEATURE_COLUMNS = ["MORI", "HAI", "FMI", "NDVI", "Elevation", "Slope", "Aspect"]
EXPORT_COLUMN_ORDER = [
    "target_id", "longitude", "latitude", "prospectivity_score", "priority_category",
    "MORI", "HAI", "FMI", "NDVI", "Elevation", "Slope", "Aspect", "validation_status", "model_version",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MMES_App")


# ============================================================================
# HELPERS (identical logic to app.py)
# ============================================================================
def format_value_safe(value, decimals=None, suffix=""):
    try:
        is_missing = pd.isna(value)
    except (TypeError, ValueError):
        is_missing = value is None
    if is_missing:
        return "Not available"
    if decimals is not None:
        try:
            return f"{float(value):.{decimals}f}{suffix}"
        except (TypeError, ValueError):
            return f"{value}{suffix}"
    return f"{value}{suffix}"


def classify_priority(score, high=HIGH_PRIORITY_THRESHOLD, medium=MEDIUM_PRIORITY_THRESHOLD):
    if pd.isna(score):
        return "Other Threshold-Qualified"
    if score > high:
        return "High Priority"
    if medium <= score <= high:
        return "Medium Priority"
    return "Other Threshold-Qualified"


def sanitize_for_filename(text):
    return re.sub(r"[^A-Za-z0-9]+", "_", str(text)).strip("_")


def generate_export_filenames(region, threshold):
    region_slug = sanitize_for_filename(region)
    threshold_str = f"{threshold:.2f}"
    return (f"m_mes_field_targets_{region_slug}_{threshold_str}.csv",
            f"m_mes_field_targets_{region_slug}_{threshold_str}.geojson")


def validate_and_clean_predictions(raw_df):
    report = {"n_start": int(len(raw_df)), "dropped_invalid_coords": 0, "dropped_invalid_score": 0,
              "missing_required_columns": [], "n_end": 0}
    missing_required = [c for c in REQUIRED_PREDICTION_COLUMNS if c not in raw_df.columns]
    report["missing_required_columns"] = missing_required
    if missing_required:
        return None, report

    df = raw_df.copy()
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["prospectivity_score"] = pd.to_numeric(df["prospectivity_score"], errors="coerce")

    invalid_coord_mask = (df["longitude"].isna() | df["latitude"].isna()
                           | df["longitude"].abs().gt(180) | df["latitude"].abs().gt(90))
    report["dropped_invalid_coords"] = int(invalid_coord_mask.sum())
    df = df.loc[~invalid_coord_mask].copy()

    invalid_score_mask = (df["prospectivity_score"].isna()
                           | (df["prospectivity_score"] < 0) | (df["prospectivity_score"] > 1))
    report["dropped_invalid_score"] = int(invalid_score_mask.sum())
    df = df.loc[~invalid_score_mask].copy()
    df = df.reset_index(drop=True)

    if "target_id" not in df.columns:
        df["target_id"] = ["T" + str(i) for i in df.index]
    else:
        missing_id_mask = df["target_id"].isna() | (df["target_id"].astype(str).str.strip() == "")
        if missing_id_mask.any():
            df.loc[missing_id_mask, "target_id"] = ["T" + str(i) for i in df.index[missing_id_mask]]

    for col in OPTIONAL_NUMERIC_FEATURE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    report["n_end"] = int(len(df))
    return df, report


def build_point_geojson_feature_collection(df):
    property_columns = [c for c in
        ["target_id", "prospectivity_score", "priority_category", "MORI", "HAI", "FMI", "NDVI",
         "Elevation", "Slope", "Aspect", "validation_status", "model_version"] if c in df.columns]
    features = []
    for _, row in df.iterrows():
        properties = {}
        for col in property_columns:
            val = row[col]
            if pd.isna(val):
                properties[col] = None
            elif isinstance(val, (int, float, np.integer, np.floating)):
                properties[col] = float(val)
            else:
                properties[col] = str(val)
        properties["recommended_action"] = EXPORT_RECOMMENDED_ACTION
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [float(row["longitude"]), float(row["latitude"])]},
            "properties": properties,
        })
    return {"type": "FeatureCollection", "features": features}


def build_popup_html(row):
    target_id = row.get("target_id", "Not available")
    lat_str = format_value_safe(row.get("latitude"), decimals=5)
    lon_str = format_value_safe(row.get("longitude"), decimals=5)
    score = row.get("prospectivity_score")
    score_str = "Not available" if pd.isna(score) else f"{float(score) * 100:.1f}%"
    priority = row.get("priority_category", "Not available")

    def field_row(label, key, decimals=3, suffix=""):
        val = row.get(key, np.nan)
        return f"<tr><td style='padding:2px 8px 2px 0;'><b>{label}</b></td><td>{format_value_safe(val, decimals=decimals, suffix=suffix)}</td></tr>"

    optional_rows = "".join([
        field_row("MORI", "MORI"), field_row("HAI", "HAI"), field_row("FMI", "FMI"), field_row("NDVI", "NDVI"),
        field_row("Elevation (m)", "Elevation", decimals=1), field_row("Slope", "Slope", decimals=1, suffix="&deg;"),
    ])
    validation_status_str = format_value_safe(row.get("validation_status"))
    model_version_str = format_value_safe(row.get("model_version"))

    return f"""
    <div style="font-family: Arial, sans-serif; font-size: 13px; min-width: 230px;">
        <h4 style="margin: 0 0 6px 0;">Target {target_id}</h4>
        <table style="width:100%; border-collapse: collapse;">
            <tr><td style='padding:2px 8px 2px 0;'><b>Latitude</b></td><td>{lat_str}</td></tr>
            <tr><td style='padding:2px 8px 2px 0;'><b>Longitude</b></td><td>{lon_str}</td></tr>
            <tr><td style='padding:2px 8px 2px 0;'><b>Prospectivity Score</b></td><td>{score_str}</td></tr>
            <tr><td style='padding:2px 8px 2px 0;'><b>Priority</b></td><td>{priority}</td></tr>
            {optional_rows}
            <tr><td style='padding:2px 8px 2px 0;'><b>Validation Status</b></td><td>{validation_status_str}</td></tr>
            <tr><td style='padding:2px 8px 2px 0;'><b>Model Version</b></td><td>{model_version_str}</td></tr>
        </table>
        <p style="margin-top:8px; font-style: italic; color:#444; font-size:12px;">
            Recommended next action: {POPUP_RECOMMENDED_ACTION}
        </p>
    </div>
    """


def build_legend_html():
    return """
    <div style="
        position: fixed; bottom: 30px; left: 30px; z-index: 9999;
        background-color: #F3EFE2; padding: 10px 14px; border: 1px solid #8B4A2B;
        border-radius: 6px; font-size: 12px; font-family: Arial, sans-serif;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.3); max-width: 260px;">
        <b>Legend</b><br>
        <span style="color:red;">&#9679;</span> High-priority candidate, score &gt; 0.85<br>
        <span style="color:orange;">&#9679;</span> Medium-priority candidate, score 0.70&ndash;0.85<br>
        <span style="color:blue;">&#9679;</span> Above selected threshold (below 0.70)<br>
        <span style="color:#555;">- - -</span> Optional structural/geology context layers<br>
        <hr style="margin:6px 0;">
        <i>Modelled surface prospectivity, not confirmed mineral reserves.</i>
    </div>
    """


@st.cache_data(show_spinner="Loading prediction data...")
def load_predictions(path_str):
    path = Path(path_str)
    if not path.exists():
        return None, {"file_missing": True, "path": str(path)}
    try:
        raw_df = pd.read_csv(path)
    except Exception as read_error:
        return None, {"file_missing": False, "read_error": str(read_error), "path": str(path)}
    df, report = validate_and_clean_predictions(raw_df)
    report["file_missing"] = False
    report["path"] = str(path)
    return df, report


@st.cache_data(show_spinner=False)
def make_demo_predictions(seed: int = 7) -> pd.DataFrame:
    """Synthetic, clearly-labelled demo dataset so the map is explorable
    with zero setup. Never presented as real -- validation_status and
    source_note both say SYNTHETIC DEMO explicitly, and the UI banners
    this above the map whenever it's in use."""
    rng = np.random.default_rng(seed)
    center = REGION_CENTERS["Balaghat Belt (MP)"]
    n = 220
    lats = center["lat"] + rng.normal(0, 0.18, n)
    lons = center["lon"] + rng.normal(0, 0.18, n)
    scores = np.clip(rng.beta(2, 3, n) * 1.15, 0, 0.99)
    df = pd.DataFrame({
        "target_id": [f"DEMO_{i:04d}" for i in range(n)],
        "longitude": lons, "latitude": lats,
        "prospectivity_score": scores,
        "region_name": "Balaghat Belt (MP)",
        "MORI": np.clip(rng.normal(1.4, 0.4, n), 0.4, 3.0),
        "HAI": np.clip(rng.normal(1.3, 0.35, n), 0.4, 2.8),
        "FMI": np.clip(rng.normal(1.1, 0.3, n), 0.3, 2.2),
        "NDVI": np.clip(rng.normal(0.25, 0.15, n), -0.1, 0.7),
        "Elevation": rng.normal(380, 60, n),
        "Slope": np.clip(rng.normal(8, 6, n), 0, 40),
        "Aspect": rng.uniform(0, 360, n),
        "validation_status": "SYNTHETIC DEMO -- not field reviewed",
        "model_version": "demo-v0 (illustrative only)",
        "source_note": "Randomly generated for UI demonstration; not derived from satellite imagery.",
    })
    return df


@st.cache_data(show_spinner=False)
def load_optional_geojson(path_str):
    path = Path(path_str)
    if not path.exists():
        return None, "missing"
    if not GEOPANDAS_AVAILABLE:
        return None, "geopandas_unavailable"
    try:
        gdf = gpd.read_file(path)
        if gdf.empty:
            return None, "empty"
        if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)
        return gdf, "ok"
    except Exception as geojson_error:
        return None, f"invalid ({geojson_error})"


@st.cache_data(show_spinner=False)
def load_model_metadata(path_str):
    path = Path(path_str)
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def create_prospectivity_map(filtered_data, selected_region, settings):
    region_info = REGION_CENTERS.get(selected_region, {"lat": 21.90, "lon": 80.05, "zoom": DEFAULT_ZOOM})
    m = folium.Map(location=[region_info["lat"], region_info["lon"]],
                    zoom_start=region_info.get("zoom", DEFAULT_ZOOM), tiles=None, control_scale=True)

    folium.TileLayer("OpenStreetMap", name="OpenStreetMap", overlay=False, control=True,
                      show=(settings["basemap"] == "OpenStreetMap")).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri, Maxar, Earthstar Geographics, and the GIS User Community",
        name="Esri World Imagery", overlay=False, control=True,
        show=(settings["basemap"] == "Esri World Imagery"),
    ).add_to(m)
    try:
        folium.TileLayer("CartoDB positron", name="CartoDB Positron", overlay=False, control=True,
                          show=(settings["basemap"] == "CartoDB Positron")).add_to(m)
    except Exception:
        pass

    if not filtered_data.empty:
        lats = filtered_data["latitude"].tolist()
        lons = filtered_data["longitude"].tolist()
        if len(filtered_data) == 1:
            m.location = [lats[0], lons[0]]
        else:
            m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])

    map_data = filtered_data
    if len(map_data) > MAX_MAP_MARKERS:
        map_data = map_data.sort_values("prospectivity_score", ascending=False).head(MAX_MAP_MARKERS)

    if settings.get("show_points", True) and not map_data.empty:
        use_cluster = len(map_data) > CLUSTER_THRESHOLD
        marker_sink = (MarkerCluster(name=f"Candidate Target Points (n={len(map_data)})").add_to(m)
                       if use_cluster else folium.FeatureGroup(name="Candidate Target Points").add_to(m))
        for _, row in map_data.iterrows():
            color = PRIORITY_COLORS.get(row.get("priority_category"), "blue")
            folium.Marker(
                location=[row["latitude"], row["longitude"]],
                popup=folium.Popup(build_popup_html(row), max_width=320),
                tooltip=f"{row.get('target_id', '')} - {row.get('priority_category', '')}",
                icon=folium.Icon(color=color, icon="flag"),
            ).add_to(marker_sink)

    if settings.get("show_hotspots", True):
        high_priority_df = map_data.loc[map_data["priority_category"] == "High Priority"]
        if not high_priority_df.empty:
            hotspot_group = folium.FeatureGroup(name="High-Priority Hotspots (Score > 0.85)").add_to(m)
            for _, row in high_priority_df.iterrows():
                folium.CircleMarker(
                    location=[row["latitude"], row["longitude"]], radius=8, color="red",
                    fill=True, fill_color="red", fill_opacity=0.6,
                    popup=folium.Popup(build_popup_html(row), max_width=320),
                ).add_to(hotspot_group)
            if len(high_priority_df) >= HEATMAP_MIN_POINTS:
                heat_group = folium.FeatureGroup(
                    name="Relative Density of Modelled Surface-Prospectivity Targets", show=False).add_to(m)
                HeatMap(high_priority_df[["latitude", "longitude"]].values.tolist(), radius=18, blur=22).add_to(heat_group)

    if settings.get("show_mori") and settings.get("mori_gdf") is not None:
        mori_gdf = settings["mori_gdf"]
        mori_group = folium.FeatureGroup(name="MORI Overlay (Exploratory Surface Proxy)", show=False).add_to(m)
        tooltip_fields = [c for c in ["MORI"] if c in mori_gdf.columns]
        folium.GeoJson(
            mori_gdf.__geo_interface__,
            style_function=lambda feat: {"fillColor": "#7bccc4", "color": "#0868ac", "weight": 1, "fillOpacity": 0.4},
            tooltip=folium.GeoJsonTooltip(fields=tooltip_fields) if tooltip_fields else None,
        ).add_to(mori_group)

    if settings.get("show_lineaments") and settings.get("lineaments_gdf") is not None:
        line_group = folium.FeatureGroup(name="Lineaments (User-Supplied Context Layer)", show=False).add_to(m)
        folium.GeoJson(settings["lineaments_gdf"].__geo_interface__,
                        style_function=lambda feat: {"color": "#555555", "weight": 2, "dashArray": "5,5"}).add_to(line_group)

    if settings.get("show_geology") and settings.get("geology_gdf") is not None:
        geo_group = folium.FeatureGroup(name="Geology Boundaries (User-Supplied Context Layer)", show=False).add_to(m)
        folium.GeoJson(settings["geology_gdf"].__geo_interface__,
                        style_function=lambda feat: {"color": "#8a6d3b", "weight": 2, "fillOpacity": 0.05, "dashArray": "3,3"}).add_to(geo_group)

    folium.LayerControl(collapsed=False).add_to(m)
    Fullscreen().add_to(m)
    MiniMap(toggle_display=True, minimized=True).add_to(m)
    m.get_root().html.add_child(folium.Element(build_legend_html()))
    return m


# ============================================================================
# PAGE BODY
# ============================================================================
df, report = load_predictions(str(PREDICTION_FILE_PATH))
using_demo = False

if report.get("file_missing") or df is None or (df is not None and df.empty):
    using_demo = True
    df = make_demo_predictions()
    report = {"file_missing": False, "dropped_invalid_coords": 0, "dropped_invalid_score": 0}
    st.info(
        f"No real prediction file found at `{PREDICTION_FILE_PATH}` yet, so this page is "
        "showing a **synthetic demo dataset** (clearly labelled `validation_status: SYNTHETIC DEMO` "
        "in every popup and export row) so you can explore the interface. Run the modeling "
        "pipeline (`train_baseline_rf.py` or `train_model.py`) to replace it with real scored output."
    )

if report.get("dropped_invalid_coords", 0) or report.get("dropped_invalid_score", 0):
    st.sidebar.caption(
        f"Data cleaning: removed {report.get('dropped_invalid_coords', 0)} row(s) with invalid "
        f"coordinates and {report.get('dropped_invalid_score', 0)} row(s) with invalid scores."
    )

st.sidebar.header("Controls")
region_options = list(REGION_CENTERS.keys())
selected_region = st.sidebar.selectbox("Target Region", region_options, index=0)

region_filtering_available = "region_name" in df.columns
region_df = df.loc[df["region_name"] == selected_region].copy() if region_filtering_available else df.copy()

threshold = st.sidebar.slider("Minimum Prospectivity Score", min_value=MIN_THRESHOLD, max_value=MAX_THRESHOLD,
                               value=DEFAULT_THRESHOLD, step=THRESHOLD_STEP)

filtered = region_df.loc[region_df["prospectivity_score"] >= threshold].copy()
if not filtered.empty:
    filtered["priority_category"] = filtered["prospectivity_score"].apply(classify_priority)

if not region_df.empty and filtered.empty:
    st.info(f"No targets meet the selected threshold of {threshold:.2f}.")

st.sidebar.subheader("Layers")
show_points = st.sidebar.checkbox("Show candidate target points", value=True)
show_hotspots = st.sidebar.checkbox("Show high-priority hotspots", value=True)

mori_gdf, _ = load_optional_geojson(str(MORI_OVERLAY_GEOJSON_PATH))
show_mori = st.sidebar.checkbox("Show MORI overlay", value=False, disabled=(mori_gdf is None))
lineaments_gdf, _ = load_optional_geojson(str(LINEAMENTS_GEOJSON_PATH))
show_lineaments = st.sidebar.checkbox("Show lineaments", value=False, disabled=(lineaments_gdf is None))
geology_gdf, _ = load_optional_geojson(str(GEOLOGY_BOUNDARIES_PATH))
show_geology = st.sidebar.checkbox("Show geology boundaries", value=False, disabled=(geology_gdf is None))

st.sidebar.subheader("Basemap")
basemap = st.sidebar.selectbox("Basemap", MAP_TILE_OPTIONS, index=0)

st.subheader("Summary")
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Selected Region", selected_region if region_filtering_available else "All")
k2.metric("Total Available Targets", int(len(region_df)))
k3.metric("Above Threshold", int(len(filtered)))
high_priority_count = int((filtered["priority_category"] == "High Priority").sum()) if not filtered.empty else 0
k4.metric("High-Priority (> 0.85)", high_priority_count)
if "validation_status" in df.columns and not filtered.empty and not using_demo:
    pct_validated = 100.0 * (filtered["validation_status"] == "Geologically reviewed").sum() / len(filtered)
    k5.metric("Field-Validated %", f"{pct_validated:.1f}%")
else:
    k5.metric("Field-Validated %", "Demo data" if using_demo else "N/A")

metadata = load_model_metadata(str(MODEL_METADATA_PATH))
if metadata is not None:
    with st.expander("Model Performance (from model_metadata.json)"):
        mv = metadata.get("mean_validation_metrics", {})
        m1, m2, m3 = st.columns(3)
        m1.metric("Pooled OOF ROC-AUC", format_value_safe(mv.get("pooled_oof_roc_auc"), decimals=3))
        m2.metric("Pooled OOF PR-AUC", format_value_safe(mv.get("pooled_oof_pr_auc"), decimals=3))
        m3.metric("CV Method", metadata.get("cross_validation_method", "Not available"))
        st.caption(metadata.get("scientific_limitation_statement", ""))

st.subheader("Interactive Prospectivity Map")
if filtered.empty:
    st.warning("No targets to display on the map for the current region/threshold selection.")
else:
    if len(filtered) > MAX_MAP_MARKERS:
        st.info(f"Displaying the highest-scoring {MAX_MAP_MARKERS} of {len(filtered)} targets.")
    settings = {
        "basemap": basemap, "show_points": show_points, "show_hotspots": show_hotspots,
        "show_mori": show_mori, "show_lineaments": show_lineaments, "show_geology": show_geology,
        "mori_gdf": mori_gdf if show_mori else None,
        "lineaments_gdf": lineaments_gdf if show_lineaments else None,
        "geology_gdf": geology_gdf if show_geology else None,
    }
    prospectivity_map = create_prospectivity_map(filtered, selected_region, settings)
    st_folium(prospectivity_map, use_container_width=True, height=600, key="mmes_map", returned_objects=[])

st.subheader("Supporting Analytics")
if filtered.empty:
    st.info("No data available for analytics at the current filter settings.")
else:
    a1, a2 = st.columns(2)
    with a1:
        fig_hist = px.histogram(filtered, x="prospectivity_score", nbins=20,
                                 title="Distribution of Prospectivity Scores",
                                 color_discrete_sequence=["#8B4A2B"])
        fig_hist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_hist, width="stretch")
    with a2:
        counts = filtered["priority_category"].value_counts().reset_index()
        counts.columns = ["priority_category", "count"]
        fig_bar = px.bar(counts, x="priority_category", y="count", color="priority_category",
                          color_discrete_map=PRIORITY_COLORS, title="Targets by Priority Category")
        fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_bar, width="stretch")

    if "Slope" in filtered.columns and filtered["Slope"].notna().any():
        hover_cols = ["target_id"] + [c for c in ["MORI", "HAI", "FMI"] if c in filtered.columns]
        fig_scatter = px.scatter(filtered, x="Slope", y="prospectivity_score", color="priority_category",
                                  color_discrete_map=PRIORITY_COLORS, hover_data=hover_cols,
                                  title="Slope vs. Prospectivity Score")
        fig_scatter.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_scatter, width="stretch")

    with st.expander("Feature Summary Table (Filtered Targets)"):
        display_cols = [c for c in EXPORT_COLUMN_ORDER if c in filtered.columns]
        st.dataframe(filtered[display_cols], width="stretch")

st.subheader("Field Export")
if filtered.empty:
    st.info("No filtered targets available to export.")
else:
    csv_name, geojson_name = generate_export_filenames(selected_region, threshold)
    export_cols = [c for c in EXPORT_COLUMN_ORDER if c in filtered.columns]
    export_df = filtered[export_cols].copy()
    export_df["recommended_action"] = EXPORT_RECOMMENDED_ACTION
    csv_bytes = export_df.to_csv(index=False).encode("utf-8")
    geojson_bytes = json.dumps(build_point_geojson_feature_collection(filtered), indent=2).encode("utf-8")

    e1, e2 = st.columns(2)
    e1.download_button("Download CSV", data=csv_bytes, file_name=csv_name, mime="text/csv")
    e2.download_button("Download GeoJSON", data=geojson_bytes, file_name=geojson_name, mime="application/geo+json")
    st.caption(f"Exports contain exactly the {len(filtered)} currently filtered record(s)."
               + (" DEMO DATA -- do not use for field planning." if using_demo else ""))

with st.expander("Model Scope and Responsible Use"):
    st.markdown(
        "- This system ranks **relative surface prospectivity**; it does not measure or detect underground reserves.\n"
        "- Satellite imagery cannot see underground; every score is derived from surface reflectance and terrain context only.\n"
        "- Vegetation, soil composition, atmospheric effects, shadow, and mixed pixels can all affect the spectral signals.\n"
        "- Scores require review against geological maps, known occurrences, field observations, and lab assays.\n"
        "- A high score is a **prioritization signal for field investigation**, not a mining or investment recommendation."
    )
