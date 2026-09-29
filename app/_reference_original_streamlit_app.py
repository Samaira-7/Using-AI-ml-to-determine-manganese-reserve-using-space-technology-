"""
streamlit_app.py

Minimal interactive dashboard: shows a prospectivity heatmap over an
India basemap, with click-to-inspect once you have predictions in
data/processed/prospectivity_score.tif.

Run with: streamlit run app/streamlit_app.py

Deployment note: Streamlit Community Cloud (share.streamlit.io, free,
connects straight to this GitHub repo) is the fastest way to get a live
link for judging. Its free tier caps out around 1 GB memory and sleeps
apps after ~12 quiet hours - keep the deployed raster small/downsampled
rather than shipping full-resolution scenes through it.
"""

import numpy as np
import streamlit as st
import folium
from streamlit_folium import st_folium
import rasterio
from rasterio.warp import transform_bounds

st.set_page_config(page_title="Manganese prospectivity - India", layout="wide")
st.title("Manganese prospectivity mapping")
st.caption(
    "AI/ML-derived prospectivity surface from ASTER, PRISMA and Landsat "
    "spectral indices plus geological priors. Hackathon prototype, not a "
    "certified resource estimate."
)

RASTER_PATH = "data/processed/prospectivity_score.tif"


@st.cache_data
def load_prospectivity_raster(path):
    with rasterio.open(path) as src:
        arr = src.read(1)
        bounds = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
    arr = np.nan_to_num(arr, nan=0.0)
    return arr, bounds


try:
    arr, bounds = load_prospectivity_raster(RASTER_PATH)
    center_lat = (bounds[1] + bounds[3]) / 2
    center_lon = (bounds[0] + bounds[2]) / 2
except (FileNotFoundError, rasterio.errors.RasterioIOError):
    st.warning(
        f"No prospectivity raster found at {RASTER_PATH} yet - showing a "
        "blank India-centered map. Run the modeling pipeline first."
    )
    arr, bounds = None, None
    center_lat, center_lon = 22.0, 82.0  # roughly central India

m = folium.Map(location=[center_lat, center_lon], zoom_start=6, tiles="CartoDB positron")

if arr is not None:
    folium.raster_layers.ImageOverlay(
        image=arr,
        bounds=[[bounds[1], bounds[0]], [bounds[3], bounds[2]]],
        colormap=lambda x: (1, 0, 0, x),
        opacity=0.6,
        name="Prospectivity score",
    ).add_to(m)

folium.LayerControl().add_to(m)

col1, col2 = st.columns([3, 1])
with col1:
    st_data = st_folium(m, width=900, height=600)
with col2:
    st.subheader("Selected point")
    if st_data and st_data.get("last_clicked"):
        lat = st_data["last_clicked"]["lat"]
        lon = st_data["last_clicked"]["lng"]
        st.write(f"Lat: {lat:.4f}, Lon: {lon:.4f}")
        st.info(
            "Wire this up to sample your feature stack + model at this "
            "coordinate to show a live probability score and the "
            "spectral signature driving it."
        )
    else:
        st.write("Click anywhere on the map to inspect that location.")
