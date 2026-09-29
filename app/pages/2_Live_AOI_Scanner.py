#!/usr/bin/env python3
"""
2_Live_AOI_Scanner.py

NEW page. On-demand band-ratio screening for a user-drawn AOI, run
against REAL Earth Engine imagery -- the closest thing to "real-time
detection" this pipeline can honestly deliver.

IMPORTANT, READ BEFORE EDITING THIS FILE:
No satellite performs onboard mineral detection. What this page actually
does is: take an AOI you specify, pull the most recent cloud-free
Landsat/Sentinel-2 composite for it via Earth Engine, compute the same
band-ratio indices as spectral_indices.py, and render a prospectivity
proxy layer -- typically in seconds to a couple of minutes depending on
AOI size and Earth Engine load. That is "near real-time analysis of
existing imagery," not "a camera in space that recognises manganese."
Never rephrase this page's copy to imply otherwise.

Requires a working Earth Engine session (ee.Authenticate() completed at
least once on this machine, plus a registered Cloud project). If that
isn't set up yet, the page detects it, explains exactly what's missing,
and does NOT fall back to fabricated results.
"""

import sys
from pathlib import Path

import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from theme import apply_theme, top_bar, scope_banner, status_pill

apply_theme("M-MES | Live AOI Scanner")
top_bar("New &middot; On-demand Earth Engine scan")
scope_banner(
    "This tool re-runs real band-ratio math against the freshest available "
    "satellite composite for the AOI you choose. It is <b>near-real-time "
    "analysis of existing imagery</b>, not live sensing from orbit &mdash; "
    "no satellite detects minerals onboard."
)

st.title("Live AOI band-ratio scan")
st.caption(
    "Pull a fresh Landsat/Sentinel-2 composite for any bounding box and screen it with "
    "MORI / HAI / FMI on demand."
)

# ----------------------------------------------------------------------------
# Earth Engine availability check -- never fake a result if this fails.
# ----------------------------------------------------------------------------
EE_READY = False
EE_ERROR = None
try:
    import ee
    try:
        ee.Initialize()
        EE_READY = True
    except Exception as init_err:
        EE_ERROR = str(init_err)
except ImportError:
    EE_ERROR = "The earthengine-api package is not installed in this environment."

col_status, col_help = st.columns([1, 2])
with col_status:
    if EE_READY:
        st.markdown(status_pill("EARTH ENGINE: CONNECTED", "real"), unsafe_allow_html=True)
    else:
        st.markdown(status_pill("EARTH ENGINE: NOT CONNECTED", "sim"), unsafe_allow_html=True)
with col_help:
    if not EE_READY:
        st.caption(f"Reason: {EE_ERROR}")

if not EE_READY:
    st.warning(
        "This page needs your own authenticated Earth Engine session to pull real imagery. "
        "**Nothing below will run on fabricated data** -- set up Earth Engine first:"
    )
    st.markdown(
        """
        1. `pip install earthengine-api` (already in `environment.yml`)
        2. Register a free Cloud project at
           [code.earthengine.google.com/register](https://code.earthengine.google.com/register)
           (choose "Unpaid usage")
        3. In a terminal: `earthengine authenticate` (one-time per machine)
        4. Reload this page
        """
    )
    st.info(
        "Once connected, the form below lets you draw a bounding box, pick a date range, "
        "and get back real MORI/HAI/FMI/NDVI rasters for that exact AOI within roughly "
        "a minute for a small area."
    )
    st.stop()

# ----------------------------------------------------------------------------
# AOI + date controls
# ----------------------------------------------------------------------------
st.subheader("1. Define the area of interest")

preset = st.selectbox(
    "Start from a known belt, or enter custom coordinates below",
    ["Custom", "Balaghat Belt (MP)", "Keonjhar Belt (Odisha)", "Bonai-Keonjhar-Sundargarh (Odisha)"],
)
PRESETS = {
    "Balaghat Belt (MP)": (79.80, 21.70, 80.30, 22.10),
    "Keonjhar Belt (Odisha)": (85.30, 21.40, 85.90, 21.90),
    "Bonai-Keonjhar-Sundargarh (Odisha)": (84.8, 21.5, 85.8, 22.3),
    "Custom": (79.80, 21.70, 80.30, 22.10),
}
default_bbox = PRESETS[preset]

c1, c2, c3, c4 = st.columns(4)
min_lon = c1.number_input("Min longitude", value=float(default_bbox[0]), format="%.4f")
min_lat = c2.number_input("Min latitude", value=float(default_bbox[1]), format="%.4f")
max_lon = c3.number_input("Max longitude", value=float(default_bbox[2]), format="%.4f")
max_lat = c4.number_input("Max latitude", value=float(default_bbox[3]), format="%.4f")

aoi_area_deg2 = abs(max_lon - min_lon) * abs(max_lat - min_lat)
if aoi_area_deg2 > 2.0:
    st.error("AOI is large (>2 sq. degrees). Earth Engine will likely time out for an on-demand scan this size — shrink the box.")
elif aoi_area_deg2 <= 0:
    st.error("Invalid AOI — max longitude/latitude must exceed min longitude/latitude.")

st.subheader("2. Choose a date range")
d1, d2 = st.columns(2)
start_date = d1.text_input("Start date (YYYY-MM-DD)", value="2025-01-01")
end_date = d2.text_input("End date (YYYY-MM-DD)", value="2025-12-31")
cloud_threshold = st.slider("Max cloud cover (%)", 0, 80, 30)

run_scan = st.button("Run live scan", type="primary", disabled=(aoi_area_deg2 <= 0 or aoi_area_deg2 > 2.0))

# ----------------------------------------------------------------------------
# The actual Earth Engine call -- extends fetch_landsat_gee.py /
# spectral_indices.py logic directly, run synchronously for this AOI.
# ----------------------------------------------------------------------------
def mask_landsat_clouds(image):
    qa = image.select("QA_PIXEL")
    cloud_bit = 1 << 3
    cloud_shadow_bit = 1 << 4
    mask = qa.bitwiseAnd(cloud_bit).eq(0).And(qa.bitwiseAnd(cloud_shadow_bit).eq(0))
    return image.updateMask(mask)


def scale_landsat(image):
    optical = image.select("SR_B.").multiply(0.0000275).add(-0.2)
    return image.addBands(optical, None, True)


@st.cache_data(show_spinner=False, ttl=3600)
def run_live_scan(min_lon, min_lat, max_lon, max_lat, start_date, end_date, cloud_threshold):
    """Runs a real Earth Engine query and returns a small results dict.
    Raises on any Earth Engine error rather than returning fake numbers."""
    aoi = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])

    def _collection(collection_id):
        return (
            ee.ImageCollection(collection_id)
            .filterBounds(aoi)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt("CLOUD_COVER", cloud_threshold))
            .map(mask_landsat_clouds)
            .map(scale_landsat)
        )

    merged = _collection("LANDSAT/LC08/C02/T1_L2").merge(_collection("LANDSAT/LC09/C02/T1_L2"))
    scene_count = merged.size().getInfo()
    if scene_count == 0:
        return {"scene_count": 0}

    composite = merged.median().clip(aoi)

    red = composite.select("SR_B4")
    blue = composite.select("SR_B2")
    nir = composite.select("SR_B5")
    swir1 = composite.select("SR_B6")
    swir2 = composite.select("SR_B7")

    eps = 1e-6
    mori = red.divide(blue.add(eps)).rename("MORI")
    hai = swir1.divide(swir2.add(eps)).rename("HAI")
    fmi = swir1.divide(nir.add(eps)).rename("FMI")
    ndvi = nir.subtract(red).divide(nir.add(red).add(eps)).rename("NDVI")

    stack = mori.addBands([hai, fmi, ndvi])
    stats = stack.reduceRegion(
        reducer=ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True).combine(
            ee.Reducer.minMax(), sharedInputs=True
        ),
        geometry=aoi, scale=30, maxPixels=1e9, bestEffort=True,
    ).getInfo()

    thumb_url = stack.select("MORI").getThumbURL({
        "min": 0.5, "max": 2.5, "palette": ["#1A1D17", "#8B4A2B", "#C9A961"],
        "region": aoi, "dimensions": 512,
    })

    return {"scene_count": scene_count, "stats": stats, "thumb_url": thumb_url, "aoi": [min_lon, min_lat, max_lon, max_lat]}


if run_scan:
    with st.spinner("Querying Earth Engine for this AOI — this calls the live API, not cached demo data..."):
        try:
            result = run_live_scan(min_lon, min_lat, max_lon, max_lat, start_date, end_date, cloud_threshold)
        except Exception as scan_error:
            st.error(
                "The Earth Engine call failed. This page never substitutes fabricated numbers "
                f"for a failed real query. Error detail: `{scan_error}`"
            )
            st.stop()

    if result["scene_count"] == 0:
        st.warning(
            "Zero cloud-free scenes matched this AOI, date range and cloud threshold. "
            "Widen the date range or raise the cloud-cover limit and try again."
        )
        st.stop()

    st.success(f"Live scan complete — {result['scene_count']} Landsat 8/9 scene(s) composited for this AOI.")

    st.subheader("3. Result — real, freshly computed band ratios")
    st.image(result["thumb_url"], caption="MORI (Manganese Oxide Ratio Index) over the requested AOI — cool to warm = low to high ratio.")

    stats = result["stats"]
    cols = st.columns(4)
    for i, band in enumerate(["MORI", "HAI", "FMI", "NDVI"]):
        mean_val = stats.get(f"{band}_mean")
        with cols[i]:
            st.metric(f"{band} mean", f"{mean_val:.3f}" if mean_val is not None else "n/a")
            std_val = stats.get(f"{band}_stdDev")
            if std_val is not None:
                st.caption(f"&plusmn; {std_val:.3f} std. dev. across AOI")

    st.markdown(
        status_pill("REAL EARTH ENGINE QUERY", "real")
        + f" &nbsp; scene count: {result['scene_count']} &nbsp;&middot;&nbsp; scale: 30m &nbsp;&middot;&nbsp; "
        f"AOI: {result['aoi']}",
        unsafe_allow_html=True,
    )

    st.info(
        "This is a screening layer, not a detection result. High MORI values correlate with "
        "iron/manganese-oxide-rich surfaces in general — confirm against GSI lithology maps "
        "and known occurrences before treating any pixel as noteworthy."
    )

    with st.expander("What happened technically, step by step"):
        st.markdown(
            f"""
            1. Queried `LANDSAT/LC08/C02/T1_L2` and `LANDSAT/LC09/C02/T1_L2` for your AOI, `{start_date}` to `{end_date}`
            2. Filtered to scenes under {cloud_threshold}% cloud cover, masked remaining cloud/shadow pixels via `QA_PIXEL`
            3. Applied Collection 2 scale factors to get true surface reflectance
            4. Took the pixel-wise median across all matching scenes (your composite)
            5. Computed `MORI = Red/Blue`, `HAI = SWIR1/SWIR2`, `FMI = SWIR1/NIR`, `NDVI = (NIR-Red)/(NIR+Red)` per pixel
            6. Reduced to AOI-wide mean/std/min/max, and rendered a false-colour MORI thumbnail — all server-side in Earth Engine, nothing downloaded or fabricated locally
            """
        )
else:
    st.caption("Set an AOI and date range above, then click **Run live scan**.")

st.divider()
st.subheader("From single scan to continuous monitoring")
st.markdown(
    """
    This on-demand scan is the manual version of what a **scheduled pipeline**
    would do automatically: re-run this same query on a timer (e.g. whenever
    a new cloud-free Sentinel-2 scene lands, roughly every 5 days for a given
    tile) and diff the new prospectivity layer against the previous one to
    flag areas of spectral change. That's the honest version of
    "continuous mineral mapping" &mdash; scheduled reprocessing of new
    imagery, not live onboard detection. See the roadmap on the project
    site for what that would take to stand up (Cloud Functions + Earth
    Engine batch export + a diffing job).
    """
)
