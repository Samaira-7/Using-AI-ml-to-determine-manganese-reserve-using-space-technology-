#!/usr/bin/env python3
"""
gee_feature_pipeline.py

M-MES: Manganese Mineral Exploration System
Surface Mineral Prospectivity Mapping Pipeline (Google Earth Engine)
Pilot region: Balaghat Mining Belt, Madhya Pradesh, India

============================================================================
SCOPE AND SCIENTIFIC LIMITATIONS (READ FIRST)
============================================================================
Satellite optical imagery (Sentinel-2) measures reflected light from the
top few micrometres to millimetres of the Earth's SURFACE. It cannot see
underground, and this script cannot and does not:
    - detect underground manganese ore bodies
    - measure ore grade, tonnage, depth, or continuity
    - confirm economically viable deposits
    - replace geological mapping, geochemistry, geophysics, or drilling

This script instead produces a SURFACE MINERAL PROSPECTIVITY SCREENING
layer, built from exposed-surface spectral ratios (iron/manganese-oxide
associated responses, hydroxyl/alteration proxies), vegetation masking,
and terrain context (elevation/slope/aspect). All outputs are exploratory
proxies intended ONLY to help prioritise areas for field-based follow-up
investigation, and must be validated against independent ground truth
(Geological Survey of India records, field mapping, assay data) before
any exploration or investment decision is made.

============================================================================
DEPENDENCIES (install once, in your local environment / virtualenv)
============================================================================
    pip install earthengine-api geemap pandas

============================================================================
USAGE
============================================================================
    python gee_feature_pipeline.py

On first run you will be prompted to authenticate with Google Earth Engine
(a browser window will open). See the accompanying usage instructions for
details on AOI/date configuration and supplying real labelled samples.
"""

import sys
import logging

import ee
import geemap
import pandas as pd


# ============================================================================
# 2. CONFIGURATION
# ============================================================================
# --- Study area identity ---------------------------------------------------
STUDY_AREA_NAME = "Balaghat_Manganese_Belt"

# --- AOI bounding box (EPSG:4326, decimal degrees) --------------------------
AOI_LON_MIN = 79.80
AOI_LAT_MIN = 21.70
AOI_LON_MAX = 80.30
AOI_LAT_MAX = 22.10

# --- Date range --------------------------------------------------------------
# Configurable. Do NOT assume Sentinel-2 imagery exists for dates beyond
# END_DATE at the time you run this script; the script checks scene
# availability at runtime instead of assuming it.
START_DATE = "2025-01-01"
END_DATE = "2026-08-31"

# --- Cloud filtering ---------------------------------------------------------
CLOUD_PERCENTAGE_THRESHOLD = 20  # CLOUDY_PIXEL_PERCENTAGE metadata, in percent

# --- Vegetation / surface-exposure threshold --------------------------------
NDVI_VEGETATION_THRESHOLD = 0.30

# --- Export scale ------------------------------------------------------------
# Sentinel-2 bands B8A, B11, and B12 (all used in the mineral-relevant
# ratios MORI and HAI) are NATIVE 20 m bands, not 10 m. Exporting the
# combined feature stack at 10 m would falsely imply 10 m spatial precision
# for the mineral-relevant SWIR information it does not have. The full
# stack is therefore produced and exported at 20 m.
EXPORT_SCALE_METERS = 20

# --- Google Drive export destination -----------------------------------------
EXPORT_FOLDER = "MMES_Exports"
EXPORT_PREFIX = "MMES_Balaghat"

# --- Sampling ------------------------------------------------------------
MAX_SAMPLES = 2000
RANDOM_SEED = 42

# --- Optional raster export toggle -------------------------------------------
# Section 11 requirement: GeoTIFF export must NOT start automatically.
ENABLE_RASTER_EXPORT = False

# --- Earth Engine Cloud Project ----------------------------------------------
# Earth Engine requires API calls to be associated with a Google Cloud
# project that has the Earth Engine API enabled. Set this to your own
# registered project ID (see usage instructions). Leave as None to let
# ee.Initialize() use its default resolution behaviour.
EE_PROJECT_ID = None  # e.g. "my-gcp-project-id"

# --- Numerical safety ---------------------------------------------------------
# Small constant added ONLY to denominators in ratio calculations, to avoid
# divide-by-zero errors. Not applied to numerators.
EPSILON = 1e-6

# ============================================================================
# Logging (no silent failures)
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("MMES")

# ============================================================================
# Band constants
# ============================================================================
# Bands read from COPERNICUS/S2_SR_HARMONIZED before scaling/masking.
RAW_S2_BANDS = ["B2", "B3", "B4", "B8", "B8A", "B11", "B12", "QA60"]

# Optical reflectance bands after cloud masking and /10000 scaling.
SCALED_OPTICAL_BANDS = ["B2", "B3", "B4", "B8", "B8A", "B11", "B12"]

# Exact required final feature-stack band order (Section 9).
FEATURE_BAND_ORDER = [
    "B2", "B3", "B4", "B8", "B8A", "B11", "B12",
    "MORI", "HAI", "FMI", "NDVI",
    "Elevation", "Slope", "Aspect",
]

# CSV export field order (unlabeled).
CSV_EXPORT_FIELDS = ["longitude", "latitude"] + FEATURE_BAND_ORDER


# ============================================================================
# 1. EARTH ENGINE INITIALIZATION
# ============================================================================
def initialize_earth_engine():
    """
    Initialize the Earth Engine API.

    ee.Authenticate() is invoked ONLY if ee.Initialize() fails (e.g. no
    valid local credentials yet). Prints a clear message and exits if
    authentication/initialization ultimately cannot be completed, rather
    than failing silently later on the first API call.
    """
    try:
        if EE_PROJECT_ID:
            ee.Initialize(project=EE_PROJECT_ID)
        else:
            ee.Initialize()
        logger.info("Earth Engine initialized using existing credentials.")
        return
    except Exception as init_error:
        logger.warning(
            f"ee.Initialize() failed on first attempt ({init_error}). "
            "Attempting ee.Authenticate() ..."
        )

    try:
        ee.Authenticate()
        if EE_PROJECT_ID:
            ee.Initialize(project=EE_PROJECT_ID)
        else:
            ee.Initialize()
        logger.info("Earth Engine authenticated and initialized.")
    except Exception as auth_error:
        logger.error(
            "Earth Engine authentication/initialization FAILED. "
            "This script cannot continue without a working Earth Engine "
            "session. Common fixes: run 'earthengine authenticate' from a "
            "terminal, verify you have a Google Cloud project registered "
            "for Earth Engine access, and set EE_PROJECT_ID in the "
            f"CONFIGURATION section. Details: {auth_error}"
        )
        sys.exit(1)


# ============================================================================
# AOI construction
# ============================================================================
def build_aoi():
    """Build the AOI rectangle geometry (EPSG:4326) from the configured bbox."""
    aoi = ee.Geometry.Rectangle(
        [AOI_LON_MIN, AOI_LAT_MIN, AOI_LON_MAX, AOI_LAT_MAX],
        proj="EPSG:4326",
        geodesic=False,
    )
    return aoi


# ============================================================================
# Shared band-presence check (used at multiple pipeline stages)
# ============================================================================
def check_required_bands(image, required_bands, context=""):
    """
    Verify that all required bands are present on `image` before proceeding.
    Raises RuntimeError with the missing band names rather than letting a
    downstream .select() or .expression() call fail with a less useful error.
    """
    band_names = image.bandNames().getInfo()
    missing = [b for b in required_bands if b not in band_names]
    if missing:
        raise RuntimeError(
            f"Missing required bands {missing}"
            f"{(' during ' + context) if context else ''}. "
            f"Bands actually present: {band_names}"
        )
    logger.info(
        f"Band presence check passed{(' (' + context + ')') if context else ''}: "
        f"{required_bands}"
    )


# ============================================================================
# 4. CLOUD AND CIRRUS MASKING (QA60)
# ============================================================================
def mask_s2_clouds_qa60(image):
    """
    Baseline cloud/cirrus masking using the Sentinel-2 QA60 bitmask, applied
    to a single COPERNICUS/S2_SR_HARMONIZED image.

    Bit 10 = opaque clouds, Bit 11 = cirrus clouds (both masked out).

    Also applies the correct surface-reflectance scaling (DN / 10000.0) to
    the optical bands BEFORE any downstream ratio/index calculation, and
    preserves the original image's properties (acquisition time, IDs, etc.)
    on the returned image.

    LIMITATION: QA60 is a coarse, scene-level quality band. It is a
    BASELINE masking approach only and does not reliably capture all cloud
    edges, thin haze, cloud shadow, or other atmospheric artefacts. A
    production version of this pipeline should be supplemented with the
    COPERNICUS/S2_CLOUD_PROBABILITY collection (s2cloudless-style
    probability masking) for more robust cloud/shadow handling.
    """
    raw = image.select(RAW_S2_BANDS)
    qa60 = raw.select("QA60")

    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    clear_mask = (
        qa60.bitwiseAnd(cloud_bit_mask).eq(0)
        .And(qa60.bitwiseAnd(cirrus_bit_mask).eq(0))
    )

    # Scale optical bands to physical reflectance (0-1 range) BEFORE any
    # ratio/index math is performed anywhere downstream.
    scaled_optical = raw.select(SCALED_OPTICAL_BANDS).divide(10000.0)

    masked = scaled_optical.updateMask(clear_mask)

    return masked.copyProperties(image, image.propertyNames())


# ============================================================================
# 5. IMAGE COMPOSITING
# ============================================================================
def build_s2_composite(aoi):
    """
    Filter COPERNICUS/S2_SR_HARMONIZED by AOI, date range, and
    CLOUDY_PIXEL_PERCENTAGE, apply QA60 cloud/cirrus masking + scaling, and
    produce a median composite clipped to the AOI.

    Raises RuntimeError with a clear message if no scenes match the filters
    (empty collection), rather than silently returning an empty/invalid
    composite.

    Returns (composite_image, scene_count).

    NOTE: a median composite is used because it suppresses residual cloud
    contamination and temporal noise relative to a single scene. It may,
    however, blur short-lived surface conditions (e.g. recent disturbance,
    soil moisture changes) and is NOT a substitute for a rigorous,
    season-aware compositing strategy in a production system.
    """
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(START_DATE, END_DATE)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", CLOUD_PERCENTAGE_THRESHOLD))
    )

    scene_count = collection.size().getInfo()

    logger.info(f"AOI: {STUDY_AREA_NAME}")
    logger.info(f"Time period requested: {START_DATE} to {END_DATE}")
    logger.info(f"Cloud threshold (CLOUDY_PIXEL_PERCENTAGE <=): {CLOUD_PERCENTAGE_THRESHOLD}")
    logger.info(f"Sentinel-2 scenes matching filters: {scene_count}")

    if scene_count == 0:
        raise RuntimeError(
            "No Sentinel-2 scenes found for the configured AOI, date range, "
            "and cloud threshold. This is an empty ImageCollection and "
            "cannot be composited. Try: widening START_DATE/END_DATE, "
            "raising CLOUD_PERCENTAGE_THRESHOLD, or double-checking the AOI "
            "bounding-box coordinates."
        )

    masked_collection = collection.map(mask_s2_clouds_qa60)
    composite = masked_collection.median().clip(aoi)

    return composite, scene_count


# ============================================================================
# 6. SPECTRAL FEATURE ENGINEERING
# ============================================================================
def add_spectral_indices(image):
    """
    Add MORI, HAI, FMI, and NDVI bands to a scaled Sentinel-2 reflectance
    image (bands already in 0-1 reflectance units, cloud-masked).

    All four indices are exploratory SURFACE spectral proxies derived from
    band ratios. None of them constitute a direct or unique measurement of
    manganese concentration; all require field or geological validation.
    """
    b2 = image.select("B2")
    b4 = image.select("B4")
    b8a = image.select("B8A")
    b11 = image.select("B11")
    b12 = image.select("B12")

    # --- MORI: Manganese Oxide Ratio Index = (B11 / B12) * (B4 / B2) -------
    # Exploratory proxy combining SWIR contrast (B11/B12) and visible-band
    # contrast (B4/B2). May help highlight surface materials or alteration
    # conditions associated with manganese-bearing environments, but it is
    # NOT a direct manganese concentration measurement and must be
    # validated using field or geological evidence.
    mori = image.expression(
        "(B11 / (B12 + EPS)) * (B4 / (B2 + EPS))",
        {"B11": b11, "B12": b12, "B4": b4, "B2": b2, "EPS": EPSILON},
    ).rename("MORI")

    # --- HAI: Hydroxyl / Alteration Index = B11 / B8A -----------------------
    # Broad SWIR-to-NIR contrast that may relate to alteration or hydrated
    # mineral assemblages. Sentinel-2's spectral resolution/bandwidth is
    # limited compared to dedicated mineral-mapping sensors (e.g. ASTER,
    # hyperspectral); HAI is therefore an alteration-SCREENING feature, not
    # proof of a specific mineral's presence.
    hai = image.expression(
        "B11 / (B8A + EPS)",
        {"B11": b11, "B8A": b8a, "EPS": EPSILON},
    ).rename("HAI")

    # --- FMI: Ferrous Mineral Index = B4 / B2 -------------------------------
    # Visible-band ratio that can help distinguish iron-related surface
    # colour responses (e.g. reddish/brownish oxidised surfaces). Must NOT
    # be interpreted as a unique indicator of manganese: iron oxides, bare
    # soil, shadow, and weathering crusts can all produce similar responses.
    fmi = image.expression(
        "B4 / (B2 + EPS)",
        {"B4": b4, "B2": b2, "EPS": EPSILON},
    ).rename("FMI")

    # --- NDVI = (B8 - B4) / (B8 + B4) ---------------------------------------
    # Standard vegetation index, used here to reduce vegetation influence:
    # dense vegetation obscures the exposed rock/soil spectral signals that
    # the other three indices depend on.
    ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")

    return image.addBands([mori, hai, fmi, ndvi])


# ============================================================================
# 7. VEGETATION / SURFACE-EXPOSURE MASKING
# ============================================================================
def apply_surface_exposure_mask(image):
    """
    Mask pixels where NDVI > NDVI_VEGETATION_THRESHOLD; retain pixels where
    NDVI <= NDVI_VEGETATION_THRESHOLD (i.e. relatively exposed surface).

    LIMITATIONS:
      - This is a conservative surface-exposure FILTER, not a validated
        vegetation classifier.
      - It reduces vegetation interference but may exclude legitimate
        mineralised terrain that lies beneath forest canopy.
      - The result is a SURFACE prospectivity screening layer. Areas that
        are masked out are NOT thereby asserted to lack manganese
        potential -- they are simply not assessable from this optical
        surface signal.
    """
    ndvi = image.select("NDVI")
    exposure_mask = ndvi.lte(NDVI_VEGETATION_THRESHOLD)
    return image.updateMask(exposure_mask)


# ============================================================================
# 8. DEM AND TERRAIN FEATURES (SRTM 30 m)
# ============================================================================
def add_terrain_features(image, aoi):
    """
    Add Elevation, Slope, and Aspect bands derived from USGS/SRTMGL1_003,
    the real 30 m SRTM dataset (NOT USGS/SRTM30_003, which is not the
    correct dataset for 30 m analysis).

    LIMITATIONS:
      - Elevation and slope provide terrain CONTEXT for interpretation
        only; neither proves mineral occurrence.
      - Aspect may help model local sun exposure or geomorphological
        patterns, but it is not, by itself, a causal mineral feature.
    """
    dem = ee.Image("USGS/SRTMGL1_003").clip(aoi)

    elevation = dem.select("elevation").rename("Elevation")
    slope = ee.Terrain.slope(dem).rename("Slope")
    aspect = ee.Terrain.aspect(dem).rename("Aspect")

    return image.addBands([elevation, slope, aspect])


# ============================================================================
# 9. FINAL FEATURE STACK ASSEMBLY
# ============================================================================
def build_feature_stacks(composite, aoi):
    """
    Build the two required feature stacks, both in the exact band order:
    B2, B3, B4, B8, B8A, B11, B12, MORI, HAI, FMI, NDVI,
    Elevation, Slope, Aspect

      1. diagnostic_stack             -- unmasked (by vegetation), for
                                          visualization/diagnostics.
      2. surface_prospectivity_stack  -- NDVI-exposure-masked, used for
                                          sample export.
    """
    with_indices = add_spectral_indices(composite)
    with_terrain = add_terrain_features(with_indices, aoi)

    diagnostic_stack = with_terrain.select(FEATURE_BAND_ORDER)

    surface_prospectivity_stack = (
        apply_surface_exposure_mask(with_terrain).select(FEATURE_BAND_ORDER)
    )

    logger.info(f"Feature stack band order: {FEATURE_BAND_ORDER}")

    return diagnostic_stack, surface_prospectivity_stack


# ============================================================================
# 10a. SAMPLE EXPORT -- OPTION A: RANDOM SURFACE SAMPLES (unlabeled)
# ============================================================================
def generate_random_samples(surface_prospectivity_stack, aoi):
    """
    Generate up to MAX_SAMPLES random samples from surface_prospectivity_stack
    within the AOI, with longitude/latitude attributes attached and
    RANDOM_SEED used for reproducibility. Pixels masked out (cloud or
    vegetation) are dropped rather than fabricated (dropNulls=True).
    """
    samples = surface_prospectivity_stack.sample(
        region=aoi,
        scale=EXPORT_SCALE_METERS,
        numPixels=MAX_SAMPLES,
        seed=RANDOM_SEED,
        geometries=True,
        dropNulls=True,
        tileScale=4,
    )

    def add_coords(feature):
        coords = feature.geometry().coordinates()
        return feature.set({
            "longitude": ee.Number(coords.get(0)),
            "latitude": ee.Number(coords.get(1)),
        })

    samples_with_coords = samples.map(add_coords)

    sample_count = samples_with_coords.size().getInfo()
    logger.info(f"Random surface samples generated: {sample_count}")

    if sample_count == 0:
        logger.warning(
            "Random sampling returned ZERO features. This can happen if "
            "the surface-exposure mask removes all pixels in the AOI (e.g. "
            "dense vegetation cover), or if the composite has no valid "
            "pixels after cloud masking. The exported CSV will be empty; "
            "no data was fabricated to fill it."
        )

    return samples_with_coords


def export_random_samples_to_drive(samples_fc):
    """Export the random (unlabeled) surface samples to Google Drive as CSV."""
    task = ee.batch.Export.table.toDrive(
        collection=samples_fc,
        description=f"{EXPORT_PREFIX}_random_samples",
        folder=EXPORT_FOLDER,
        fileNamePrefix=f"{EXPORT_PREFIX}_random_samples",
        fileFormat="CSV",
        selectors=CSV_EXPORT_FIELDS,
    )
    task.start()
    logger.info(
        f"Started export task '{EXPORT_PREFIX}_random_samples' "
        f"(Google Drive folder: {EXPORT_FOLDER}, task id: {task.id})"
    )
    return task


# ============================================================================
# 10b. SAMPLE EXPORT -- OPTION B: LABELLED TRAINING SAMPLES (placeholder)
# ============================================================================
def build_labelled_samples(surface_prospectivity_stack, labelled_fc=None):
    """
    PLACEHOLDER for labelled training-sample extraction.

    labelled_fc : ee.FeatureCollection or None
        A user-supplied FeatureCollection of REAL, documented ground-truth
        points or polygons. Each feature MUST carry an integer property
        named 'class_label':
            1 = verified manganese occurrence/prospect (e.g. from
                Geological Survey of India records, published prospect
                reports, or field-verified sampling)
            0 = documented non-mineralized/background location

    This function does NOT invent, guess, or randomly assign labels of any
    kind. If labelled_fc is None (the default), it returns None; the
    caller must then rely on the unlabeled random-sample export only, and
    this function prints an explicit warning that supervised
    classification cannot be scientifically validated without real ground
    truth.
    """
    if labelled_fc is None:
        logger.warning(
            "No labelled FeatureCollection was supplied to "
            "build_labelled_samples(). Supervised classification CANNOT be "
            "scientifically validated without real ground-truth labels. "
            "Falling back to the unlabeled random-sample export only. See "
            "usage instructions for how to supply real labelled occurrence "
            "data."
        )
        return None

    fc_size = labelled_fc.size().getInfo()
    if fc_size == 0:
        raise ValueError(
            "The supplied labelled_fc is empty (0 features). Provide at "
            "least one real, documented ground-truth feature with a "
            "'class_label' property."
        )

    property_names = labelled_fc.first().propertyNames().getInfo()
    if "class_label" not in property_names:
        raise ValueError(
            "The supplied labelled_fc is missing the required 'class_label' "
            "property. Each feature must have class_label = 1 (verified "
            "manganese occurrence/prospect) or class_label = 0 (documented "
            "non-mineralized/background location)."
        )

    sampled = surface_prospectivity_stack.sampleRegions(
        collection=labelled_fc,
        properties=["class_label"],
        scale=EXPORT_SCALE_METERS,
        geometries=True,
        tileScale=4,
    )

    def add_coords(feature):
        coords = feature.geometry().coordinates()
        return feature.set({
            "longitude": ee.Number(coords.get(0)),
            "latitude": ee.Number(coords.get(1)),
        })

    sampled_with_coords = sampled.map(add_coords)

    sample_count = sampled_with_coords.size().getInfo()
    logger.info(f"Labelled samples extracted: {sample_count} (of {fc_size} input features)")

    if sample_count == 0:
        logger.warning(
            "Labelled sampling returned ZERO features. Check that the "
            "labelled_fc geometries fall within the AOI and are not fully "
            "removed by the cloud mask or the surface-exposure mask."
        )

    return sampled_with_coords


def export_labelled_samples_to_drive(labelled_samples_fc):
    """Export real labelled training samples to Google Drive as CSV."""
    export_fields = CSV_EXPORT_FIELDS + ["class_label"]

    task = ee.batch.Export.table.toDrive(
        collection=labelled_samples_fc,
        description=f"{EXPORT_PREFIX}_labelled_samples",
        folder=EXPORT_FOLDER,
        fileNamePrefix=f"{EXPORT_PREFIX}_labelled_samples",
        fileFormat="CSV",
        selectors=export_fields,
    )
    task.start()
    logger.info(
        f"Started export task '{EXPORT_PREFIX}_labelled_samples' "
        f"(Google Drive folder: {EXPORT_FOLDER}, task id: {task.id})"
    )
    return task


# ============================================================================
# 11. OPTIONAL RASTER EXPORT (GeoTIFF) -- not auto-started
# ============================================================================
def export_raster_to_drive(surface_prospectivity_stack, aoi):
    """
    Export surface_prospectivity_stack as a GeoTIFF to Google Drive.
    Only invoked from main() if ENABLE_RASTER_EXPORT = True.
    """
    task = ee.batch.Export.image.toDrive(
        image=surface_prospectivity_stack,
        description=f"{EXPORT_PREFIX}_prospectivity_raster",
        folder=EXPORT_FOLDER,
        fileNamePrefix=f"{EXPORT_PREFIX}_prospectivity_raster",
        region=aoi,
        scale=EXPORT_SCALE_METERS,
        maxPixels=1e10,
        fileFormat="GeoTIFF",
    )
    task.start()
    logger.info(
        f"Started raster export task '{EXPORT_PREFIX}_prospectivity_raster' "
        f"(Google Drive folder: {EXPORT_FOLDER}, task id: {task.id})"
    )
    return task


# ============================================================================
# 12. MAP PREVIEW (geemap) -- optional, for interactive/Jupyter use
# ============================================================================
def build_preview_map(aoi, composite, diagnostic_stack):
    """
    Build an interactive geemap preview containing:
      - AOI boundary
      - Sentinel-2 true-colour composite (B4/B3/B2)
      - MORI layer, labelled "Exploratory Surface Prospectivity Proxy"
      - NDVI layer
      - Slope layer
      - Surface-exposure mask layer

    Colour palettes are chosen to avoid implying confirmed ore reserves.
    This map object is only meaningfully rendered in a Jupyter/Colab
    environment; in a plain terminal run it is built but not displayed.
    """
    m = geemap.Map()
    m.centerObject(aoi, zoom=10)

    m.addLayer(
        ee.Image().paint(aoi, 0, 2),
        {"palette": ["black"]},
        f"{STUDY_AREA_NAME} AOI Boundary",
    )

    m.addLayer(
        composite,
        {"bands": ["B4", "B3", "B2"], "min": 0.0, "max": 0.3},
        "Sentinel-2 True Colour (B4/B3/B2)",
    )

    # MORI is an exploratory proxy, NOT a confirmed-ore layer. A cool
    # sequential palette (rather than a hot "reserve-like" palette) is used
    # deliberately to avoid implying validated deposits.
    m.addLayer(
        diagnostic_stack.select("MORI"),
        {"min": 0.5, "max": 2.0, "palette": ["#f7fcf0", "#7bccc4", "#0868ac"]},
        "Exploratory Surface Prospectivity Proxy",
    )

    m.addLayer(
        diagnostic_stack.select("NDVI"),
        {"min": -0.2, "max": 0.8, "palette": ["#a50026", "#ffffbf", "#1a9850"]},
        "NDVI",
    )

    m.addLayer(
        diagnostic_stack.select("Slope"),
        {"min": 0, "max": 45, "palette": ["#ffffcc", "#fd8d3c", "#800026"]},
        "Slope (degrees)",
    )

    surface_exposure_mask = diagnostic_stack.select("NDVI").lte(NDVI_VEGETATION_THRESHOLD)
    m.addLayer(
        surface_exposure_mask.selfMask(),
        {"palette": ["#666666"]},
        "Surface-Exposure Mask (NDVI <= threshold)",
    )

    return m


# ============================================================================
# Local (client-side) feature metadata table -- pandas used only here
# ============================================================================
def build_feature_metadata_table():
    """
    Build a small local pandas table describing the 14 feature bands, for
    console/reference output only. This performs NO pixel-level processing
    and fabricates no data; it is local metadata handling as permitted by
    the project requirements.
    """
    descriptions = {
        "B2": "Blue surface reflectance (native 10 m)",
        "B3": "Green surface reflectance (native 10 m)",
        "B4": "Red surface reflectance (native 10 m)",
        "B8": "NIR surface reflectance (native 10 m)",
        "B8A": "Narrow NIR surface reflectance (native 20 m)",
        "B11": "SWIR1 surface reflectance (native 20 m)",
        "B12": "SWIR2 surface reflectance (native 20 m)",
        "MORI": "Manganese Oxide Ratio Index (exploratory proxy; not a concentration measurement)",
        "HAI": "Hydroxyl/Alteration Index (alteration-screening feature)",
        "FMI": "Ferrous Mineral Index (iron-related surface colour response)",
        "NDVI": "Vegetation index, used for surface-exposure masking",
        "Elevation": "SRTM elevation, metres (terrain context only)",
        "Slope": "SRTM-derived slope, degrees (terrain context only)",
        "Aspect": "SRTM-derived aspect, degrees (terrain context only)",
    }
    df = pd.DataFrame({
        "band": FEATURE_BAND_ORDER,
        "description": [descriptions[b] for b in FEATURE_BAND_ORDER],
    })
    return df


# ============================================================================
# 13. MAIN
# ============================================================================
def main():
    logger.info("=== M-MES: Manganese Surface Prospectivity Pipeline -- START ===")

    # --- Step 1: Earth Engine session -------------------------------------
    initialize_earth_engine()
    aoi = build_aoi()

    # --- Step 2: Sentinel-2 acquisition + compositing ---------------------
    composite, scene_count = build_s2_composite(aoi)
    check_required_bands(
        composite, SCALED_OPTICAL_BANDS, context="post-cloud-mask composite"
    )

    # --- Step 3: Spectral + terrain feature stacks -------------------------
    diagnostic_stack, surface_prospectivity_stack = build_feature_stacks(composite, aoi)
    check_required_bands(diagnostic_stack, FEATURE_BAND_ORDER, context="diagnostic stack")
    check_required_bands(
        surface_prospectivity_stack, FEATURE_BAND_ORDER, context="surface prospectivity stack"
    )

    # --- Step 4: Option A -- random unlabeled surface samples ---------------
    random_samples = generate_random_samples(surface_prospectivity_stack, aoi)
    random_task = export_random_samples_to_drive(random_samples)

    # --- Step 5: Option B -- labelled training samples (placeholder) -------
    # To use REAL ground-truth labels, build an ee.FeatureCollection with a
    # 'class_label' property (1 = verified manganese occurrence,
    # 0 = documented background) and assign it to user_supplied_labelled_fc
    # instead of None. See the accompanying usage instructions.
    user_supplied_labelled_fc = None
    labelled_samples = build_labelled_samples(
        surface_prospectivity_stack, labelled_fc=user_supplied_labelled_fc
    )
    labelled_task = None
    if labelled_samples is not None:
        labelled_task = export_labelled_samples_to_drive(labelled_samples)

    # --- Step 6: Optional GeoTIFF export ------------------------------------
    raster_task = None
    if ENABLE_RASTER_EXPORT:
        raster_task = export_raster_to_drive(surface_prospectivity_stack, aoi)
    else:
        logger.info("ENABLE_RASTER_EXPORT is False -- GeoTIFF export was NOT started.")

    # --- Step 7: Optional interactive preview map ---------------------------
    try:
        preview_map = build_preview_map(aoi, composite, diagnostic_stack)  # noqa: F841
        logger.info(
            "Preview map object created. In a Jupyter/Colab environment, "
            "display it by evaluating 'preview_map' as the last line of a cell."
        )
    except Exception as map_error:
        logger.warning(f"Preview map creation skipped due to an error: {map_error}")

    # --- Step 8: Final summary -----------------------------------------------
    logger.info("=== SUMMARY ===")
    feature_table = build_feature_metadata_table()
    print(feature_table.to_string(index=False))

    logger.info(f"Feature band list ({len(FEATURE_BAND_ORDER)} bands): {FEATURE_BAND_ORDER}")
    logger.info(f"Sentinel-2 source scenes used in composite: {scene_count}")
    logger.info(
        f"Random surface-sample export task: id={random_task.id}, "
        f"description='{EXPORT_PREFIX}_random_samples', status=submitted"
    )
    if labelled_task is not None:
        logger.info(
            f"Labelled sample export task: id={labelled_task.id}, "
            f"description='{EXPORT_PREFIX}_labelled_samples', status=submitted"
        )
    else:
        logger.info(
            "Labelled sample export task: NOT started "
            "(no real labelled FeatureCollection was supplied)."
        )
    if raster_task is not None:
        logger.info(
            f"Raster (GeoTIFF) export task: id={raster_task.id}, "
            f"description='{EXPORT_PREFIX}_prospectivity_raster', status=submitted"
        )
    else:
        logger.info("Raster (GeoTIFF) export task: NOT started (ENABLE_RASTER_EXPORT = False).")

    logger.info(
        "SCIENTIFIC LIMITATION: Outputs indicate relative surface "
        "prospectivity for follow-up investigation and do not confirm "
        "underground manganese reserves."
    )
    logger.info("=== M-MES: Manganese Surface Prospectivity Pipeline -- END ===")
    logger.info(
        "Check task progress at https://code.earthengine.google.com/tasks "
        "or via ee.batch.Task.list() until status shows COMPLETED."
    )


if __name__ == "__main__":
    main()
