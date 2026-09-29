"""
spectral_indices.py

Band-ratio / spectral index calculations used in mineral mapping. These
work on numpy arrays - read a stacked raster with rasterio, pull out the
bands you need by index, and pass them in. To use across sensors, map
each sensor's bands to the same logical name first using
SENSOR_BAND_MAP below, then call these functions the same way regardless
of which sensor the pixels came from.

These ratios follow the classic alteration-mapping formulas (see Sabins,
1999, "Remote Sensing: Principles and Interpretation") adapted to
Landsat 8/9 OLI band numbering.
"""

import numpy as np

# Which physical band each logical name maps to, per sensor. Adjust the
# ASTER/PRISMA entries to match your own band order after clipping/stacking.
SENSOR_BAND_MAP = {
    "landsat89_oli": {
        "blue": "SR_B2", "green": "SR_B3", "red": "SR_B4",
        "nir": "SR_B5", "swir1": "SR_B6", "swir2": "SR_B7",
    },
    "aster_vnir_tir_only": {
        # SWIR (bands 4-9) is NOT usable on any ASTER scene acquired
        # after April 2008 - see README. Only include it here if you are
        # deliberately using pre-2008 archival scenes.
        "green": "band1", "red": "band2", "nir": "band3",
    },
    "prisma_resampled": {
        # After resampling PRISMA's ~239 hyperspectral bands to the
        # nearest wavelength match for each logical band
        "green": "~560nm", "red": "~660nm", "nir": "~860nm",
        "swir1": "~1650nm", "swir2": "~2200nm",
    },
}


def _safe_divide(numerator, denominator, eps=1e-6):
    return numerator / (denominator + eps)


def iron_oxide_index(red, blue):
    """Ferric iron / laterite indicator (Landsat OLI: Band4 / Band2).
    Higher values suggest more Fe(III) oxide - relevant since Mn and Fe
    oxides commonly co-occur in Indian laterite and BIF-hosted deposits."""
    return _safe_divide(red, blue)


def ferrous_iron_index(swir1, nir):
    """Ferrous iron (Fe2+) indicator (Landsat OLI: Band6 / Band5), useful
    for the mafic/ultramafic host rocks associated with gondite/BIF
    sequences in the Sausar and Iron Ore Group belts."""
    return _safe_divide(swir1, nir)


def clay_alteration_index(swir1, swir2):
    """Clay / hydroxyl-bearing alteration indicator (Landsat OLI:
    Band6 / Band7). Elevated values often mark supergene-enriched or
    hydrothermally altered zones worth flagging for follow-up."""
    return _safe_divide(swir1, swir2)


def ndvi(nir, red):
    """Normalized Difference Vegetation Index. Used here less as a
    vegetation metric and more as a *negative control*: sparse or
    stressed vegetation over otherwise favourable geology can help you
    narrow down exposed or near-surface zones worth a closer look."""
    return _safe_divide(nir - red, nir + red)


def manganese_composite_index(red, blue, swir1, swir2, nir):
    """A simple, UNVALIDATED composite score combining the ratios above.
    Treat this as a starting feature for your model to weigh, not a
    validated ore indicator on its own - always check it against known
    occurrences before trusting it."""
    fe = iron_oxide_index(red, blue)
    clay = clay_alteration_index(swir1, swir2)
    veg = ndvi(nir, red)

    def _norm(x):
        return (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x) + 1e-6)

    return 0.5 * _norm(fe) + 0.3 * _norm(clay) - 0.2 * _norm(veg)
