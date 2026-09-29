"""
geo_utils.py

Small helpers for getting multi-sensor rasters (different resolutions,
sometimes different CRSs) onto one common grid before stacking them into
a single feature cube for the model.
"""

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling


def reproject_to_reference(src_path, ref_profile, out_path, resampling=Resampling.bilinear):
    """Reproject/resample a raster to match a reference grid's CRS,
    transform, width and height.

    Use Resampling.bilinear for continuous values (reflectance, indices).
    Use Resampling.nearest for categorical layers (e.g. lithology codes)
    so you don't invent fractional rock types at the edges of polygons.
    """
    with rasterio.open(src_path) as src:
        kwargs = ref_profile.copy()
        kwargs.update(count=src.count, dtype=src.dtypes[0])

        with rasterio.open(out_path, "w", **kwargs) as dst:
            for band_idx in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, band_idx),
                    destination=rasterio.band(dst, band_idx),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=ref_profile["transform"],
                    dst_crs=ref_profile["crs"],
                    resampling=resampling,
                )
    return out_path


def stack_feature_rasters(raster_paths, out_path):
    """Stack several single/multi-band rasters - already reprojected onto
    the same grid via reproject_to_reference - into one multi-band
    GeoTIFF, in the order given. Keep a note of what band order means;
    your modeling code needs to agree with whoever built this stack."""
    arrays = []
    ref_profile = None
    for path in raster_paths:
        with rasterio.open(path) as src:
            arrays.append(src.read())
            if ref_profile is None:
                ref_profile = src.profile

    stacked = np.concatenate(arrays, axis=0)
    ref_profile.update(count=stacked.shape[0])

    with rasterio.open(out_path, "w", **ref_profile) as dst:
        dst.write(stacked)
    return out_path
