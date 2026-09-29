"""
fetch_landsat_gee.py

Downloads a cloud-masked Landsat 8+9 Collection 2 Level-2 surface
reflectance composite over a given Area of Interest (AOI) using Google
Earth Engine. Landsat is the fastest data source to get moving with -
no registration friction beyond Earth Engine itself.

BEFORE RUNNING:
1. pip install earthengine-api geemap   (already in requirements.txt)
2. As of November 2024, Earth Engine requires every user - including
   noncommercial/research use, which remains free - to register a
   Google Cloud project. First-time setup:
     a. Go to https://code.earthengine.google.com/register
     b. Register a new Cloud project, choose "Unpaid usage" ->
        a noncommercial/research project type
     c. Copy the project ID into EE_PROJECT below
   Do this once per teammate - each person needs their own registered
   project (they're free, this isn't a shared-quota concern for a
   hackathon-sized workload).
3. First run will open a browser window for `ee.Authenticate()`.

Everyone on the team should run this same script so you're all pulling
identical pixels instead of passing multi-GB files around.
"""

import ee
import geemap

EE_PROJECT = "snu-hackathon"  # <-- replace with your registered project ID

# Example AOI: part of the Bonai-Keonjhar-Sundargarh manganese belt,
# Odisha. Swap in whichever belt/state your team is targeting.
DEFAULT_AOI_COORDS = [84.8, 21.5, 85.8, 22.3]  # [minLon, minLat, maxLon, maxLat]


def mask_landsat_clouds(image):
    """Mask clouds and cloud shadows using the QA_PIXEL band.

    Collection 2 QA_PIXEL bit flags: bit 3 = Cloud, bit 4 = Cloud Shadow.
    """
    qa = image.select("QA_PIXEL")
    cloud_bit = 1 << 3
    cloud_shadow_bit = 1 << 4
    mask = qa.bitwiseAnd(cloud_bit).eq(0).And(qa.bitwiseAnd(cloud_shadow_bit).eq(0))
    return image.updateMask(mask)


def scale_landsat(image):
    """Apply Collection 2 Level-2 scale factors to get true surface
    reflectance (USGS: SR = DN * 0.0000275 - 0.2)."""
    optical = image.select("SR_B.").multiply(0.0000275).add(-0.2)
    return image.addBands(optical, None, True)


def get_landsat_composite(aoi, start_date, end_date, max_cloud_cover=30):
    """Return a median, cloud-masked Landsat 8+9 SR composite clipped to aoi."""
    def _collection(collection_id):
        return (
            ee.ImageCollection(collection_id)
            .filterBounds(aoi)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt("CLOUD_COVER", max_cloud_cover))
            .map(mask_landsat_clouds)
            .map(scale_landsat)
        )

    merged = _collection("LANDSAT/LC08/C02/T1_L2").merge(
        _collection("LANDSAT/LC09/C02/T1_L2")
    )
    return merged.median().clip(aoi)


def main(aoi_coords=None, start_date="2024-11-01", end_date="2025-03-31",
         out_path="data/raw/landsat_composite.tif"):
    ee.Authenticate()  # no-op / cached if you've already authenticated once
    ee.Initialize(project=EE_PROJECT)

    aoi_coords = aoi_coords or DEFAULT_AOI_COORDS
    aoi = ee.Geometry.Rectangle(aoi_coords)

    composite = get_landsat_composite(aoi, start_date, end_date)

    # Blue, green, red, NIR, SWIR1, SWIR2 - everything you need for the
    # band-ratio mineral indices in src/preprocessing/spectral_indices.py
    bands = ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]
    composite = composite.select(bands)

    geemap.ee_export_image(
        composite,
        filename=out_path,
        scale=30,
        region=aoi,
        file_per_band=False,
    )
    print(f"Saved composite to {out_path}")


if __name__ == "__main__":
    main()
