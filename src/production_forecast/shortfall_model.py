"""
shortfall_model.py

Track B of SIH26009 ("...and Overcome Production Shortfalls") - predict
production shortfalls at MOIL's mines using historical production
figures and weather/satellite indicators. This is the half of the
problem statement most teams skip in favour of the flashier reserve map
- don't skip it, it's explicitly in the brief.

HONESTY NOTE (read before running):
- Production and rainfall numbers must be REAL. Do not invent tonnage or
  rainfall figures - a judge who catches one fabricated statistic will
  distrust every other number in your deck. Real sources:
    - Indian Bureau of Mines, "Indian Minerals Yearbook", Manganese Ore
      chapter - has actual year-over-year production tables (search
      ibm.gov.in for the latest edition's Manganese_Ore PDF).
    - MOIL Limited's own annual reports / investor presentations
      (moil.nic.in; BSE ticker 533286 / NSE ticker MOIL).
- Rainfall below is pulled live from Earth Engine (CHIRPS) - that part
  is genuinely real satellite data, not something you need to fake.
- Equipment downtime / maintenance logs are NOT public. Don't invent
  numbers and present them as real - simulate a clearly-labeled series
  instead (see simulate_equipment_downtime) and say so out loud in your
  app's UI and in your deck: "downtime is simulated to demonstrate the
  forecasting logic; the pipeline is built to ingest MOIL's real
  SCADA/maintenance feed directly."
"""

import numpy as np
import pandas as pd
import ee

EE_PROJECT = "your-gee-project-id"  # same registered project as fetch_landsat_gee.py

# MOIL's core operating belt: Balaghat (MP) + Nagpur/Bhandara (Maharashtra).
# Narrow or split this per-mine if you have time to georeference individual
# mines rather than the whole belt.
MOIL_BELT_AOI_COORDS = [79.0, 20.8, 80.5, 22.2]  # [minLon, minLat, maxLon, maxLat]


def fetch_rainfall_series(start_date, end_date, aoi_coords=None):
    """Pull real daily rainfall (CHIRPS, ~5 km) over the MOIL belt and
    return a monthly-aggregated pandas Series indexed by month-start.

    Note: pulling several years of DAILY images via getInfo() can be
    slow or hit Earth Engine response-size limits. If this hangs during
    the hackathon, narrow start_date/end_date to 1-2 years for the demo
    rather than debugging Earth Engine internals under time pressure.
    """
    ee.Authenticate()
    ee.Initialize(project=EE_PROJECT)

    aoi = ee.Geometry.Rectangle(aoi_coords or MOIL_BELT_AOI_COORDS)
    chirps = (
        ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
        .filterDate(start_date, end_date)
        .filterBounds(aoi)
    )

    def _to_feature(img):
        stat = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=aoi, scale=5000)
        return ee.Feature(None, {
            "date": img.date().format("YYYY-MM-dd"),
            "precip": stat.get("precipitation"),
        })

    features = chirps.map(_to_feature).getInfo()["features"]
    df = pd.DataFrame([f["properties"] for f in features])
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").resample("MS").sum(numeric_only=True)
    return df["precip"]


def load_real_production_series(csv_path="data/moil_production_real.csv"):
    """Load MOIL's actual historical production figures.

    You need to build this CSV yourself from the real sources listed in
    the module docstring - two columns, `month` (YYYY-MM-01) and
    `tonnes`. If you can only find annual totals, work at annual
    granularity instead of pretending you have monthly resolution you
    don't have. This function deliberately has no fallback/dummy data -
    if the file is missing, fix that with real numbers rather than
    stubbing it out.
    """
    df = pd.read_csv(csv_path, parse_dates=["month"])
    return df.set_index("month")["tonnes"]


def simulate_equipment_downtime(index, rng_seed=42, mean_events_per_month=1.5):
    """Clearly-SYNTHETIC equipment downtime series (Poisson event counts
    per month), standing in for real MOIL maintenance/SCADA logs that
    aren't public. Only use this to demonstrate the forecasting logic -
    label it as simulated everywhere it surfaces in the UI or the deck."""
    rng = np.random.default_rng(rng_seed)
    events = rng.poisson(lam=mean_events_per_month, size=len(index))
    return pd.Series(events, index=index, name="simulated_downtime_events")


def build_shortfall_model(production, rainfall, downtime):
    """Deliberately simple, explainable baseline: linear regression of
    monthly production on rainfall and downtime events. A transparent
    model you can explain in 30 seconds under judge questioning beats an
    opaque one you can't defend - there's no reason to reach for
    anything fancier in a 12-hour build."""
    from sklearn.linear_model import LinearRegression

    df = pd.concat([production, rainfall, downtime], axis=1).dropna()
    df.columns = ["tonnes", "precip", "downtime_events"]

    X = df[["precip", "downtime_events"]].values
    y = df["tonnes"].values

    model = LinearRegression().fit(X, y)
    r2 = model.score(X, y)
    print(f"R^2 on training data: {r2:.3f}")
    print(f"Coefficients: precip={model.coef_[0]:.2f}, "
          f"downtime={model.coef_[1]:.2f}")
    return model, df


def flag_shortfall_risk(model, upcoming_precip, upcoming_downtime, baseline_tonnes):
    """Given a forecast/assumed rainfall and downtime scenario for an
    upcoming month, flag whether predicted production falls meaningfully
    below baseline - this is the "predict shortfalls" output the PS
    actually asks for, not just a production number in isolation."""
    predicted = model.predict([[upcoming_precip, upcoming_downtime]])[0]
    shortfall_pct = (baseline_tonnes - predicted) / baseline_tonnes * 100
    return {
        "predicted_tonnes": predicted,
        "shortfall_pct_vs_baseline": shortfall_pct,
        "at_risk": shortfall_pct > 10,  # threshold is a starting guess - justify or tune it, don't just assert it
    }


if __name__ == "__main__":
    rainfall = fetch_rainfall_series("2019-01-01", "2024-12-31")
    production = load_real_production_series()
    downtime = simulate_equipment_downtime(production.index)
    model, merged = build_shortfall_model(production, rainfall, downtime)
