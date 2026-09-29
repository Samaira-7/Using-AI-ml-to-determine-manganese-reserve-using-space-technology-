# M-MES — Manganese Mineral Exploration System

Built for **SIH26009** — "Using AI/ML and Space Technology to Identify
Manganese Reserves and Overcome Production Shortfalls" — sponsored by
the Ministry of Steel through MOIL Limited.

This repo now ships two things:

1. **`/site`** — a static marketing/explainer website (open
   `site/index.html` in a browser, or deploy to GitHub Pages / Netlify).
   No setup required. Explains the science, walks through the pipeline,
   and includes a live in-browser band-ratio calculator.
2. **`/app`** — the upgraded multi-page Streamlit dashboard. Requires
   your own Earth Engine credentials to pull live satellite data (see
   below); runs immediately on synthetic demo data otherwise.

## What's new in this pass

- **Full visual redesign** — both the static site and the Streamlit app
  now share one deliberate design system (see `app/theme.py` and
  `site/styles.css`): a terrain-derived palette (laterite cream, basalt
  dark, manganese-oxide rust, ore-map gold), Fraunces/Inter/IBM Plex Mono
  type, and consistent status pills (`REAL DATA` / `SIMULATED` /
  `ROADMAP`) so nobody — teammate or judge — has to guess what's real.
- **Multi-page Streamlit app** (`app/Home.py` + `app/pages/`) replacing
  the single monolithic `app.py`:
  - `1_Prospectivity_Map.py` — the original app.py logic, restyled, with
    a clearly-labelled synthetic demo dataset so the map is explorable
    with zero setup.
  - `2_Live_AOI_Scanner.py` — **new**: run a real, on-demand Earth Engine
    query against any bounding box you specify and get back live
    MORI/HAI/FMI/NDVI statistics and a rendered thumbnail. This is the
    closest honest equivalent to "real-time detection from space" —
    see the scope note in that file for exactly what it does and
    doesn't do.
  - `3_Production_Shortfall.py` — Track B dashboard: upload real MOIL
    production data, optionally pull real CHIRPS rainfall live, tune
    the simulated downtime parameter, see the regression fit and a
    shortfall-risk flag.
  - `4_Band_Ratio_Lab.py` — **new**: interactive educational page for
    building intuition about why band ratios work, with live sliders.
- **Static demo site** (`/site`) for anyone without Python installed —
  judges, teammates on a phone, a README embed target.

## Quickstart

### Static site (no setup)
```bash
# Windows PowerShell
Start-Process .\site\index.html
# Or serve it locally at http://localhost:8080
python -m http.server 8080 --directory site
```

### Streamlit app

From the repository root, create and activate the environment:

```bash
# Recommended: Conda (handles geospatial dependencies reliably)
conda env create -f environment.yml
conda activate manganese-prospectivity

# Or use Python 3.11 and pip in a virtual environment:
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Start the dashboard
streamlit run app/Home.py
```
The app boots and is fully explorable with **zero external
credentials** — the Prospectivity Map and Band Ratio Lab pages work
immediately on synthetic/demo data. The Live AOI Scanner and the "pull
real rainfall" button on the Shortfall page need Earth Engine (see
below); they detect a missing connection and explain what's needed
rather than faking a result.

### Turning on real satellite data (Live AOI Scanner)
```bash
pip install earthengine-api
# register a free Cloud project: https://code.earthengine.google.com/register
earthengine authenticate     # one-time per machine
```
Every teammate needs their own registered project (free, mandatory
since Nov 2024) — see the docstring in `src/data_acquisition/fetch_landsat_gee.py`.

## On "real-time detection from space"

Worth stating plainly, because it shapes what this project honestly
claims: **no satellite performs onboard mineral detection.** What
Landsat/Sentinel-2 measure is reflected light in a handful of
wavelength bands; "detecting manganese" means computing band ratios
against that reflectance *after* the imagery reaches the ground, then
comparing the result to known occurrences. The Live AOI Scanner page
is the honest version of "near real-time" — it re-runs that math
on-demand against the freshest available cloud-free composite, which
for Sentinel-2 refreshes roughly every 5 days per tile. A production
version of "continuous monitoring" would be a scheduled job doing the
same query automatically and diffing successive layers — described in
the roadmap section of `site/index.html`, not yet built here.

## Repo layout

```
manganese-prospectivity/
├── site/                        # static marketing/explainer website
│   ├── index.html
│   ├── app.html                 # "how to run the dashboard" bridge page
│   ├── styles.css
│   └── script.js                # live band-ratio calculator (mirrors spectral_indices.py)
├── app/
│   ├── Home.py                  # Streamlit entry point
│   ├── theme.py                 # shared design system / CSS injection
│   └── pages/
│       ├── 1_Prospectivity_Map.py
│       ├── 2_Live_AOI_Scanner.py
│       ├── 3_Production_Shortfall.py
│       └── 4_Band_Ratio_Lab.py
├── src/
│   ├── data_acquisition/        # fetch_landsat_gee.py, gee_feature_pipeline.py
│   ├── preprocessing/           # spectral_indices.py, geo_utils.py
│   ├── modeling/                # train_baseline_rf.py, train_model.py
│   └── production_forecast/     # shortfall_model.py
├── data/                        # gitignored except known_occurrences_template.geojson
├── models/                      # gitignored — trained model files
└── outputs/                     # gitignored — figures, metadata, exports
```

## Data honesty policy (unchanged, still the most important section)

| Data | Status | What to do |
|---|---|---|
| Landsat, Sentinel-2, SRTM DEM | Real, instant via Earth Engine | Use real data — no excuses |
| Rainfall / soil moisture / land surface temp | Real, via Earth Engine (CHIRPS, MODIS LST) | Use real data — no excuses |
| GSI lithology / occurrence points | Real, needs manual lookup | Pull a real sample from Bhukosh; cite each point's source |
| MOIL production figures | Real, public | IBM Yearbook or MOIL's annual report/investor deck — never invent tonnage |
| PRISMA hyperspectral | Real, registration takes days | Roadmap item; don't fake a PRISMA scene for a demo |
| Equipment downtime / maintenance logs | Not public | Clearly label as simulated everywhere it appears |

Never present a simulated number as if it were measured. A judge who
catches one fabricated statistic will distrust every other number you
show them.

## The two tracks (unchanged)

**Track A — Identify reserves.** Satellite spectral indices (MORI, HAI,
FMI, NDVI) + terrain (elevation, slope, aspect), fed into a spatially
cross-validated model, ranked as field-validation targets. See
`src/modeling/train_model.py` for the full spatially-blocked CV
pipeline with SHAP explainability.

**Track B — Predict shortfalls.** Real rainfall + real MOIL production
history, regressed against a clearly-labelled simulated downtime
series, producing a shortfall-risk flag. See
`src/production_forecast/shortfall_model.py`.

## Roadmap

- Scheduled Earth Engine exports (Cloud Functions on a timer) for
  genuinely continuous — not literally real-time — layer refresh.
- PRISMA hyperspectral integration once registration clears.
- Spatial-CV-backed confidence intervals on every prospectivity score.
- Extend beyond the MOIL footprint to other GSI-mapped belts, feeding
  into the National Critical Mineral Mission's exploration push.

## Disclaimer

This is a hackathon prototype demonstrating an AI/ML + remote sensing +
production-forecasting workflow. It is not a certified reserve estimate
or an operational planning tool, and shouldn't be presented as one.
