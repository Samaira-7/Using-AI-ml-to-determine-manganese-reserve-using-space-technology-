# AI/ML + Space Tech for Manganese Reserves — SIH26009

Built around SIH 2026 problem statement **SIH26009** — "Using AI/ML and
Space Technology to Identify Manganese Reserves and Overcome Production
Shortfalls" — sponsored by the **Ministry of Steel** through **MOIL
Limited**, India's largest manganese ore producer (~48–50% of domestic
dioxide-ore supply, 10 mines across Nagpur/Bhandara in Maharashtra and
Balaghat in Madhya Pradesh, listed as BSE:533286 / NSE:MOIL). This repo
is built from the publicly listed problem statement text, not pulled
from the official portal directly — **verify exact wording, datasets
and submission rules on sih.gov.in** before you lock your pitch.

## The PS has two parts — both count

1. **Identify and map manganese reserves** more accurately using
   satellite/geological indicators (surface and sub-surface) — **Track A**.
2. **Predict production shortfalls** using equipment performance and
   weather/satellite inputs (rainfall, soil moisture, vegetation index,
   land temperature) — **Track B**.

Most teams over-invest in Track A (it's the flashier demo — a heatmap on
a map) and under-invest in Track B (it's explicitly in the PS, and
MOIL/Ministry of Steel judges will be listening for it). This repo
covers both, on purpose.

**Why this matters beyond the hackathon**: India imports roughly half
of its manganese needs. The National Critical Mineral Mission (launched
January 2025, ~Rs 34,300 crore outlay from 2024-25 to 2030-31) is
explicitly trying to cut that import dependence by accelerating domestic
exploration — GSI alone ran about 195 exploration projects in the
2024-25 field season under this push. A tool that speeds up reserve
identification and production planning sits directly inside that
national effort, not just a class assignment. Use this framing on your
opening/motivation slide.

## Data honesty policy — read this before you build anything

You will not get real, full-resolution data for every input the PS
mentions in 12 hours. That's fine. **Being upfront about what's real vs.
simulated is a strength in front of judges, not a weakness.** Follow
this for every data source, in the app UI and in the deck:

| Data | Status in 12 hours | What to do |
|---|---|---|
| Landsat, ASTER, Sentinel-2, SRTM DEM | Real, instant via Earth Engine | Use real data — no excuses |
| Rainfall / soil moisture / land surface temp | Real, via Earth Engine (CHIRPS, MODIS LST) | Use real data — no excuses |
| GSI lithology / occurrence points | Real, needs manual lookup | Pull a real (small) sample from Bhukosh; cite where each point came from |
| MOIL production figures | Real, public | Use IBM's Indian Minerals Yearbook tables or MOIL's annual report/investor deck — never invent tonnage |
| PRISMA hyperspectral | Real, but registration takes days | Describe as your production roadmap; don't fake a PRISMA scene for the demo |
| Equipment downtime / maintenance logs | Not public | Clearly label as **simulated for demo purposes**, both on-screen and in the deck: "the pipeline ingests MOIL's real SCADA/maintenance feed; we've simulated realistic downtime events to demonstrate the forecasting logic." |

Never present a simulated number as if it were measured. A judge who
catches one fabricated statistic will distrust every other number you
show them.

## Pipeline — Track A: reserve identification

```
Data sources (ASTER, PRISMA, Landsat, GSI, DEM)
        |
Preprocessing (co-register everything onto one grid)
        |
Feature engineering (mineral indices, terrain, structure)
        |
ML model (Random Forest baseline, cross-validated)
        |
Prospectivity map -> unified dashboard
```

### Why these data sources, and what to watch out for

- **ASTER** — free, 14-band VNIR/SWIR/TIR archive on Terra. The SWIR
  bands (4-9) have been unusable since April 2008 (a detector fault,
  never fixed) — only VNIR and TIR are reliable. Terra itself is
  expected to wind down in late 2026/early 2027, so treat ASTER as an
  archive to mine, not a live tasking source. It's directly queryable in
  Earth Engine as `ASTER/AST_L1T_003` — no separate download needed.
- **PRISMA** — Italian Space Agency hyperspectral mission, 239 bands
  (400-2505 nm), ~30 m. Free for research use but needs registration at
  prismauserregistration.asi.it plus a few days' approval — treat as
  roadmap, not something you wait on inside the 12 hours.
- **Landsat 8/9** — your workhorse. Free, 30 m, full SWIR, instant via
  Earth Engine (`LANDSAT/LC08/C02/T1_L2`, `LANDSAT/LC09/C02/T1_L2`).
  Build your first working pipeline on this alone.
- **GSI Bhukosh** (bhukosh.gsi.gov.in/Bhukosh/Public) — geological maps
  and the Metallogenic/Mineral Atlas of India. This is where your
  training labels and host-rock context come from.
- **DEM** — SRTM or Bhuvan's CartoDEM, for slope/curvature (laterite
  caps that host secondary Mn enrichment form on specific weathering
  profiles).

Add **Sentinel-2** (free, 10 m, 5-day revisit) as a practical stand-in
for what ASTER's SWIR would have given you.

### Feature engineering

Classic alteration-mapping ratios (see `src/preprocessing/spectral_indices.py`):
iron oxide (red/blue), ferrous iron (SWIR1/NIR), clay alteration
(SWIR1/SWIR2). Add lithology as a categorical feature, not just spectral
indices — India's manganese sits in specific host rocks (gondite in the
Sausar Group across Madhya Pradesh/Maharashtra, kodurite in the Eastern
Ghats, laterite-capped secondary ore in Odisha/Goa), and that geological
context predicts strongly on its own.

### The model

Random Forest baseline — interpretable, handles small positive-sample
counts better than deep learning, `class_weight="balanced"` for the
inevitable imbalance. Swap naive random CV for spatial/blocked CV before
reporting a final number — nearby pixels are correlated, so random
splits are optimistic.

## Pipeline — Track B: production shortfall forecasting

```
Real production history (IBM Yearbook / MOIL reports)
        |
Real rainfall + weather (Earth Engine, CHIRPS)
        |
Simulated equipment downtime (clearly labeled as synthetic)
        |
Regression model -> shortfall risk flag -> unified dashboard
```

See `src/production_forecast/shortfall_model.py`. Deliberately simple:
a linear regression of monthly production on rainfall and downtime
events, with a threshold flag for "at risk of shortfall." Most of
MOIL's mines are underground (only 3 of 10 are opencast: Dongri Buzurg,
Sitapatore, Tirodi) — weather exposure is likely concentrated on those,
which is a genuine domain insight worth stating out loud in your pitch
rather than treating all 10 mines identically.

Read the honesty note at the top of that file before running it —
production and rainfall must be real; downtime is explicitly simulated.

## Repo layout

```
manganese-prospectivity/
├── data/
│   ├── raw/                    # gitignored — satellite downloads land here
│   └── processed/              # gitignored — co-registered stacks, features
├── src/
│   ├── data_acquisition/       # pulling Landsat via Earth Engine (Track A)
│   ├── preprocessing/          # spectral indices / band ratios (Track A)
│   ├── modeling/               # train + validate the RF baseline (Track A)
│   ├── production_forecast/    # shortfall regression model (Track B)
│   └── utils/
├── app/                        # Streamlit dashboard — both tracks
├── models/                     # gitignored — trained model files
└── data/known_occurrences_template.geojson
```

## Setup (Windows)

The geospatial Python stack (GDAL, rasterio, geopandas) is notoriously
painful to `pip install` on Windows because of C-library dependencies.
Two options — pick one and **use it on every laptop**:

**Option A — Conda (simplest, stay on native Windows)**
```powershell
conda env create -f environment.yml
conda activate manganese-prospectivity
```

**Option B — WSL2** (if anyone hits native install issues)
```powershell
wsl --install
```
Then use the same `environment.yml` inside the Ubuntu shell.

Either way, install [Git for Windows](https://git-scm.com/download/win)
and [VS Code](https://code.visualstudio.com/).

## Getting six laptops working as one system

1. One shared GitHub repo, feature branches per track/stage, small pull
   requests.
2. Same environment everywhere — don't let six laptops drift to six
   different rasterio versions.
3. Use Earth Engine as your shared data layer instead of emailing
   satellite files — everyone runs the same script, gets identical
   pixels. Each teammate needs their own free registered Cloud project
   (mandatory since Nov 2024) — see the docstring in
   `fetch_landsat_gee.py`.
4. Agree on the *interface* between stages (band order/names in the
   feature-stack file) before anyone starts coding, so the model-builder
   doesn't need to know how preprocessing works internally.
5. **Build against mock data first, integrate real data second** — this
   is the single most important rule for keeping six people productive
   in parallel instead of blocked on each other. See the 12-hour plan
   below.

## Running it

```powershell
python src/data_acquisition/fetch_landsat_gee.py
python src/modeling/train_baseline_rf.py
python src/production_forecast/shortfall_model.py
streamlit run app/streamlit_app.py
```

## 12-hour hackathon plan (6 people)

**Hour 0-1 — Setup, together.** Everyone registers their Earth Engine
Cloud project, clones the repo, gets the environment running. Read the
PS twice. Assign roles (below). Do not skip this to "save time" — an
unsynced team loses far more than an hour later.

**Roles**
- **P1 + P2 — Track A data**: Landsat/ASTER via Earth Engine, GSI Bhukosh
  occurrence points + DEM. Ship a rough feature stack by hour 4.
- **P3 — Track A model**: builds and tests the Random Forest against a
  small synthetic feature stack from hour 1 (don't wait idle for P1/P2),
  swaps in the real stack once it lands.
- **P4 — Track B**: sources real MOIL/IBM production numbers, pulls real
  CHIRPS rainfall, builds the shortfall regression, clearly labels the
  simulated downtime piece.
- **P5 — Dashboard**: builds the Streamlit app against dummy outputs
  from hour 0, wires in real model outputs from both tracks as they
  arrive.
- **P6 — Narrative + honesty QA + deck**: gathers the MOIL/NCMM context
  above, drafts the pitch story, preps for judge Q&A, and is the person
  who checks every number that goes on a slide against the data-honesty
  table before it's shown to anyone.

**Hour 1-4 — Build in parallel.** Everyone builds against mocks first.

**Hour 4-6 — First integration.** Get a crude end-to-end version of both
tracks running inside the dashboard, even if ugly. This checkpoint
exists to surface interface mismatches while there's still time to fix
them.

**Hour 6-9 — Refine.** Improve whichever track is weaker. If time allows:
add SHAP/feature-importance output on Track A (a few lines with the
`shap` library, disproportionate demo impact) and tighten the Track B
threshold logic. Don't attempt spatial CV, PRISMA integration, or deep
learning in this window — that's Phase 2 territory, not 12-hour territory.

**Hour 9-10.5 — Polish + start the deck.** Screenshots and real metrics
now exist — build the actual PPT using them (outline below).

**Hour 10.5-11.5 — Rehearse.** Run the live demo twice. Record a backup
screen capture in case Wi-Fi/Earth Engine hiccups during judging. Assign
who answers what in Q&A.

**Hour 11.5-12 — Submit, breathe.**

## PPT outline (content, not a filled deck — ask once you have real results)

1. Title + team + PS ID (SIH26009)
2. The problem, in MOIL's own terms — reserve estimation is manual/slow,
   production planning mismatches happen (use the real MOIL mine/tonnage
   figures above, not generic filler)
3. Why it matters nationally — the ~50% import-dependence figure + NCMM
4. Solution architecture — both tracks, one diagram
5. Data sources — explicit real-vs-simulated table (yes, put this in
   the deck; it reads as rigor, not weakness)
6. Track A results — prospectivity map + what drives it
7. Track B results — shortfall flag + what drives it
8. Live demo (or backup recording)
9. What's next — PRISMA integration, spatial CV, scaling to other GSI
   belts beyond MOIL's current footprint
10. Impact — tie back to NCMM / Atmanirbhar Bharat framing

## Disclaimer

This is a hackathon prototype demonstrating an AI/ML + remote sensing +
production-forecasting workflow. It is not a certified reserve estimate
or an operational planning tool, and shouldn't be presented as one.
