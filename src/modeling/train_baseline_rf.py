"""
train_baseline_rf.py

Baseline mineral-prospectivity model: a Random Forest classifier trained
on pixel-level features sampled at known manganese-occurrence points
(positive class) plus randomly sampled background points (negative
class). This is intentionally simple - the goal is a working, end-to-end
prospectivity map fast, so your team can iterate (swap in PRISMA
hyperspectral features, try XGBoost, add a CNN) without rebuilding the
whole pipeline first.

Honest caveat: the cross-validation below is a plain stratified k-fold,
which is fine for a first baseline but will likely overstate accuracy on
spatial data (nearby pixels are correlated, so random splits leak
information between train and test). Before you report a final AUC,
swap in a spatial/blocked CV scheme - group points by region and hold
out whole regions, not random points.
"""

import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
from shapely.geometry import Point
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import joblib


def load_feature_stack(raster_path):
    """Read a multi-band GeoTIFF (each band = one engineered feature)."""
    with rasterio.open(raster_path) as src:
        stack = src.read()
        profile = src.profile
    return stack, profile


def sample_points(stack, profile, points_gdf, label):
    """Extract feature vectors at given point locations."""
    records = []
    for geom in points_gdf.geometry:
        row, col = rasterio.transform.rowcol(profile["transform"], geom.x, geom.y)
        if 0 <= row < stack.shape[1] and 0 <= col < stack.shape[2]:
            records.append(stack[:, row, col])
    df = pd.DataFrame(records)
    df["label"] = label
    return df


def make_background_points(stack, profile, n_points, exclude_gdf, buffer_m=500):
    """Randomly sample background (assumed non-deposit) pixels, buffered
    away from known occurrences so the negative class doesn't get
    contaminated by unmapped deposits sitting right next to a known one.

    Note: buffer_m assumes a projected CRS in metres. Reproject exclude_gdf
    first if your working CRS is geographic (lat/lon)."""
    exclude_buffer = exclude_gdf.buffer(buffer_m).unary_union
    rng = np.random.default_rng(42)
    n_rows, n_cols = stack.shape[1], stack.shape[2]
    records = []
    attempts = 0
    while len(records) < n_points and attempts < n_points * 20:
        attempts += 1
        row = rng.integers(0, n_rows)
        col = rng.integers(0, n_cols)
        x, y = rasterio.transform.xy(profile["transform"], row, col)
        if exclude_buffer.contains(Point(x, y)):
            continue
        pixel = stack[:, row, col]
        if np.isnan(pixel).any():
            continue
        records.append(pixel)
    df = pd.DataFrame(records)
    df["label"] = 0
    return df


def train(raster_path, occurrences_path, out_model_path="models/baseline_rf.joblib"):
    stack, profile = load_feature_stack(raster_path)
    occurrences = gpd.read_file(occurrences_path).to_crs(profile["crs"])

    positives = sample_points(stack, profile, occurrences, label=1)
    negatives = make_background_points(
        stack, profile, n_points=len(positives) * 3, exclude_gdf=occurrences
    )

    data = pd.concat([positives, negatives], ignore_index=True)
    feature_cols = [c for c in data.columns if c != "label"]
    X = data[feature_cols].values
    y = data["label"].values

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1,
    )

    aucs = []
    for train_idx, test_idx in cv.split(X, y):
        model.fit(X[train_idx], y[train_idx])
        probs = model.predict_proba(X[test_idx])[:, 1]
        aucs.append(roc_auc_score(y[test_idx], probs))
    print(f"5-fold CV ROC-AUC: {np.mean(aucs):.3f} +/- {np.std(aucs):.3f}")
    print("(Random split CV - treat as optimistic. Add spatial blocking "
          "before trusting this number in your final report.)")

    model.fit(X, y)  # final fit on all data for deployment
    joblib.dump(model, out_model_path)
    print(f"Saved trained model to {out_model_path}")
    return model


if __name__ == "__main__":
    train(
        raster_path="data/processed/feature_stack.tif",
        occurrences_path="data/known_occurrences.geojson",
    )
