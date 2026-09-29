#!/usr/bin/env python3
"""
train_model.py

M-MES: Manganese Mineral Exploration System
Spatially-validated XGBoost training pipeline for surface prospectivity.

============================================================================
SCOPE AND SCIENTIFIC LIMITATIONS (READ FIRST)
============================================================================
This script trains a classifier that estimates RELATIVE SURFACE
PROSPECTIVITY from satellite-derived spectral indices and terrain
derivatives, using real labelled ground-truth samples supplied by the user.

It does NOT and CANNOT:
    - estimate underground manganese reserves, ore grade, depth, or tonnage
    - assess commercial/economic viability of any location
    - prove that a spectral feature (e.g. MORI) is geologically caused by
      manganese -- feature importance reflects statistical association in
      THIS dataset under THIS validation setup, not geological causation

Nearby samples share similar spectral/terrain context (spatial
autocorrelation). Randomly splitting such samples between train and
validation folds leaks information and produces optimistic, unreliable
performance estimates. This script therefore uses BLOCK-BASED SPATIAL
CROSS-VALIDATION (grouping nearby points into blocks, never splitting a
block across train/validation) as its primary evaluation method, and
never falls back silently to ordinary random StratifiedKFold.

Even spatial blocking is not a perfect guarantee against all leakage
(e.g. very large blocks, or spatial autocorrelation ranges exceeding the
block size, can still cause some optimism). Reported metrics should be
read as a reasonable, defensible estimate -- not a certified accuracy.

    "The model estimates relative surface prospectivity based on the
    supplied labelled observations and remote-sensing/terrain features.
    Predictions require geological interpretation and field validation
    before any exploration decision."

============================================================================
DEPENDENCIES (install once, in your local environment / virtualenv)
============================================================================
    pip install pandas numpy scikit-learn xgboost joblib matplotlib seaborn shap

SHAP is optional. If it is not installed, or fails at runtime for any
environment-specific reason, the script logs a clear warning and continues
using XGBoost's built-in gain-based feature importance only.

============================================================================
USAGE
============================================================================
    python train_model.py

Update INPUT_CSV_PATH (and other CONFIGURATION values) below before running.
"""

import json
import logging
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; safe for headless/server runs
import matplotlib.pyplot as plt
import seaborn as sns

import joblib

from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
    balanced_accuracy_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    classification_report,
)

try:
    from sklearn.model_selection import StratifiedGroupKFold
    _HAS_STRATIFIED_GROUP_KFOLD = True
except ImportError:  # older scikit-learn versions
    _HAS_STRATIFIED_GROUP_KFOLD = False

try:
    from xgboost import XGBClassifier
    import xgboost
except ImportError as xgb_import_error:  # pragma: no cover
    raise ImportError(
        "xgboost is required but could not be imported. Install it with: "
        "pip install xgboost. "
        f"Original error: {xgb_import_error}"
    )

SHAP_AVAILABLE = True
_SHAP_IMPORT_ERROR = None
try:
    import shap
except Exception as shap_import_error:  # pragma: no cover
    SHAP_AVAILABLE = False
    _SHAP_IMPORT_ERROR = shap_import_error


# ============================================================================
# CONFIGURATION
# ============================================================================
INPUT_CSV_PATH = Path("data/mmes_labelled_samples.csv")
OUTPUT_DIR = Path("outputs")

TARGET_COLUMN = "class_label"
LONGITUDE_COLUMN = "longitude"
LATITUDE_COLUMN = "latitude"

RANDOM_SEED = 42
N_SPATIAL_FOLDS = 5

# Size, in decimal degrees, of each spatial block used for grouping nearby
# samples during cross-validation. At the Balaghat AOI latitude (~22N),
# 0.05 degrees is roughly 5-5.5 km. Widen this if spatial autocorrelation
# in your features extends further than that; narrow it if you have very
# few, tightly clustered samples.
SPATIAL_BLOCK_SIZE_DEGREES = 0.05

# Minimum number of samples required in EACH class for the pipeline to run
# at all. This is a data-adequacy guard, not a statement about what sample
# size is "enough" for a trustworthy model -- more is always better.
MIN_SAMPLES_PER_CLASS = 15

# Optional: fraction of unique spatial blocks (0 < TEST_SIZE < 1) to hold
# out, untouched, as a final spatial test set BEFORE cross-validation.
# Set to None to disable (recommended default for small hackathon-scale
# datasets, where held-out spatial CV already serves this role).
TEST_SIZE = None

USE_SHAP = True
ENABLE_TUNING = False

MODEL_OUTPUT_PATH = OUTPUT_DIR / "manganese_model.json"
PIPELINE_OUTPUT_PATH = OUTPUT_DIR / "manganese_prospectivity_pipeline.joblib"
METADATA_OUTPUT_PATH = OUTPUT_DIR / "model_metadata.json"

# Default feature list and required order (Section 4 / project spec).
DEFAULT_FEATURE_COLUMNS = [
    "B2", "B3", "B4", "B8", "B8A", "B11", "B12",
    "MORI", "HAI", "FMI", "NDVI",
    "Elevation", "Slope", "Aspect",
]

# Columns that must NEVER be used as model predictors, regardless of what
# else is present in the CSV. Coordinates are used only for spatial
# blocking and map output, never as direct predictors by default.
NON_FEATURE_COLUMNS = {LONGITUDE_COLUMN, LATITUDE_COLUMN, TARGET_COLUMN, "block_id"}

# Baseline XGBoost hyperparameters. These are reasonable defaults for a
# small tabular binary-classification problem -- NOT claimed to be optimal
# for any particular dataset.
DEFAULT_XGB_PARAMS = dict(
    learning_rate=0.05,
    max_depth=4,
    n_estimators=300,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=1,
    reg_lambda=1.0,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=RANDOM_SEED,
    n_jobs=-1,
)

# Hyperparameter search space used only if ENABLE_TUNING = True and enough
# data is available (see is_tuning_feasible()).
XGB_PARAM_DISTRIBUTIONS = {
    "classifier__learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1],
    "classifier__max_depth": [3, 4, 5, 6],
    "classifier__n_estimators": [150, 300, 500],
    "classifier__min_child_weight": [1, 3, 5],
    "classifier__subsample": [0.6, 0.8, 1.0],
    "classifier__colsample_bytree": [0.6, 0.8, 1.0],
}
N_TUNING_ITER = 20

CLASSIFICATION_THRESHOLD_DEFAULT = 0.50

SCIENTIFIC_LIMITATION_STATEMENT = (
    "The model estimates relative surface prospectivity based on the "
    "supplied labelled observations and remote-sensing/terrain features. "
    "Predictions require geological interpretation and field validation "
    "before any exploration decision. This model does not detect "
    "underground reserves, ore grade, depth, tonnage, or economic "
    "viability. Feature importance reflects statistical association in "
    "this dataset under this validation setup, not geological causation."
)

FEATURES_OF_INTEREST = ["MORI", "HAI", "FMI", "Slope", "Elevation"]

# ============================================================================
# Logging
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("MMES_Train")


# ============================================================================
# Utility: output directory
# ============================================================================
def ensure_output_dir(output_dir: Path) -> None:
    """Create the output directory (and parents) safely, or raise a clear error."""
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory ready: {output_dir.resolve()}")
    except OSError as dir_error:
        raise OSError(
            f"Could not create output directory '{output_dir}'. "
            f"Check filesystem permissions. Original error: {dir_error}"
        )


# ============================================================================
# 3. INPUT DATA VALIDATION
# ============================================================================
def load_and_validate_data():
    """
    Load, clean, and validate the labelled samples CSV.

    Returns
    -------
    data : pd.DataFrame
        Cleaned dataframe, containing at least LONGITUDE_COLUMN,
        LATITUDE_COLUMN, TARGET_COLUMN, and the active feature columns.
    active_feature_columns : list[str]
        The subset of DEFAULT_FEATURE_COLUMNS actually present and usable.

    This function NEVER imputes the target column and NEVER fabricates
    rows. It only removes rows it cannot legitimately use, and reports
    exactly why and how many.
    """
    if not INPUT_CSV_PATH.exists():
        raise FileNotFoundError(
            f"Input CSV not found at '{INPUT_CSV_PATH}'. Update INPUT_CSV_PATH "
            "in the CONFIGURATION section to point to your real labelled "
            "samples export (e.g. from gee_feature_pipeline.py's labelled "
            "sample export, joined with verified ground-truth class_label "
            "values)."
        )

    try:
        data = pd.read_csv(INPUT_CSV_PATH)
    except Exception as read_error:
        raise ValueError(f"Failed to read CSV '{INPUT_CSV_PATH}': {read_error}")

    logger.info(f"Loaded {len(data)} rows from {INPUT_CSV_PATH}")

    # --- Required non-feature columns -------------------------------------
    required_core_columns = [LONGITUDE_COLUMN, LATITUDE_COLUMN, TARGET_COLUMN]
    missing_core = [c for c in required_core_columns if c not in data.columns]
    if missing_core:
        raise ValueError(
            f"Missing required column(s): {missing_core}. The CSV must "
            f"contain '{LONGITUDE_COLUMN}', '{LATITUDE_COLUMN}', and "
            f"'{TARGET_COLUMN}'."
        )

    # --- Feature-column availability check (documented, not silent) --------
    present_features = [c for c in DEFAULT_FEATURE_COLUMNS if c in data.columns]
    missing_features = [c for c in DEFAULT_FEATURE_COLUMNS if c not in data.columns]
    if missing_features:
        logger.warning(
            f"The following default feature columns are ABSENT from the "
            f"input CSV and will be excluded from training: {missing_features}. "
            f"Proceeding with: {present_features}"
        )
    if not present_features:
        raise ValueError(
            "None of the expected feature columns "
            f"{DEFAULT_FEATURE_COLUMNS} were found in the input CSV. Cannot "
            "train a model with zero features."
        )

    n_start = len(data)

    # --- Coerce target to numeric, drop invalid/missing labels --------------
    data[TARGET_COLUMN] = pd.to_numeric(data[TARGET_COLUMN], errors="coerce")
    invalid_target_mask = ~data[TARGET_COLUMN].isin([0, 1])
    n_invalid_target = int(invalid_target_mask.sum())
    if n_invalid_target > 0:
        logger.warning(
            f"Dropping {n_invalid_target} row(s) with missing/invalid "
            f"'{TARGET_COLUMN}' (must be exactly 0 or 1 after cleaning). "
            "Target labels are never imputed or fabricated."
        )
    data = data.loc[~invalid_target_mask].copy()
    data[TARGET_COLUMN] = data[TARGET_COLUMN].astype(int)

    # --- Drop rows missing coordinates ---------------------------------------
    coord_missing_mask = data[LONGITUDE_COLUMN].isna() | data[LATITUDE_COLUMN].isna()
    n_coord_missing = int(coord_missing_mask.sum())
    if n_coord_missing > 0:
        logger.warning(
            f"Dropping {n_coord_missing} row(s) with missing "
            f"{LONGITUDE_COLUMN}/{LATITUDE_COLUMN} (required for spatial "
            "blocking)."
        )
    data = data.loc[~coord_missing_mask].copy()

    # --- Coerce feature columns to numeric (invalid entries -> NaN) --------
    for col in present_features:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    # --- Drop rows where ALL feature columns are NaN (unusable rows) --------
    all_features_nan_mask = data[present_features].isna().all(axis=1)
    n_all_features_nan = int(all_features_nan_mask.sum())
    if n_all_features_nan > 0:
        logger.warning(
            f"Dropping {n_all_features_nan} row(s) where every feature "
            "column is missing/non-numeric (nothing to impute from)."
        )
    data = data.loc[~all_features_nan_mask].copy()

    n_end = len(data)
    logger.info(
        f"Row count after cleaning: {n_end} (dropped {n_start - n_end} of "
        f"{n_start} total: {n_invalid_target} invalid target, "
        f"{n_coord_missing} missing coordinates, {n_all_features_nan} rows "
        "with no usable feature values)."
    )

    # --- Class checks --------------------------------------------------------
    class_counts = data[TARGET_COLUMN].value_counts().sort_index()
    n_pos = int(class_counts.get(1, 0))
    n_neg = int(class_counts.get(0, 0))
    logger.info(f"Class counts -> negative(0): {n_neg}, positive(1): {n_pos}")
    if n_pos + n_neg > 0:
        logger.info(f"Positive class ratio: {n_pos / (n_pos + n_neg):.4f}")

    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            "The cleaned dataset contains only one class "
            f"(negative={n_neg}, positive={n_pos}). A binary classifier "
            "cannot be trained or evaluated on single-class data. This "
            "script will NOT fabricate samples of the missing class."
        )

    if n_pos < MIN_SAMPLES_PER_CLASS or n_neg < MIN_SAMPLES_PER_CLASS:
        raise ValueError(
            f"Insufficient samples per class: negative={n_neg}, "
            f"positive={n_pos}, required minimum per class = "
            f"{MIN_SAMPLES_PER_CLASS} (see MIN_SAMPLES_PER_CLASS). Collect "
            "more verified ground-truth data before training."
        )

    if n_end < N_SPATIAL_FOLDS * 2:
        raise ValueError(
            f"Only {n_end} usable rows remain, which is too few to "
            f"credibly support {N_SPATIAL_FOLDS}-fold spatial "
            "cross-validation. Reduce N_SPATIAL_FOLDS or collect more data."
        )

    return data.reset_index(drop=True), present_features


# ============================================================================
# 5. SPATIAL BLOCK CREATION
# ============================================================================
def create_spatial_blocks(data: pd.DataFrame, longitude_column: str,
                           latitude_column: str, block_size_degrees: float) -> pd.DataFrame:
    """
    Assign each sample to a reproducible spatial block based on a coordinate
    grid of size `block_size_degrees` x `block_size_degrees`.

    Spatial blocks are used (instead of ordinary random StratifiedKFold) so
    that nearby samples -- which tend to share similar spectral and terrain
    context due to spatial autocorrelation -- are never split between train
    and validation within the same fold. This reduces (though does not
    perfectly eliminate) the optimistic bias that random splitting would
    otherwise introduce into the reported performance metrics.
    """
    data = data.copy()
    lon_bin = np.floor(data[longitude_column] / block_size_degrees).astype(int)
    lat_bin = np.floor(data[latitude_column] / block_size_degrees).astype(int)
    data["block_id"] = lon_bin.astype(str) + "_" + lat_bin.astype(str)

    n_blocks = data["block_id"].nunique()
    per_block = data.groupby("block_id").size()
    logger.info(f"Spatial blocking: block_size_degrees={block_size_degrees}")
    logger.info(f"Number of unique spatial blocks: {n_blocks}")
    logger.info(
        "Samples-per-block summary -> "
        f"min={per_block.min()}, mean={per_block.mean():.2f}, "
        f"median={per_block.median():.1f}, max={per_block.max()}"
    )

    if TARGET_COLUMN in data.columns:
        block_class = data.groupby("block_id")[TARGET_COLUMN].agg(["size", "mean"])
        n_pure_negative = int((block_class["mean"] == 0).sum())
        n_pure_positive = int((block_class["mean"] == 1).sum())
        n_mixed = int(((block_class["mean"] > 0) & (block_class["mean"] < 1)).sum())
        logger.info(
            f"Block class composition -> pure-negative blocks: "
            f"{n_pure_negative}, pure-positive blocks: {n_pure_positive}, "
            f"mixed blocks: {n_mixed}"
        )

    return data


# ============================================================================
# 6. SPATIAL CROSS-VALIDATION SPLITTER SELECTION
# ============================================================================
def get_spatial_cv_splitter(n_splits: int, groups: np.ndarray, y: np.ndarray):
    """
    Return (splits, method_name).

    Preferred: StratifiedGroupKFold (respects both group integrity AND
    class stratification across folds).

    Fallback: GroupKFold (respects group integrity only; class balance
    across folds is NOT guaranteed). Used only if StratifiedGroupKFold is
    unavailable in the installed scikit-learn version, or if it raises an
    error for this specific dataset (e.g. degenerate block/class
    combinations). This script never silently falls back to ordinary
    random StratifiedKFold.
    """
    if _HAS_STRATIFIED_GROUP_KFOLD:
        try:
            splitter = StratifiedGroupKFold(
                n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED
            )
            splits = list(splitter.split(np.zeros(len(y)), y, groups=groups))
            return splits, "StratifiedGroupKFold"
        except Exception as sgkf_error:
            logger.warning(
                "StratifiedGroupKFold failed for this dataset "
                f"({sgkf_error}). Falling back to GroupKFold. "
                "*** CLASS BALANCE ACROSS FOLDS MAY BE IMPERFECT. ***"
            )
    else:
        logger.warning(
            "StratifiedGroupKFold is not available in this scikit-learn "
            "version. Falling back to GroupKFold. "
            "*** CLASS BALANCE ACROSS FOLDS MAY BE IMPERFECT. ***"
        )

    splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    splits = list(splitter.split(np.zeros(len(y)), y, groups=groups))
    return splits, "GroupKFold (fallback)"


# ============================================================================
# 7. MODEL PIPELINE
# ============================================================================
def build_pipeline(xgb_params: dict) -> Pipeline:
    """
    Build the scikit-learn Pipeline: median imputation (fit per-fold, never
    on the full dataset before splitting) followed by an XGBoost classifier.
    """
    imputer = SimpleImputer(strategy="median")
    imputer.set_output(transform="pandas")  # preserves feature names for
    # importance/SHAP reporting downstream.
    classifier = XGBClassifier(**xgb_params)
    return Pipeline([("imputer", imputer), ("classifier", classifier)])


def compute_scale_pos_weight(y) -> float:
    """scale_pos_weight = n_negative / n_positive, computed on TRAINING data only."""
    y = np.asarray(y)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0:
        logger.warning(
            "compute_scale_pos_weight: zero positive samples in this "
            "training split; defaulting scale_pos_weight to 1.0."
        )
        return 1.0
    return n_neg / n_pos


# ============================================================================
# 6 (cont.). SPATIAL CROSS-VALIDATION EXECUTION
# ============================================================================
def run_spatial_cross_validation(X: pd.DataFrame, y: pd.Series, block_ids: pd.Series,
                                  coords: pd.DataFrame, xgb_params: dict,
                                  n_splits: int, threshold: float = CLASSIFICATION_THRESHOLD_DEFAULT):
    """
    Run block-based spatial cross-validation.

    Returns (fold_metrics_df, oof_predictions_df, cv_method_name).

    For every fold: preprocessing (median imputation) and the model are fit
    ONLY on that fold's training rows; predictions are made on the held-out
    validation rows only. Block membership is confirmed disjoint between
    train/validation before fitting.
    """
    groups = block_ids.to_numpy()
    y_arr = y.to_numpy()

    splits, cv_method_name = get_spatial_cv_splitter(n_splits, groups, y_arr)
    logger.info(f"Spatial cross-validation method: {cv_method_name} ({n_splits} folds)")

    base_pipeline = build_pipeline(xgb_params)

    fold_records = []
    oof_rows = []

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        train_blocks = set(groups[train_idx])
        val_blocks = set(groups[val_idx])
        overlap = train_blocks & val_blocks
        if overlap:
            raise AssertionError(
                f"Fold {fold_idx}: spatial leakage detected -- block_id(s) "
                f"{overlap} appear in BOTH train and validation sets. "
                "Aborting rather than reporting an invalid metric."
            )

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y_arr[train_idx], y_arr[val_idx]

        scale_pos_weight = compute_scale_pos_weight(y_train)
        pipe = clone(base_pipeline)
        pipe.set_params(classifier__scale_pos_weight=scale_pos_weight)
        pipe.fit(X_train, y_train)

        y_proba = pipe.predict_proba(X_val)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)

        val_has_both_classes = len(np.unique(y_val)) > 1
        if not val_has_both_classes:
            logger.warning(
                f"Fold {fold_idx}: validation set contains a SINGLE class "
                f"({np.unique(y_val).tolist()}). ROC-AUC and PR-AUC cannot "
                "be calculated for this fold; skipping those two metrics "
                "for this fold only."
            )
            roc_auc = np.nan
            pr_auc = np.nan
        else:
            roc_auc = roc_auc_score(y_val, y_proba)
            pr_auc = average_precision_score(y_val, y_proba)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_val, y_pred, average="binary", zero_division=0
        )
        bal_acc = balanced_accuracy_score(y_val, y_pred)
        cm = confusion_matrix(y_val, y_pred, labels=[0, 1])

        fold_records.append({
            "fold": fold_idx,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "n_train_blocks": len(train_blocks),
            "n_val_blocks": len(val_blocks),
            "val_has_both_classes": val_has_both_classes,
            "roc_auc": roc_auc,
            "pr_auc_average_precision": pr_auc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "balanced_accuracy": bal_acc,
            "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
            "scale_pos_weight_used": scale_pos_weight,
        })

        val_coords = coords.iloc[val_idx]
        for i, orig_idx in enumerate(val_idx):
            oof_rows.append({
                "longitude": val_coords[LONGITUDE_COLUMN].iloc[i],
                "latitude": val_coords[LATITUDE_COLUMN].iloc[i],
                "block_id": groups[orig_idx],
                "actual_label": int(y_val[i]),
                "predicted_probability": float(y_proba[i]),
                "predicted_class": int(y_pred[i]),
                "fold": fold_idx,
            })

        logger.info(
            f"Fold {fold_idx}: train={len(train_idx)} ({len(train_blocks)} blocks), "
            f"val={len(val_idx)} ({len(val_blocks)} blocks), "
            f"ROC-AUC={'n/a' if np.isnan(roc_auc) else round(roc_auc, 4)}, "
            f"PR-AUC={'n/a' if np.isnan(pr_auc) else round(pr_auc, 4)}, "
            f"F1={f1:.4f}"
        )

    fold_metrics_df = pd.DataFrame(fold_records)
    oof_df = pd.DataFrame(oof_rows)
    return fold_metrics_df, oof_df, cv_method_name


# ============================================================================
# 9. OPERATIONAL PROBABILITY THRESHOLD SELECTION
# ============================================================================
def select_operational_threshold(oof_df: pd.DataFrame, objective: str = "max_f1",
                                  min_recall: float = None):
    """
    Optionally select an operational classification threshold from POOLED
    out-of-fold predictions, using a transparent objective:
        - "max_f1": the threshold that maximizes F1 on pooled OOF predictions
        - "min_recall": the lowest threshold achieving recall >= min_recall

    IMPORTANT CAVEAT (always reported alongside this threshold): the
    threshold is selected using the same out-of-fold predictions that were
    also used to report performance. Metrics AT this threshold are
    therefore an optimistic operational illustration, not an unbiased
    estimate of future performance. The threshold-independent metrics
    (ROC-AUC, PR-AUC) remain the primary, more trustworthy performance
    indicators.
    """
    y_true = oof_df["actual_label"].to_numpy()
    y_proba = oof_df["predicted_probability"].to_numpy()

    if len(np.unique(y_true)) < 2:
        logger.warning(
            "select_operational_threshold: pooled OOF labels contain a "
            "single class; cannot select a data-driven threshold. Falling "
            f"back to the default threshold {CLASSIFICATION_THRESHOLD_DEFAULT}."
        )
        return CLASSIFICATION_THRESHOLD_DEFAULT, "fallback_default_single_class"

    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

    if objective == "max_f1":
        eps = 1e-12
        f1_scores = 2 * precision * recall / (precision + recall + eps)
        f1_scores = f1_scores[:-1]  # align with `thresholds` length
        if len(thresholds) == 0:
            chosen_threshold = CLASSIFICATION_THRESHOLD_DEFAULT
        else:
            best_idx = int(np.nanargmax(f1_scores))
            chosen_threshold = float(thresholds[best_idx])
    elif objective == "min_recall":
        if min_recall is None:
            raise ValueError("min_recall must be provided when objective='min_recall'.")
        eligible = np.where(recall[:-1] >= min_recall)[0]
        if len(eligible) == 0:
            logger.warning(
                f"select_operational_threshold: no threshold in pooled OOF "
                f"predictions achieves recall >= {min_recall}. Falling back "
                f"to default threshold {CLASSIFICATION_THRESHOLD_DEFAULT}."
            )
            chosen_threshold = CLASSIFICATION_THRESHOLD_DEFAULT
        else:
            # Among thresholds meeting the recall target, pick the one with
            # the highest precision (best F1-like trade-off).
            best_idx = eligible[np.argmax(precision[:-1][eligible])]
            chosen_threshold = float(thresholds[best_idx])
    else:
        raise ValueError(f"Unknown threshold-selection objective: {objective}")

    logger.info(
        f"Operational threshold selected via '{objective}' objective on "
        f"POOLED out-of-fold predictions: {chosen_threshold:.4f}. This is "
        "labeled as an OPERATIONAL threshold, reported separately from the "
        "threshold-independent probability metrics (ROC-AUC/PR-AUC), and is "
        "NOT an unbiased estimate of future performance."
    )
    return chosen_threshold, objective


# ============================================================================
# 8. OPTIONAL HYPERPARAMETER TUNING
# ============================================================================
def is_tuning_feasible(y: pd.Series, n_splits: int) -> (bool, str):
    """
    Decide whether there is enough data to credibly support hyperparameter
    tuning on top of spatial cross-validation. Tuning multiplies the number
    of model fits considerably, so it needs a comfortable data margin.
    """
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    min_class = min(n_pos, n_neg)
    required_min_class = max(MIN_SAMPLES_PER_CLASS * 2, n_splits * 4)
    if min_class < required_min_class:
        return False, (
            f"Smallest class has {min_class} samples; at least "
            f"{required_min_class} recommended per class to credibly "
            "support hyperparameter search on top of spatial CV."
        )
    if len(y) < n_splits * 20:
        return False, (
            f"Only {len(y)} total samples; recommend at least "
            f"{n_splits * 20} for credible tuning. Training a baseline "
            "model instead."
        )
    return True, "Sufficient data for tuning."


def run_hyperparameter_tuning(X: pd.DataFrame, y: pd.Series, block_ids: pd.Series,
                               n_splits: int):
    """
    Run RandomizedSearchCV using the SAME spatial (group-based) splitter as
    the main evaluation -- never ordinary random CV -- optimizing ROC-AUC
    where folds permit it.

    NOTE (limitation): because tuning and the main reported spatial-CV
    metrics both use folds built from the same data, the final reported
    fold metrics may be mildly optimistic relative to a fully independent
    (nested) cross-validation. This is a known, documented trade-off in a
    time-constrained prototype and is not claimed to be eliminated.
    """
    groups = block_ids.to_numpy()
    y_arr = y.to_numpy()
    splits, cv_method_name = get_spatial_cv_splitter(n_splits, groups, y_arr)

    def safe_roc_auc_scorer(estimator, X_val, y_val):
        if len(np.unique(y_val)) < 2:
            return np.nan
        proba = estimator.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, proba)

    base_params = dict(DEFAULT_XGB_PARAMS)
    base_pipeline = build_pipeline(base_params)

    search = RandomizedSearchCV(
        estimator=base_pipeline,
        param_distributions=XGB_PARAM_DISTRIBUTIONS,
        n_iter=N_TUNING_ITER,
        scoring=safe_roc_auc_scorer,
        cv=splits,
        random_state=RANDOM_SEED,
        n_jobs=1,
        error_score=np.nan,
        refit=True,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        search.fit(X, y_arr)

    mean_scores = np.asarray(search.cv_results_["mean_test_score"], dtype=float)
    if np.all(np.isnan(mean_scores)):
        logger.warning(
            "Hyperparameter tuning: every candidate scored NaN (likely due "
            "to single-class validation folds across the board). Falling "
            "back to baseline hyperparameters instead of tuning results."
        )
        return dict(DEFAULT_XGB_PARAMS), False

    best_score = float(np.nanmax(mean_scores))
    logger.info(
        f"Hyperparameter tuning complete using {cv_method_name} splitter "
        f"({N_TUNING_ITER} candidates). Best pooled-fold mean ROC-AUC (over "
        f"folds where calculable): {best_score:.4f}. Selected parameters "
        f"are the best FOUND within this search, not proven globally optimal."
    )

    best_params = {
        k.replace("classifier__", ""): v
        for k, v in search.best_params_.items()
    }
    tuned_xgb_params = dict(DEFAULT_XGB_PARAMS)
    tuned_xgb_params.update(best_params)
    return tuned_xgb_params, True


# ============================================================================
# 10. FIGURES
# ============================================================================
def plot_roc_curve(oof_df: pd.DataFrame, output_path: Path):
    y_true = oof_df["actual_label"].to_numpy()
    y_proba = oof_df["predicted_probability"].to_numpy()
    if len(np.unique(y_true)) < 2:
        logger.warning("plot_roc_curve: pooled OOF labels are single-class; skipping ROC plot.")
        return None
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    pooled_auc = roc_auc_score(y_true, y_proba)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"Pooled out-of-fold ROC (AUC = {pooled_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Pooled Out-of-Fold Predictions)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Saved ROC curve: {output_path} (pooled AUC = {pooled_auc:.4f})")
    return pooled_auc


def plot_pr_curve(oof_df: pd.DataFrame, output_path: Path):
    y_true = oof_df["actual_label"].to_numpy()
    y_proba = oof_df["predicted_probability"].to_numpy()
    if len(np.unique(y_true)) < 2:
        logger.warning("plot_pr_curve: pooled OOF labels are single-class; skipping PR plot.")
        return None
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pooled_ap = average_precision_score(y_true, y_proba)

    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, label=f"Pooled out-of-fold PR (AP = {pooled_ap:.3f})")
    baseline = y_true.mean()
    plt.axhline(baseline, linestyle="--", color="grey", label=f"Baseline (prevalence = {baseline:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (Pooled Out-of-Fold Predictions)")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Saved PR curve: {output_path} (pooled AP = {pooled_ap:.4f})")
    return pooled_ap


def plot_confusion_matrix_figure(y_true, y_pred, output_path: Path, title: str):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(5, 4.5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Predicted 0", "Predicted 1"],
        yticklabels=["Actual 0", "Actual 1"],
        cbar=False,
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Saved confusion matrix: {output_path}")
    return cm


def plot_feature_importance_figure(importance_df: pd.DataFrame, output_path: Path):
    plot_df = importance_df.sort_values("gain", ascending=True)
    plt.figure(figsize=(7, max(4, 0.35 * len(plot_df))))
    plt.barh(plot_df["feature"], plot_df["gain"], color="#3b6e8f")
    plt.xlabel("XGBoost Gain-Based Importance")
    plt.title("Feature Importance (Gain) -- Statistical Association Only")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Saved feature importance chart: {output_path}")


# ============================================================================
# 11. MODEL EXPLAINABILITY
# ============================================================================
def extract_gain_importance(fitted_classifier: XGBClassifier, feature_columns: list) -> pd.DataFrame:
    """
    Extract XGBoost gain-based feature importance, defaulting any feature
    never used in a split to a gain of 0.0 (rather than omitting it).
    """
    booster = fitted_classifier.get_booster()
    gain_dict = booster.get_score(importance_type="gain")
    rows = [{"feature": f, "gain": float(gain_dict.get(f, 0.0))} for f in feature_columns]
    importance_df = pd.DataFrame(rows).sort_values("gain", ascending=False).reset_index(drop=True)
    importance_df["rank"] = np.arange(1, len(importance_df) + 1)
    return importance_df


def report_features_of_interest(importance_df: pd.DataFrame, features_of_interest: list):
    logger.info("Rank / gain for scientifically noteworthy features:")
    for feat in features_of_interest:
        row = importance_df.loc[importance_df["feature"] == feat]
        if row.empty:
            logger.info(f"  {feat}: not present in this model's feature set.")
            continue
        rank = int(row["rank"].iloc[0])
        gain = float(row["gain"].iloc[0])
        logger.info(f"  {feat}: rank {rank} of {len(importance_df)}, gain = {gain:.4f}")
    logger.info(
        "Reminder: a feature ranking highly here was a USEFUL PREDICTOR in "
        "this model, under this dataset and validation setup. This is a "
        "statistical association, not proof that the underlying spectral "
        "signal is geologically caused by manganese."
    )


def run_shap_explainability(fitted_classifier: XGBClassifier, X_transformed: pd.DataFrame,
                             output_path: Path) -> bool:
    """
    Attempt to generate a SHAP summary plot. Returns True on success, False
    if SHAP is unavailable or fails for any reason (never raises).
    """
    if not USE_SHAP:
        logger.info("USE_SHAP is False; skipping SHAP explainability.")
        return False
    if not SHAP_AVAILABLE:
        logger.warning(
            f"SHAP is not available in this environment ({_SHAP_IMPORT_ERROR}). "
            "Continuing with gain-based feature importance only."
        )
        return False

    try:
        explainer = shap.TreeExplainer(fitted_classifier)
        shap_values = explainer.shap_values(X_transformed)

        # Handle both legacy (list per class) and modern (single array)
        # SHAP return conventions defensively.
        if isinstance(shap_values, list):
            values_for_plot = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        else:
            values_for_plot = shap_values

        plt.figure()
        shap.summary_plot(values_for_plot, X_transformed, show=False)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        logger.info(
            f"Saved SHAP summary plot: {output_path}. SHAP values reflect "
            "model associations learned from the training data, NOT "
            "geological causality."
        )
        return True
    except Exception as shap_error:
        logger.warning(
            f"SHAP explainability failed ({shap_error}); this can happen "
            "due to version incompatibilities between shap/xgboost. "
            "Continuing with gain-based feature importance only."
        )
        return False


# ============================================================================
# MAIN
# ============================================================================
def main():
    logger.info("=== M-MES: Spatially-Validated Manganese Prospectivity Model -- START ===")

    ensure_output_dir(OUTPUT_DIR)

    # --- Step 1: Load and validate data --------------------------------------
    data, feature_columns = load_and_validate_data()

    # --- Step 2: Spatial blocks -----------------------------------------------
    data = create_spatial_blocks(data, LONGITUDE_COLUMN, LATITUDE_COLUMN, SPATIAL_BLOCK_SIZE_DEGREES)
    n_blocks = data["block_id"].nunique()
    if n_blocks < N_SPATIAL_FOLDS:
        raise ValueError(
            f"Only {n_blocks} unique spatial block(s) were formed with "
            f"SPATIAL_BLOCK_SIZE_DEGREES={SPATIAL_BLOCK_SIZE_DEGREES}, which "
            f"is fewer than N_SPATIAL_FOLDS={N_SPATIAL_FOLDS}. Reduce "
            "N_SPATIAL_FOLDS, shrink the block size, or collect more "
            "spatially distributed samples."
        )

    X = data[feature_columns].copy()
    y = data[TARGET_COLUMN].copy()
    block_ids = data["block_id"].copy()
    coords = data[[LONGITUDE_COLUMN, LATITUDE_COLUMN]].copy()

    logger.info(f"Active feature columns ({len(feature_columns)}): {feature_columns}")
    logger.info(
        "Excluded from training by policy: "
        f"{sorted(NON_FEATURE_COLUMNS)} (coordinates are used only for "
        "spatial blocking and map output, never as direct predictors)."
    )

    # --- Step 3: Hyperparameters (baseline or tuned) --------------------------
    xgb_params = dict(DEFAULT_XGB_PARAMS)
    tuning_used = False
    if ENABLE_TUNING:
        feasible, reason = is_tuning_feasible(y, N_SPATIAL_FOLDS)
        if feasible:
            logger.info("ENABLE_TUNING=True and data volume is sufficient; running RandomizedSearchCV.")
            xgb_params, tuning_used = run_hyperparameter_tuning(X, y, block_ids, N_SPATIAL_FOLDS)
        else:
            logger.info(f"ENABLE_TUNING=True but tuning skipped: {reason} Training a baseline model instead.")
    else:
        logger.info("ENABLE_TUNING=False; training a baseline model with default hyperparameters.")

    # --- Step 4: Spatial cross-validation -------------------------------------
    fold_metrics_df, oof_df, cv_method_name = run_spatial_cross_validation(
        X, y, block_ids, coords, xgb_params, N_SPATIAL_FOLDS, threshold=CLASSIFICATION_THRESHOLD_DEFAULT
    )

    fold_metrics_path = OUTPUT_DIR / "fold_metrics.csv"
    fold_metrics_df.to_csv(fold_metrics_path, index=False)
    logger.info(f"Saved per-fold metrics: {fold_metrics_path}")

    oof_predictions_path = OUTPUT_DIR / "out_of_fold_predictions.csv"
    oof_df.to_csv(oof_predictions_path, index=False)
    logger.info(f"Saved out-of-fold predictions: {oof_predictions_path}")

    # --- Step 5: Operational threshold (reported separately) -------------------
    operational_threshold, threshold_objective = select_operational_threshold(oof_df, objective="max_f1")
    oof_df["predicted_class_operational_threshold"] = (
        oof_df["predicted_probability"] >= operational_threshold
    ).astype(int)

    # --- Step 6: Figures based on pooled OOF predictions ------------------------
    pooled_auc = plot_roc_curve(oof_df, OUTPUT_DIR / "roc_curve.png")
    pooled_ap = plot_pr_curve(oof_df, OUTPUT_DIR / "precision_recall_curve.png")
    plot_confusion_matrix_figure(
        oof_df["actual_label"], oof_df["predicted_class"],
        OUTPUT_DIR / "confusion_matrix.png",
        f"Pooled Out-of-Fold Confusion Matrix (threshold = {CLASSIFICATION_THRESHOLD_DEFAULT})",
    )

    # --- Step 7: Classification reports (default + operational threshold) ------
    classification_report_path = OUTPUT_DIR / "classification_report.txt"
    with open(classification_report_path, "w") as f:
        f.write("M-MES Manganese Surface Prospectivity Model -- Classification Report\n")
        f.write("=" * 78 + "\n\n")
        f.write(f"Cross-validation method: {cv_method_name} ({N_SPATIAL_FOLDS} folds)\n")
        f.write(f"Pooled out-of-fold ROC-AUC: {pooled_auc if pooled_auc is not None else 'n/a'}\n")
        f.write(f"Pooled out-of-fold PR-AUC (average precision): {pooled_ap if pooled_ap is not None else 'n/a'}\n\n")

        f.write(f"--- Metrics at DEFAULT threshold ({CLASSIFICATION_THRESHOLD_DEFAULT}) ---\n")
        f.write(classification_report(
            oof_df["actual_label"], oof_df["predicted_class"], zero_division=0
        ))
        f.write("\n")

        f.write(
            f"--- Metrics at OPERATIONAL threshold ({operational_threshold:.4f}, "
            f"selected via '{threshold_objective}' on pooled out-of-fold "
            "predictions -- OPTIMISTIC illustration, NOT an unbiased estimate) ---\n"
        )
        f.write(classification_report(
            oof_df["actual_label"], oof_df["predicted_class_operational_threshold"], zero_division=0
        ))
        f.write("\n")
        f.write(SCIENTIFIC_LIMITATION_STATEMENT + "\n")
    logger.info(f"Saved classification report: {classification_report_path}")

    # --- Step 8: Final model trained on ALL validated labelled samples ---------
    final_scale_pos_weight = compute_scale_pos_weight(y)
    final_pipeline = build_pipeline(xgb_params)
    final_pipeline.set_params(classifier__scale_pos_weight=final_scale_pos_weight)
    final_pipeline.fit(X, y)
    logger.info(
        f"Final model trained on all {len(X)} validated rows "
        f"(scale_pos_weight={final_scale_pos_weight:.4f})."
    )

    final_classifier = final_pipeline.named_steps["classifier"]
    X_transformed_full = final_pipeline.named_steps["imputer"].transform(X)

    # --- Step 9: Feature importance (gain-based) --------------------------------
    importance_df = extract_gain_importance(final_classifier, feature_columns)
    importance_csv_path = OUTPUT_DIR / "feature_importance_gain.csv"
    importance_df.to_csv(importance_csv_path, index=False)
    logger.info(f"Saved gain-based feature importance: {importance_csv_path}")
    plot_feature_importance_figure(importance_df, OUTPUT_DIR / "feature_importance_gain.png")
    report_features_of_interest(importance_df, FEATURES_OF_INTEREST)

    # --- Step 10: SHAP (optional) -------------------------------------------
    shap_succeeded = run_shap_explainability(
        final_classifier, X_transformed_full, OUTPUT_DIR / "shap_summary.png"
    )

    # --- Step 11: Save model, pipeline, and metadata ----------------------------
    final_classifier.save_model(str(MODEL_OUTPUT_PATH))
    logger.info(f"Saved raw XGBoost model: {MODEL_OUTPUT_PATH}")

    joblib.dump(final_pipeline, PIPELINE_OUTPUT_PATH)
    logger.info(f"Saved complete preprocessing+model pipeline: {PIPELINE_OUTPUT_PATH}")

    fold_roc_auc_values = fold_metrics_df["roc_auc"].dropna()
    fold_pr_auc_values = fold_metrics_df["pr_auc_average_precision"].dropna()
    n_single_class_folds = int((~fold_metrics_df["val_has_both_classes"]).sum())

    metadata = {
        "generation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "feature_list_and_order": feature_columns,
        "training_row_count": int(len(X)),
        "class_distribution": {
            "negative_0": int((y == 0).sum()),
            "positive_1": int((y == 1).sum()),
        },
        "spatial_block_configuration": {
            "block_size_degrees": SPATIAL_BLOCK_SIZE_DEGREES,
            "n_unique_blocks": int(n_blocks),
        },
        "cross_validation_method": cv_method_name,
        "n_spatial_folds": N_SPATIAL_FOLDS,
        "n_folds_with_single_class_validation_set": n_single_class_folds,
        "mean_validation_metrics": {
            "roc_auc_mean": float(fold_roc_auc_values.mean()) if len(fold_roc_auc_values) else None,
            "roc_auc_std": float(fold_roc_auc_values.std()) if len(fold_roc_auc_values) > 1 else None,
            "pr_auc_mean": float(fold_pr_auc_values.mean()) if len(fold_pr_auc_values) else None,
            "pr_auc_std": float(fold_pr_auc_values.std()) if len(fold_pr_auc_values) > 1 else None,
            "precision_mean": float(fold_metrics_df["precision"].mean()),
            "recall_mean": float(fold_metrics_df["recall"].mean()),
            "f1_mean": float(fold_metrics_df["f1"].mean()),
            "balanced_accuracy_mean": float(fold_metrics_df["balanced_accuracy"].mean()),
            "pooled_oof_roc_auc": float(pooled_auc) if pooled_auc is not None else None,
            "pooled_oof_pr_auc": float(pooled_ap) if pooled_ap is not None else None,
        },
        "default_classification_threshold": CLASSIFICATION_THRESHOLD_DEFAULT,
        "operational_threshold": {
            "value": float(operational_threshold),
            "selection_objective": threshold_objective,
            "selected_from": "pooled_out_of_fold_predictions",
            "caveat": (
                "Selected using the same out-of-fold predictions used for "
                "reporting; metrics at this threshold are an optimistic "
                "operational illustration, not an unbiased future-performance "
                "estimate."
            ),
        },
        "hyperparameter_tuning_enabled": bool(ENABLE_TUNING),
        "hyperparameter_tuning_used": bool(tuning_used),
        "model_hyperparameters": xgb_params,
        "shap_explainability_generated": bool(shap_succeeded),
        "scientific_limitation_statement": SCIENTIFIC_LIMITATION_STATEMENT,
    }

    with open(METADATA_OUTPUT_PATH, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info(f"Saved model metadata: {METADATA_OUTPUT_PATH}")

    # --- Step 12: Console summary ---------------------------------------------
    logger.info("=== SUMMARY ===")
    logger.info(f"Total samples used: {len(X)} (negative={int((y == 0).sum())}, positive={int((y == 1).sum())})")
    logger.info(f"Unique spatial blocks: {n_blocks}")
    logger.info(f"Cross-validation: {cv_method_name}, {N_SPATIAL_FOLDS} folds")
    if len(fold_roc_auc_values):
        logger.info(
            f"Fold ROC-AUC: mean={fold_roc_auc_values.mean():.4f}, "
            f"std={fold_roc_auc_values.std():.4f} "
            f"(calculable in {len(fold_roc_auc_values)}/{N_SPATIAL_FOLDS} folds)"
        )
    else:
        logger.info("Fold ROC-AUC: not calculable in any fold (all folds had single-class validation sets).")
    if len(fold_pr_auc_values):
        logger.info(
            f"Fold PR-AUC: mean={fold_pr_auc_values.mean():.4f}, "
            f"std={fold_pr_auc_values.std():.4f}"
        )
    logger.info(
        f"Mean precision={fold_metrics_df['precision'].mean():.4f}, "
        f"recall={fold_metrics_df['recall'].mean():.4f}, "
        f"F1={fold_metrics_df['f1'].mean():.4f}, "
        f"balanced accuracy={fold_metrics_df['balanced_accuracy'].mean():.4f}"
    )
    if pooled_auc is not None:
        logger.info(f"Pooled out-of-fold ROC-AUC: {pooled_auc:.4f}")
    if pooled_ap is not None:
        logger.info(f"Pooled out-of-fold PR-AUC: {pooled_ap:.4f}")
    if n_single_class_folds > 0:
        logger.warning(
            f"WARNING: {n_single_class_folds} of {N_SPATIAL_FOLDS} fold(s) had a "
            "single-class validation set; ROC-AUC/PR-AUC for those folds were "
            "skipped. Interpret aggregate metrics with this in mind."
        )
    per_block_counts = data.groupby("block_id").size()
    if per_block_counts.max() > 5 * per_block_counts.median():
        logger.warning(
            "WARNING: spatial block sizes are highly uneven (some blocks "
            "contain far more samples than others). This can make individual "
            "folds unrepresentative; consider adjusting SPATIAL_BLOCK_SIZE_DEGREES."
        )

    logger.info(SCIENTIFIC_LIMITATION_STATEMENT)
    logger.info("=== M-MES: Spatially-Validated Manganese Prospectivity Model -- END ===")


if __name__ == "__main__":
    main()
