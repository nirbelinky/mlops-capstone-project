"""
config.py — Central configuration for the MLOps capstone project.

Why centralise configuration?
-----------------------------
Every tuneable constant — column names, feature lists, clipping bounds,
integrity thresholds, and model-gate parameters — lives in this single file.
This gives the project a **single source of truth**: when you need to adjust
a threshold (e.g. make the integrity gate more lenient), you change it here
and every module that imports from ``config`` picks up the new value
automatically.  No grep-and-replace across multiple files, no risk of
inconsistent magic numbers.

How other modules use this file
-------------------------------
Modules import either the whole module or individual names::

    import config                          # flow.py, integrity_checks.py,
                                           # model_utils.py, decision_logger.py
    from config import TARGET_COL, ...     # feature_engineering.py

The constants are grouped into logical sections below.  Each section
corresponds to a different stage of the pipeline (integrity checking,
feature engineering, model evaluation, promotion).

Thresholds are aligned with the range specs proven in
``06_monitoring_data_drift/green_taxi_drift_lib.py``.
"""

from __future__ import annotations

# ===========================================================================
# MLflow settings
# ---------------------------------------------------------------------------
# These configure where experiment tracking data is stored and how the
# registered model is named.  Used primarily by model_utils.py (to
# initialise the tracking URI and experiment) and flow.py (to look up
# the champion model by name).
# ===========================================================================

# File-based SQLite backend for MLflow tracking.  The relative path
# points to the shared ``mlflow_tracking/`` directory one level above
# this project folder.  Change to an HTTP URI (e.g. "http://localhost:5000")
# if you switch to a remote tracking server.
# Used by: model_utils.setup_mlflow(), flow.py
MLFLOW_TRACKING_URI = "sqlite:///../mlflow_tracking/mlflow.db"

# Experiment name under which all runs for this project are grouped.
# Changing this creates a new experiment in the MLflow UI; existing runs
# remain under the old name.
# Used by: model_utils.setup_mlflow()
MLFLOW_EXPERIMENT_NAME = "green_taxi_capstone"

# Registered-model name in the MLflow Model Registry.  The pipeline uses
# aliases ("champion", "candidate") on this model to manage promotion.
# Renaming it effectively starts a fresh model lineage.
# Used by: flow.py (load_champion, retrain, promote steps)
MODEL_NAME = "green_taxi_tip_model"

# ===========================================================================
# Column names (raw data)
# ---------------------------------------------------------------------------
# Canonical names for columns in the raw NYC TLC Green Taxi parquet files.
# Keeping them here avoids scattering string literals across the codebase.
# ===========================================================================

# The regression target — dollar amount of the tip.
# Used by: feature_engineering.py (clipping, schema), flow.py (train/eval splits)
TARGET_COL = "tip_amount"

# Pickup timestamp — used to derive time-based features (hour, day-of-week,
# month) and to compute trip duration together with the dropoff column.
# Used by: feature_engineering.py, integrity_checks.py, flow.py
DATETIME_COL = "lpep_pickup_datetime"

# Dropoff timestamp — paired with DATETIME_COL to calculate trip duration
# in minutes.  Also validated in integrity_checks.py (dropoff >= pickup).
# Used by: feature_engineering.py, integrity_checks.py
DROPOFF_DATETIME_COL = "lpep_dropoff_datetime"

# ---------------------------------------------------------------------------
# Required raw columns
# ---------------------------------------------------------------------------
# Hard integrity check (integrity_checks.run_hard_checks) rejects the entire
# batch if any of these columns are missing.  This list mirrors the subset of
# ``EXPECTED_SCHEMA`` from green_taxi_drift_lib.py that the capstone pipeline
# actually needs.  Adding a column here means the pipeline will refuse to
# proceed unless the raw data contains it.
# Used by: integrity_checks.py (schema validation), feature_engineering.py
#          (column-presence guard)
REQUIRED_COLUMNS: list[str] = [
    "lpep_pickup_datetime",       # pickup timestamp — time features + duration
    "lpep_dropoff_datetime",      # dropoff timestamp — duration calculation
    "PULocationID",               # pickup taxi zone — categorical feature
    "DOLocationID",               # dropoff taxi zone — categorical feature
    "trip_distance",              # odometer distance — numeric feature
    "passenger_count",            # number of passengers — numeric feature
    "tip_amount",                 # regression target
    "fare_amount",                # meter fare — numeric feature
    "total_amount",               # total charge — used in integrity range checks
    "payment_type",               # payment method — used to filter credit-card trips
]

# ===========================================================================
# Feature columns produced by feature engineering (stable schema)
# ---------------------------------------------------------------------------
# These lists define the exact set of columns the trained model expects as
# input.  feature_engineering.engineer_features() produces a DataFrame whose
# columns are exactly  NUMERIC_FEATURES + CATEGORICAL_FEATURES + TIME_FEATURES
# + [TARGET_COL].  Changing these lists changes the model's input schema, so
# any existing registered model will be incompatible — you would need to
# retrain from scratch.
# Used by: feature_engineering.py (get_feature_columns, get_feature_spec),
#          flow.py (to split X/y for training and evaluation)
# ===========================================================================

# Continuous numeric features fed directly to the model.
# - trip_distance, fare_amount, passenger_count: raw numeric columns (clipped)
# - log1p_trip_distance: log(1 + trip_distance), reduces right-skew
# - duration_min: derived from pickup/dropoff timestamps, clipped to [0, 360]
NUMERIC_FEATURES: list[str] = [
    "trip_distance",
    "fare_amount",
    "passenger_count",
    "log1p_trip_distance",
    "duration_min",
]

# Taxi-zone IDs treated as unordered categories.  XGBoost's
# ``enable_categorical=True`` handles them natively without one-hot encoding.
# Adding more categorical columns here requires matching changes in
# engineer_features() to cast them to ``category`` dtype.
CATEGORICAL_FEATURES: list[str] = [
    "PULocationID",
    "DOLocationID",
]

# Integer features extracted from the pickup timestamp.
# They capture temporal patterns (rush-hour, weekday vs weekend, seasonality).
# Stored as nullable ``Int64`` to handle any NaT values gracefully.
TIME_FEATURES: list[str] = [
    "pickup_hour",        # 0–23; captures rush-hour / late-night patterns
    "pickup_dayofweek",   # 0=Mon … 6=Sun; weekday vs weekend tipping behaviour
    "pickup_month",       # 1–12; seasonal variation in taxi usage
]

# ===========================================================================
# Hard integrity thresholds
# ---------------------------------------------------------------------------
# Physical / business-logic bounds used by integrity_checks.run_hard_checks()
# to flag or reject batches with extreme values.  These are aligned with the
# RANGE_SPECS dictionary in green_taxi_drift_lib.py so that the capstone
# pipeline and the drift-monitoring notebook agree on what "valid" means.
#
# Also used by feature_engineering.py to clip numeric columns before they
# reach the model — any value beyond these bounds is capped, not dropped.
#
# Used by: integrity_checks.py (range validation), feature_engineering.py
#          (numeric clipping), decision_logger.py (audit trail)
# ===========================================================================

# Maximum plausible trip distance in miles.  NYC taxi trips rarely exceed
# ~50 miles; 200 is a generous upper bound that catches data-entry errors
# (e.g. 9999) while keeping legitimate long trips.
# Increase → more lenient (allows longer trips); decrease → stricter filtering.
MAX_TRIP_DISTANCE: float = 200.0   # miles  (drift_lib uses 200)

# Maximum plausible fare in dollars.  Fares above $500 almost certainly
# indicate a data error or a negotiated flat-rate that doesn't reflect
# normal metered behaviour.
# Increase → more lenient; decrease → stricter.
MAX_FARE_AMOUNT: float = 500.0     # dollars (drift_lib uses 500)

# Maximum plausible tip in dollars.  Tips above $200 are extremely rare
# and likely erroneous.  This also clips the target variable to prevent
# outlier tips from dominating the loss function during training.
# Increase → allows larger tips in training; decrease → tighter target range.
MAX_TIP_AMOUNT: float = 200.0      # dollars (drift_lib uses 200)

# Minimum trip distance.  Negative distances are physically impossible;
# this floor catches sign errors in the source data.
# Typically left at 0.0.
MIN_TRIP_DISTANCE: float = 0.0     # cannot be negative

# Maximum passenger count.  NYC taxis seat at most ~6 passengers; 10 is
# a generous ceiling that catches obvious data errors (e.g. 255).
# Increase → more lenient; decrease → stricter.
MAX_PASSENGER_COUNT: float = 10.0  # drift_lib uses 10

# Maximum trip duration in minutes (6 hours).  Trips longer than this are
# almost certainly the result of a meter left running or a data error.
# Used to clip the derived ``duration_min`` feature.
# Increase → allows longer durations; decrease → stricter.
MAX_DURATION_MIN: float = 360.0    # 6 hours (drift_lib uses 360)

# ===========================================================================
# Soft integrity thresholds (NannyML layer)
# ---------------------------------------------------------------------------
# These thresholds power the "soft" integrity checks in
# integrity_checks.run_soft_checks().  Unlike hard checks (which reject the
# batch outright), soft checks produce warnings that are logged but do not
# block the pipeline.  They detect data-quality drift between the reference
# set and the incoming batch.
#
# Used by: integrity_checks.py (soft checks), decision_logger.py (audit trail)
# ===========================================================================

# If the missing-value rate for any column increases by more than this
# fraction (batch rate − reference rate), a soft warning is raised.
# Example: reference has 2 % missing, batch has 13 % → increase = 11 % > 10 %.
# Increase → fewer warnings (more tolerant of missing data);
# decrease → more sensitive to missingness spikes.
MISSINGNESS_SPIKE_THRESHOLD: float = 0.10   # 10 % increase in missing rate

# If more than this fraction of a categorical column's batch values are
# categories never seen in the reference set, a soft warning is raised.
# Example: 6 % of PULocationID values are new zones → 6 % > 5 % → warning.
# Increase → more tolerant of new categories; decrease → stricter.
UNSEEN_CATEGORICAL_THRESHOLD: float = 0.05  # 5 % unseen categories

# ===========================================================================
# Model gate thresholds
# ---------------------------------------------------------------------------
# These control the retrain and promotion decisions in model_utils.py.
# The retrain gate (model_utils.should_retrain) decides whether the champion
# model's performance has degraded enough to justify retraining.
# The promotion gate (model_utils.should_promote) decides whether a newly
# trained candidate model is good enough to replace the current champion.
#
# Used by: model_utils.py (should_retrain, should_promote),
#          decision_logger.py (audit trail)
# ===========================================================================

# Minimum relative improvement the candidate must show over the champion.
# Candidate RMSE must be < champion_RMSE × (1 − MIN_IMPROVEMENT).
# Example: champion RMSE = 1.00 → candidate must achieve < 0.99.
# Increase → harder to promote (requires bigger improvement);
# decrease → easier to promote (even marginal gains suffice).
MIN_IMPROVEMENT: float = 0.01               # 1 % — candidate must beat champion

# If the champion's RMSE on the new batch exceeds its RMSE on the reference
# set by more than this fraction, retraining is triggered.
# Example: ref RMSE = 2.63, batch RMSE = 2.86 → degradation ≈ 8.7 % > 5 %.
# Increase → more tolerant of performance drops (retrain less often);
# decrease → more sensitive (retrain sooner).
RMSE_DEGRADATION_THRESHOLD: float = 0.05    # 5 % RMSE increase triggers retrain

# ===========================================================================
# Stability / promotion checks
# ---------------------------------------------------------------------------
# An additional safety net for the promotion gate: even if the candidate
# beats the champion on the new batch, it must not perform significantly
# worse on the reference data.  This prevents "overfitting to drift" —
# a candidate that adapts to a noisy batch but forgets the baseline.
#
# Used by: model_utils.should_promote() (P3 stability check),
#          decision_logger.py (audit trail)
# ===========================================================================

# Maximum allowed regression on the reference set.  Candidate RMSE on
# reference must be ≤ champion_RMSE_on_ref × (1 + tolerance).
# Example: champion ref RMSE = 1.60 → candidate ref RMSE must be ≤ 2.40.
# Increase → more lenient (allows larger regression on reference);
# decrease → stricter (candidate must closely match champion on reference).
#
# NOTE: 50 % is appropriate because the candidate is trained on combined
# (reference + batch) data, so it naturally trades some reference-set
# accuracy for better generalisation on new batches.  A tight tolerance
# (e.g. 5 %) would block every candidate that adapts to distribution
# shift, defeating the purpose of retraining.
REFERENCE_REGRESSION_TOLERANCE: float = 0.50  # candidate can't regress reference > 50 %

# ===========================================================================
# Credit-card filter
# ---------------------------------------------------------------------------
# The NYC TLC data only records tip amounts reliably for credit-card
# payments (payment_type == 1).  Cash tips are not captured by the meter,
# so including cash trips would introduce a large number of zero-tip rows
# that don't reflect actual tipping behaviour.  This constant is used by
# feature_engineering.py to filter the dataset before training/evaluation.
#
# Used by: feature_engineering.py (credit-card filter step)
# ===========================================================================

# Integer code for credit-card payments in the TLC data dictionary.
# 1 = Credit card, 2 = Cash, 3 = No charge, 4 = Dispute, 5 = Unknown.
# Changing this value would select a different payment type — almost
# certainly not what you want unless the TLC schema changes.
CREDIT_CARD_PAYMENT_TYPE: int = 1
