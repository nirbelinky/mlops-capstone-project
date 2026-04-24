"""
feature_engineering.py — Deterministic feature transforms for the capstone pipeline.

Pipeline position: **Step C** (called in ``flow.py`` after integrity checks pass).

Purpose
-------
This module converts raw NYC TLC Green Taxi trip records into a clean,
model-ready DataFrame with a **stable, predictable schema**.  Every entry
point in the pipeline — training, evaluation, and (future) inference — calls
:func:`engineer_features` so the model always sees exactly the same columns
in exactly the same order.  This eliminates a common class of ML bugs where
training and serving disagree on feature names, types, or transformations.

Why a stable schema matters
---------------------------
* **Reproducibility** — the same raw parquet always produces the same
  feature table (given the same config values), making experiments
  comparable across runs.
* **Registry compatibility** — a model registered in MLflow expects a
  fixed input signature.  If the feature schema drifts between training
  and serving, ``mlflow.pyfunc.predict`` will raise at inference time.
* **Monitoring** — NannyML and drift checks rely on column names and
  dtypes being consistent between the reference set and incoming batches.

Feature engineering strategy
----------------------------
1. **Filter** to credit-card trips (``payment_type == 1``) because the
   TLC meter only records tip amounts for card payments; cash tips are
   unobserved and would appear as zeros, biasing the model.
2. **Extract temporal features** (hour, day-of-week, month) from the
   pickup timestamp to capture time-of-day and seasonal tipping patterns.
3. **Derive trip duration** in minutes from pickup/dropoff timestamps.
4. **Clip** numeric columns to physically plausible bounds (from
   ``config.py``) to limit the influence of data-entry errors without
   discarding entire rows.
5. **Log-transform** trip distance (``log1p``) to compress the heavy
   right tail and make the distribution more symmetric for tree-based
   learners.
6. **Cast** location IDs to ``category`` dtype so XGBoost can use its
   native categorical split algorithm instead of treating zone IDs as
   ordered integers.

All column names and clipping bounds are imported from ``config.py`` —
no magic strings or numbers leak into this module.

Design choices
--------------
* **Credit-card filter** — tips are only reliably recorded for card payments
  (``payment_type == 1``), matching the approach in
  ``06_monitoring_data_drift/green_taxi_drift_lib.make_tip_frame``.
* **Clipping** before log-transform avoids extreme outliers while keeping the
  distribution shape intact.
* **Stable output schema** — the returned DataFrame always contains exactly
  ``get_feature_columns() + [TARGET_COL]``, regardless of what extra columns
  the raw input carries.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# All column names, feature lists, and clipping thresholds are imported from
# the central configuration module.  This ensures that if a threshold changes
# (e.g. MAX_TRIP_DISTANCE is raised from 200 to 300), both the integrity
# checks and the feature engineering step pick up the new value automatically.
from config import (
    TARGET_COL,
    DATETIME_COL,
    DROPOFF_DATETIME_COL,
    REQUIRED_COLUMNS,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TIME_FEATURES,
    MAX_TRIP_DISTANCE,
    MAX_FARE_AMOUNT,
    MAX_TIP_AMOUNT,
    MIN_TRIP_DISTANCE,
    MAX_PASSENGER_COUNT,
    MAX_DURATION_MIN,
    CREDIT_CARD_PAYMENT_TYPE,
)


# ── public API ────────────────────────────────────────────────────────────


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Transform a raw Green Taxi dataframe into a model-ready table.

    This is the **single entry point** for all feature engineering in the
    pipeline.  Both the reference set and every incoming batch pass through
    this function, guaranteeing schema consistency.

    Transformation pipeline (executed in order)
    --------------------------------------------
    1. **Validate required columns** — fail fast if the raw data is missing
       any column listed in ``config.REQUIRED_COLUMNS``.
    2. **Parse datetime columns** — convert pickup and dropoff strings to
       ``datetime64`` so ``.dt`` accessors work.  Unparseable values become
       ``NaT`` (``errors="coerce"``).
    3. **Credit-card filter** — keep only rows where ``payment_type == 1``
       because the TLC meter does not record cash tips; including cash trips
       would flood the dataset with artificial zero-tip rows.
    4. **Remove invalid rows** — drop rows with negative tip or negative
       distance, which are physically impossible and indicate data errors.
    5. **Extract time features** — derive ``pickup_hour``, ``pickup_dayofweek``,
       and ``pickup_month`` from the pickup timestamp.  These capture
       temporal tipping patterns (e.g. higher tips during rush hour).
    6. **Compute trip duration** — ``(dropoff − pickup)`` in minutes, clipped
       to ``[0, MAX_DURATION_MIN]``.  If the dropoff column is missing,
       duration is set to ``NaN`` (and the row will be dropped later).
    7. **Clip numeric columns** — cap ``trip_distance``, ``fare_amount``,
       ``passenger_count``, and ``tip_amount`` to the bounds defined in
       ``config.py``.  Clipping is preferred over dropping because it
       preserves the row while limiting outlier influence.
    8. **Log-transform trip distance** — ``log1p(trip_distance)`` compresses
       the heavy right tail of the distance distribution, reducing the
       impact of extreme values on tree splits and making the feature
       more informative for the model.
    9. **Cast categorical location IDs** — convert ``PULocationID`` and
       ``DOLocationID`` to pandas ``category`` dtype so XGBoost can use
       native categorical splits (no one-hot encoding needed).
    10. **Select stable output schema** — keep only the columns in
        ``get_feature_columns() + [TARGET_COL]`` and drop any rows that
        still contain ``NaN`` in feature columns.

    Parameters
    ----------
    df : pd.DataFrame
        Raw trip-record data (e.g. a single month parquet from TLC).
        Must contain at least the columns listed in
        ``config.REQUIRED_COLUMNS``.

    Returns
    -------
    pd.DataFrame
        A clean DataFrame with columns =
        ``get_feature_columns() + [TARGET_COL]``.
        Rows with invalid target/distance or NaN features are dropped.
        Index is reset to a contiguous ``RangeIndex``.

    Raises
    ------
    ValueError
        If any column listed in ``config.REQUIRED_COLUMNS`` is missing
        from the input DataFrame.

    Examples
    --------
    >>> import pandas as pd
    >>> from feature_engineering import engineer_features
    >>> raw = pd.read_parquet("data/reference_2024-01.parquet")
    >>> clean = engineer_features(raw)
    >>> clean.columns.tolist()
    ['trip_distance', 'fare_amount', 'passenger_count',
     'log1p_trip_distance', 'duration_min',
     'PULocationID', 'DOLocationID',
     'pickup_hour', 'pickup_dayofweek', 'pickup_month',
     'tip_amount']
    """
    # Work on a copy so the caller's original DataFrame is never mutated.
    out = df.copy()

    # ── 1. validate required columns ──────────────────────────────────
    # Fail fast: if the raw data is missing any expected column, raise
    # immediately rather than producing a subtly broken feature table.
    missing = [c for c in REQUIRED_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(
            f"Raw data is missing required columns: {missing}"
        )

    # ── 2. parse datetimes ────────────────────────────────────────────
    # Convert pickup and dropoff columns from string/object to proper
    # datetime64 dtype.  ``errors="coerce"`` turns unparseable values
    # into NaT instead of raising, so downstream steps can handle them
    # gracefully (e.g. NaT.dt.hour → NaN, which gets dropped later).
    for col in (DATETIME_COL, DROPOFF_DATETIME_COL):
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")

    # ── 3. credit-card filter ─────────────────────────────────────────
    # Keep only credit-card payments (payment_type == 1).
    # Rationale: the TLC taxi meter records tip amounts only for card
    # transactions.  Cash tips are not captured, so cash rows show
    # tip_amount == 0 even when a tip was given.  Including them would
    # teach the model that most trips have zero tip, which is wrong.
    # This matches the filtering in green_taxi_drift_lib.make_tip_frame.
    if "payment_type" in out.columns:
        out = out[out["payment_type"] == CREDIT_CARD_PAYMENT_TYPE].copy()

    # ── 4. remove invalid rows ────────────────────────────────────────
    # Coerce target and distance to numeric (handles any stray strings),
    # then drop rows where either is negative — these are physically
    # impossible and indicate data-entry or ETL errors.
    out[TARGET_COL] = pd.to_numeric(out[TARGET_COL], errors="coerce")
    out["trip_distance"] = pd.to_numeric(out["trip_distance"], errors="coerce")
    out = out[out[TARGET_COL] >= 0].copy()       # negative tip is impossible
    out = out[out["trip_distance"] >= 0].copy()   # negative distance is impossible

    # ── 5. time features ─────────────────────────────────────────────
    # Extract hour, day-of-week, and month from the pickup timestamp.
    # These integer features let the model learn temporal patterns:
    #   - pickup_hour (0–23): rush-hour vs late-night tipping behaviour
    #   - pickup_dayofweek (0=Mon … 6=Sun): weekday vs weekend patterns
    #   - pickup_month (1–12): seasonal variation in taxi usage / tourism
    # Stored as nullable Int64 so that NaT timestamps produce <NA>
    # instead of raising; those rows are dropped in step 10.
    dt = out[DATETIME_COL]
    out["pickup_hour"] = dt.dt.hour.astype("Int64")
    out["pickup_dayofweek"] = dt.dt.dayofweek.astype("Int64")
    out["pickup_month"] = dt.dt.month.astype("Int64")

    # ── 6. trip duration ──────────────────────────────────────────────
    # Compute elapsed time in minutes between pickup and dropoff.
    # Clipped to [0, MAX_DURATION_MIN] (default 360 min = 6 hours) to
    # cap unrealistic durations caused by meters left running or data
    # errors.  If the dropoff column is absent, duration defaults to NaN
    # and the row will be dropped in step 10.
    if DROPOFF_DATETIME_COL in out.columns:
        dur = (
            out[DROPOFF_DATETIME_COL] - out[DATETIME_COL]
        ).dt.total_seconds() / 60.0
        out["duration_min"] = dur.clip(lower=0.0, upper=MAX_DURATION_MIN)
    else:
        out["duration_min"] = np.nan

    # ── 7. clip numerics ──────────────────────────────────────────────
    # Cap numeric columns to the physically plausible bounds defined in
    # config.py.  Clipping (rather than dropping) preserves the row
    # while limiting the influence of extreme outliers on model training.
    # The bounds are aligned with RANGE_SPECS in green_taxi_drift_lib.py.

    # Trip distance: clip to [0, 200] miles (default).
    out["trip_distance"] = out["trip_distance"].clip(
        lower=MIN_TRIP_DISTANCE, upper=MAX_TRIP_DISTANCE
    )
    # Fare amount: coerce to numeric first (handles stray strings),
    # then clip to [0, 500] dollars (default).
    out["fare_amount"] = pd.to_numeric(
        out["fare_amount"], errors="coerce"
    ).clip(lower=0.0, upper=MAX_FARE_AMOUNT)
    # Passenger count: clip to [0, 10] (default).  NYC taxis seat ~6
    # passengers; 10 is a generous ceiling.
    out["passenger_count"] = pd.to_numeric(
        out["passenger_count"], errors="coerce"
    ).clip(lower=0.0, upper=MAX_PASSENGER_COUNT)
    # Tip amount (target): clip to [0, 200] dollars (default).
    # Capping the target prevents extreme tips from dominating the loss.
    out[TARGET_COL] = out[TARGET_COL].clip(lower=0.0, upper=MAX_TIP_AMOUNT)

    # ── 8. log-transform ──────────────────────────────────────────────
    # Apply log(1 + x) to trip_distance.  The raw distance distribution
    # has a heavy right tail (most trips are short, a few are very long).
    # The log transform compresses this tail, making the feature more
    # symmetric and giving tree-based models finer resolution in the
    # dense low-distance region where most data lives.
    # Using log1p (instead of log) avoids log(0) = -inf for zero-distance trips.
    out["log1p_trip_distance"] = np.log1p(out["trip_distance"])

    # ── 9. categorical location IDs ───────────────────────────────────
    # Cast PULocationID and DOLocationID to int.  XGBoost 3.x with
    # enable_categorical=True requires category dtype at both train and
    # predict time, but MLflow pyfunc schema enforcement cannot convert
    # category → int32.  Using plain int avoids this conflict while
    # still letting tree-based models learn useful splits on zone IDs.
    for col in CATEGORICAL_FEATURES:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype(int)

    # ── 10. select stable schema ──────────────────────────────────────
    # Keep only the columns the model expects (features + target) and
    # discard everything else (payment_type, raw datetimes, total_amount,
    # etc.).  This enforces schema stability: no matter what extra
    # columns the raw parquet contains, the output is always the same.
    output_cols = get_feature_columns() + [TARGET_COL]
    out = out[output_cols].copy()

    # Drop any rows that still have NaN in critical feature columns.
    # This catches residual NaNs from unparseable datetimes (step 2),
    # missing passenger counts, or absent dropoff timestamps (step 6).
    out = out.dropna(subset=get_feature_columns()).reset_index(drop=True)

    return out


def get_feature_columns() -> list[str]:
    """Return the ordered list of feature column names the model expects.

    The order is: numeric features, then categorical features, then time
    features.  This ordering is defined by the lists in ``config.py``
    (``NUMERIC_FEATURES``, ``CATEGORICAL_FEATURES``, ``TIME_FEATURES``).

    The target column (``tip_amount``) is **not** included — use
    ``config.TARGET_COL`` separately when you need it.

    Returns
    -------
    list[str]
        Concatenation of ``NUMERIC_FEATURES + CATEGORICAL_FEATURES +
        TIME_FEATURES``.  For the default config this is::

            ['trip_distance', 'fare_amount', 'passenger_count',
             'log1p_trip_distance', 'duration_min',
             'PULocationID', 'DOLocationID',
             'pickup_hour', 'pickup_dayofweek', 'pickup_month']

    Notes
    -----
    This function is called by :func:`engineer_features` (step 10) to
    select the output columns, and by ``flow.py`` to split ``X`` from
    ``y`` before training and evaluation.
    """
    return NUMERIC_FEATURES + CATEGORICAL_FEATURES + TIME_FEATURES


def get_feature_spec() -> dict[str, str]:
    """Return a JSON-serialisable dict mapping each feature name to its dtype.

    This specification is logged as an MLflow artifact (``feature_spec.json``)
    so that downstream consumers — monitoring dashboards, serving endpoints,
    data-quality checks — can validate that incoming data matches the schema
    the model was trained on.

    The dtype mapping is:

    * Numeric features → ``"float64"``
    * Categorical features → ``"category"``
    * Time features → ``"Int64"`` (nullable integer)

    Returns
    -------
    dict[str, str]
        Feature-name → dtype-string mapping, e.g.::

            {
                "trip_distance": "float64",
                "fare_amount": "float64",
                "passenger_count": "float64",
                "log1p_trip_distance": "float64",
                "duration_min": "float64",
                "PULocationID": "category",
                "DOLocationID": "category",
                "pickup_hour": "Int64",
                "pickup_dayofweek": "Int64",
                "pickup_month": "Int64",
            }

    Notes
    -----
    This function is called in ``flow.py`` during the feature-engineering
    step to produce the ``feature_spec.json`` artifact that is logged to
    the MLflow run.
    """
    spec: dict[str, str] = {}

    # Continuous numeric features are stored as 64-bit floats.
    for col in NUMERIC_FEATURES:
        spec[col] = "float64"

    # Categorical features (taxi zone IDs) are stored as int.
    for col in CATEGORICAL_FEATURES:
        spec[col] = "int64"

    # Time features are nullable integers (Int64) to handle NaT gracefully.
    for col in TIME_FEATURES:
        spec[col] = "Int64"

    return spec
