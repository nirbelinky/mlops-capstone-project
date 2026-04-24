"""
integrity_checks.py — Two-layer integrity gate for the MLOps capstone pipeline.

Implements **Step B (Integrity Gate)** from the design doc
(``08_mlops_capstone_project/design_doc.md``).

Architecture — two layers
-------------------------
The gate is split into two complementary layers:

**Layer 1 — Hard rules (fail-fast)**
    Deterministic, schema-level checks that run on the *raw* batch DataFrame
    *before* any feature engineering.  If any hard rule fails the entire batch
    is rejected and the pipeline short-circuits — there is no point in
    training a model on data that is structurally broken.

    Hard rules cover:
    • Required-column presence (schema contract).
    • Datetime parseability and temporal ordering (pickup ≤ dropoff).
    • Impossible / extreme numeric values (negative distances, fares beyond
      physical limits) — flagged only when they exceed a 1 % fraction of
      the batch, not on individual rows.
    • Target-column (``tip_amount``) missingness above 50 %.

**Layer 2 — Soft checks (warn-only)**
    Statistical comparisons between the incoming batch and a *reference*
    dataset (typically the training-era data).  Soft checks never block the
    pipeline; instead they set an ``integrity_warn`` flag that downstream
    decision functions (``model_utils.should_retrain``,
    ``model_utils.should_promote``) use to tighten their thresholds.

    Soft checks cover:
    • Per-column missingness spikes (batch vs reference).
    • Unseen categorical values (taxi-zone IDs not present in reference).
    • Univariate distribution drift on key numeric columns.

Why run on RAW data?
--------------------
Integrity checks operate on the raw DataFrame (straight from the parquet
file) rather than on the feature-engineered version.  This is intentional:

1. **Catch schema / pipeline issues early** — a missing column or corrupt
   datetime should be detected before we spend time computing derived
   features.
2. **Avoid masking problems** — feature engineering clips and transforms
   values; running checks *after* clipping would hide the very anomalies
   we want to detect.
3. **Separation of concerns** — integrity validation is a data-quality
   concern, not a modelling concern.

Graceful degradation — NannyML → scipy fallback
------------------------------------------------
The soft-check drift layer first tries to use NannyML's
``UnivariateDriftCalculator``, which provides production-grade drift
detection with automatic threshold calibration.  If NannyML is not
installed (it is an optional heavy dependency) or raises any error at
runtime, the module transparently falls back to ``scipy.stats.ks_2samp``
(the two-sample Kolmogorov–Smirnov test) which is always available in the
conda environment.  This ensures the pipeline never crashes due to a
missing optional library.

Public API
----------
- ``run_hard_checks(batch_df)``        → ``(passed, failures)``
- ``run_soft_checks(reference_df, batch_df)`` → ``(has_warnings, report)``
- ``run_integrity_checks(reference_df, batch_df)``
      → ``(hard_pass, has_soft_warnings, report)``
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Layer 1 — Hard rules (fail-fast)
# ---------------------------------------------------------------------------

# Why 1 % and not 0 %?  Real-world taxi data contains a small number of
# legitimate anomalies (refunds coded as negative fares, GPS glitches
# producing extreme distances).  Rejecting on a single bad row would make
# the pipeline too brittle.  The 1 % threshold tolerates isolated noise
# while still catching systematic data corruption (e.g. a schema change
# that flips the sign of an entire column).
_OUTLIER_FRACTION_THRESHOLD = 0.01  # reject if >1 % of rows are outliers

# If more than half the target values are missing, the batch is useless
# for supervised learning — reject it outright.
_TARGET_MISSING_THRESHOLD = 0.50    # reject if >50 % of target is NaN


def run_hard_checks(batch_df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Run deterministic hard-rule checks on the raw *batch_df*.

    Hard rules are **fail-fast**: if *any* check fails the batch is rejected
    and the pipeline skips feature engineering, training, and promotion
    entirely.  The checks are intentionally conservative — they only fire
    when the data is structurally broken, not merely noisy.

    The checks run in order:

    1. **Schema** — every column listed in ``config.REQUIRED_COLUMNS`` must
       be present.  A missing column means the upstream data source changed
       its schema.
    2. **Datetime validity** — pickup timestamps must be parseable; dropoff
       must not precede pickup (catches timezone / column-swap bugs).
    3. **Impossible values** — negative distances, negative fares, and
       values exceeding physical upper bounds (``config.MAX_TRIP_DISTANCE``,
       ``config.MAX_FARE_AMOUNT``) are counted.  The check fails only when
       the fraction of bad rows exceeds ``_OUTLIER_FRACTION_THRESHOLD``
       (1 %).
    4. **Target missingness** — if more than 50 % of ``tip_amount`` is NaN,
       the batch cannot support meaningful model training.

    Parameters
    ----------
    batch_df : pd.DataFrame
        Raw batch DataFrame, straight from the parquet file.  Must not be
        feature-engineered (we need the original column names and values).

    Returns
    -------
    passed : bool
        ``True`` when every check passes.
    failures : list[str]
        Human-readable descriptions of each failed check (empty when
        *passed* is ``True``).
    """
    failures: List[str] = []
    n_rows = len(batch_df)

    # -- 1. Required columns present -------------------------------------------
    # This is the schema contract: if the upstream parquet file drops or
    # renames a column, we catch it here before anything else runs.
    present = set(batch_df.columns)
    missing_cols = [c for c in config.REQUIRED_COLUMNS if c not in present]
    if missing_cols:
        failures.append(
            f"Missing required columns: {missing_cols}"
        )

    # -- 2. Invalid datetimes --------------------------------------------------
    # Datetime checks catch two classes of bugs:
    #   (a) Unparseable strings — indicates a format change in the source.
    #   (b) Temporal ordering violations — dropoff before pickup suggests
    #       column swaps or timezone mishandling.
    if config.DATETIME_COL in batch_df.columns:
        # Convert the specified column to datetime; any invalid values are coerced to NaT
        #  (missing datetime) to keep the pipeline from failing
        pickup = pd.to_datetime(batch_df[config.DATETIME_COL], errors="coerce")
        # Count values that were NOT originally NaN but became NaN after
        # coercion — those are the truly unparseable entries.
        unparseable_pickup = int(pickup.isna().sum() - batch_df[config.DATETIME_COL].isna().sum())
        if unparseable_pickup > 0:
            failures.append(
                f"{config.DATETIME_COL}: {unparseable_pickup} rows could not be "
                f"parsed as datetime"
            )

        if config.DROPOFF_DATETIME_COL in batch_df.columns:
            dropoff = pd.to_datetime(
                batch_df[config.DROPOFF_DATETIME_COL], errors="coerce"
            )
            # Only compare rows where both timestamps parsed successfully.
            # dropoff must be >= pickup (strictly after or equal).
            both_valid = pickup.notna() & dropoff.notna()
            bad_order = (dropoff[both_valid] < pickup[both_valid]).sum()
            if bad_order > 0:
                failures.append(
                    f"Datetime order violation: {int(bad_order)} rows have "
                    f"dropoff before pickup"
                )

    # -- 3. Impossible values --------------------------------------------------
    # Each sub-check uses pd.to_numeric(..., errors="coerce") to handle
    # columns that might contain non-numeric strings.  The check fires only
    # when the fraction of bad rows exceeds _OUTLIER_FRACTION_THRESHOLD (1 %).
    if n_rows > 0:
        # 3a. Negative trip_distance — physically impossible; a negative
        #     odometer reading indicates a data-entry or ETL sign error.
        if "trip_distance" in batch_df.columns:
            neg_dist = (
                pd.to_numeric(batch_df["trip_distance"], errors="coerce") < 0
            ).sum()
            if neg_dist / n_rows > _OUTLIER_FRACTION_THRESHOLD:
                failures.append(
                    f"trip_distance: {int(neg_dist)} rows "
                    f"({neg_dist / n_rows:.2%}) have negative values"
                )

        # 3b. Negative fare_amount — a small number of negative fares
        #     (refunds / adjustments) is normal in real TLC data, which is
        #     why we use the 1 % threshold rather than rejecting any negative.
        if "fare_amount" in batch_df.columns:
            neg_fare = (
                pd.to_numeric(batch_df["fare_amount"], errors="coerce") < 0
            ).sum()
            if neg_fare / n_rows > _OUTLIER_FRACTION_THRESHOLD:
                failures.append(
                    f"fare_amount: {int(neg_fare)} rows "
                    f"({neg_fare / n_rows:.2%}) have negative values"
                )

        # 3c. Negative tip_amount — tips should never be negative; a
        #     systematic negative-tip pattern signals a data corruption.
        if "tip_amount" in batch_df.columns:
            neg_tip = (
                pd.to_numeric(batch_df["tip_amount"], errors="coerce") < 0
            ).sum()
            if neg_tip / n_rows > _OUTLIER_FRACTION_THRESHOLD:
                failures.append(
                    f"tip_amount: {int(neg_tip)} rows "
                    f"({neg_tip / n_rows:.2%}) have negative values"
                )

        # 3d. Extreme trip_distance — trips beyond MAX_TRIP_DISTANCE miles
        #     (default 200) are almost certainly GPS errors or data glitches.
        if "trip_distance" in batch_df.columns:
            extreme_dist = (
                pd.to_numeric(batch_df["trip_distance"], errors="coerce")
                > config.MAX_TRIP_DISTANCE
            ).sum()
            if extreme_dist / n_rows > _OUTLIER_FRACTION_THRESHOLD:
                failures.append(
                    f"trip_distance: {int(extreme_dist)} rows "
                    f"({extreme_dist / n_rows:.2%}) exceed "
                    f"{config.MAX_TRIP_DISTANCE} miles"
                )

        # 3e. Extreme fare_amount — fares beyond MAX_FARE_AMOUNT dollars
        #     (default $500) indicate data errors or non-metered flat rates.
        if "fare_amount" in batch_df.columns:
            extreme_fare = (
                pd.to_numeric(batch_df["fare_amount"], errors="coerce")
                > config.MAX_FARE_AMOUNT
            ).sum()
            if extreme_fare / n_rows > _OUTLIER_FRACTION_THRESHOLD:
                failures.append(
                    f"fare_amount: {int(extreme_fare)} rows "
                    f"({extreme_fare / n_rows:.2%}) exceed "
                    f"${config.MAX_FARE_AMOUNT}"
                )

    # -- 4. Target missing (>50 % NaN) ----------------------------------------
    # Without a usable target column, supervised training is impossible.
    # The 50 % threshold is generous — in practice even 10 % NaN in the
    # target would be concerning, but we leave room for partial batches.
    if "tip_amount" in batch_df.columns:
        tip_nan_rate = float(batch_df["tip_amount"].isna().mean())
        if tip_nan_rate > _TARGET_MISSING_THRESHOLD:
            failures.append(
                f"tip_amount: {tip_nan_rate:.1%} NaN values "
                f"(threshold {_TARGET_MISSING_THRESHOLD:.0%})"
            )
    elif "tip_amount" not in batch_df.columns:
        # Already captured by the missing-columns check above, but we add
        # an explicit message as a safety net in case REQUIRED_COLUMNS is
        # ever modified to exclude tip_amount.
        if "Missing required columns" not in " ".join(failures):
            failures.append("tip_amount column is entirely absent")

    passed = len(failures) == 0
    return passed, failures


# ---------------------------------------------------------------------------
# Layer 2 — Soft checks (statistical warnings)
# ---------------------------------------------------------------------------

# Key numeric columns whose distributions are compared between reference
# and batch.  These are the features most likely to drift over time due to
# seasonal patterns, fare policy changes, or data-source issues.
_DRIFT_NUMERIC_COLS = ["trip_distance", "fare_amount", "passenger_count"]

# Taxi-zone ID columns treated as unordered categories.  New zone IDs
# appearing in a batch (but absent from the reference) may indicate a
# geographic expansion or a data-encoding change.
_CATEGORICAL_COLS = ["PULocationID", "DOLocationID"]


def _check_missingness(
    reference_df: pd.DataFrame,
    batch_df: pd.DataFrame,
) -> Dict[str, Dict[str, Any]]:
    """Compare per-column missing-value rates between reference and batch.

    For every column present in *either* DataFrame, compute the NaN rate in
    both the reference and the batch.  A column is **flagged** when the
    batch's NaN rate exceeds the reference's by more than
    ``config.MISSINGNESS_SPIKE_THRESHOLD``.

    Missingness is computed as ``Series.isna().mean()`` — the fraction of
    rows that are NaN.  If a column exists only in the reference, the batch
    rate defaults to 1.0 (entirely missing).  If a column exists only in
    the batch, the reference rate defaults to 0.0 (no baseline missingness).

    Parameters
    ----------
    reference_df : pd.DataFrame
        Training-era reference data (used as the baseline).
    batch_df : pd.DataFrame
        Incoming batch to evaluate.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Keyed by column name.  Each value dict contains:
        - ``ref_rate`` (float): NaN fraction in the reference.
        - ``batch_rate`` (float): NaN fraction in the batch.
        - ``flagged`` (bool): ``True`` when the increase exceeds the
          configured threshold.
    """
    result: Dict[str, Dict[str, Any]] = {}
    # Union of columns from both DataFrames — we want to detect columns
    # that disappeared from the batch as well as new columns.
    all_cols = sorted(
        set(reference_df.columns).union(set(batch_df.columns))
    )
    for col in all_cols:
        # Default ref_rate=0 if column is new (not in reference);
        # default batch_rate=1 if column is missing from batch entirely.
        ref_rate = float(reference_df[col].isna().mean()) if col in reference_df.columns else 0.0
        batch_rate = float(batch_df[col].isna().mean()) if col in batch_df.columns else 1.0
        # Flag only when the *increase* (not absolute level) exceeds the
        # threshold — a column that is always 30 % NaN is fine as long as
        # it doesn't jump to 50 %.
        increase = batch_rate - ref_rate
        flagged = increase > config.MISSINGNESS_SPIKE_THRESHOLD
        result[col] = {
            "ref_rate": round(ref_rate, 6),
            "batch_rate": round(batch_rate, 6),
            "flagged": flagged,
        }
    return result


def _check_unseen_categories(
    reference_df: pd.DataFrame,
    batch_df: pd.DataFrame,
) -> Dict[str, Dict[str, Any]]:
    """Detect categorical values in the batch that were never seen in the reference.

    For each column in ``_CATEGORICAL_COLS``, build the set of unique
    non-NaN values from the reference.  Then count how many non-NaN batch
    values fall outside that set.  The column is **flagged** when the
    fraction of unseen values exceeds ``config.UNSEEN_CATEGORICAL_THRESHOLD``.

    This catches scenarios like:
    - New taxi zones added after the reference period.
    - Encoding changes (e.g. zone IDs shifted by an offset).
    - Data from a different city accidentally mixed in.

    Parameters
    ----------
    reference_df : pd.DataFrame
        Training-era reference data.
    batch_df : pd.DataFrame
        Incoming batch to evaluate.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Keyed by column name.  Each value dict contains:
        - ``unseen_count`` (int): number of batch rows with unseen values.
        - ``unseen_pct`` (float): fraction of batch rows with unseen values.
        - ``flagged`` (bool): ``True`` when ``unseen_pct`` exceeds the
          configured threshold.
    """
    result: Dict[str, Dict[str, Any]] = {}
    for col in _CATEGORICAL_COLS:
        if col not in reference_df.columns or col not in batch_df.columns:
            continue
        # Build the "known vocabulary" from the reference data.
        ref_values = set(reference_df[col].dropna().unique())
        batch_values = batch_df[col].dropna()
        if len(batch_values) == 0:
            # No non-NaN values in the batch — nothing to compare.
            result[col] = {"unseen_count": 0, "unseen_pct": 0.0, "flagged": False}
            continue
        # Boolean mask: True for batch values NOT in the reference vocabulary.
        unseen_mask = ~batch_values.isin(ref_values)
        unseen_count = int(unseen_mask.sum())
        unseen_pct = float(unseen_count / len(batch_values))
        flagged = unseen_pct > config.UNSEEN_CATEGORICAL_THRESHOLD
        result[col] = {
            "unseen_count": unseen_count,
            "unseen_pct": round(unseen_pct, 6),
            "flagged": flagged,
        }
    return result


def _check_drift_nannyml(
    reference_df: pd.DataFrame,
    batch_df: pd.DataFrame,
) -> Tuple[bool, Dict[str, Dict[str, Any]]]:
    """Attempt NannyML-based univariate drift detection.

    NannyML's ``UnivariateDriftCalculator`` computes a drift statistic for
    each numeric column by comparing the distribution of the analysis
    (batch) data against the reference data.  Internally it uses the
    Jensen–Shannon distance by default and determines an alert threshold
    from the reference data's own variability.

    **How it works here:**

    1. We ``fit()`` the calculator on the reference data — this learns the
       baseline distribution and calibrates the alert threshold.
    2. We ``calculate()`` on the batch data — this computes the drift
       statistic for a single chunk (the entire batch).
    3. For each column we extract the ``value`` (drift statistic) and
       ``alert`` (boolean: did the statistic exceed the learned threshold?).

    The ``chunk_size`` is set to ``len(batch)`` so the entire batch is
    treated as one chunk — we want a single drift verdict per column, not
    a time-series of chunk-level results.

    Parameters
    ----------
    reference_df : pd.DataFrame
        Training-era reference data.
    batch_df : pd.DataFrame
        Incoming batch to evaluate.

    Returns
    -------
    tuple[bool, Dict[str, Dict[str, Any]]]
        ``(success, drift_dict)`` where *success* is ``False`` when
        NannyML is not installed or raises any error (triggering the
        scipy fallback).  *drift_dict* is keyed by column name with:
        - ``statistic`` (float | None): the drift statistic value.
        - ``p_value`` (None): NannyML uses statistic-based alerting, not
          p-values, so this is always ``None``.
        - ``drifted`` (bool): ``True`` when the alert fired.
    """
    try:
        # Lazy import — NannyML is an optional dependency.  If it's not
        # installed, the ImportError is caught and we return (False, {}).
        import nannyml  # type: ignore[import-untyped]

        # Only check columns that exist in both DataFrames.
        cols = [c for c in _DRIFT_NUMERIC_COLS if c in reference_df.columns and c in batch_df.columns]
        if not cols:
            return True, {}

        # NannyML's calculator expects plain DataFrames without extra
        # metadata columns.  The _partition column is added here only for
        # clarity during debugging — it is dropped before fit/calculate.
        ref = reference_df[cols].copy()
        ref["_partition"] = "reference"
        batch = batch_df[cols].copy()
        batch["_partition"] = "analysis"

        # UnivariateDriftCalculator: fits on reference to learn baseline
        # distributions, then calculates drift statistics on analysis data.
        # treat_as_categorical=[] ensures all columns are treated as
        # continuous (they are numeric features).
        # chunk_size=len(batch) treats the entire batch as a single chunk.
        calc = nannyml.UnivariateDriftCalculator(
            column_names=cols,
            treat_as_categorical=[],
            chunk_size=len(batch),
        )
        calc.fit(ref.drop(columns=["_partition"]))
        results = calc.calculate(batch.drop(columns=["_partition"]))

        drift_dict: Dict[str, Dict[str, Any]] = {}
        for col in cols:
            # Filter results to the current column and extract the single
            # chunk's drift statistic and alert flag.
            col_results = results.filter(column_names=[col])
            stat_val = float("nan")
            drifted = False
            try:
                # "value" = the drift statistic (e.g. Jensen–Shannon distance).
                # "alert" = True if the statistic exceeds the learned threshold.
                stat_val = float(col_results.to_df().iloc[0]["value"])
                drifted = bool(col_results.to_df().iloc[0]["alert"])
            except (IndexError, KeyError):
                # Gracefully handle unexpected result structure.
                pass
            drift_dict[col] = {
                "statistic": round(stat_val, 6) if np.isfinite(stat_val) else None,
                # NannyML does not produce p-values; it uses statistic-based
                # alerting with thresholds learned from the reference data.
                "p_value": None,
                "drifted": drifted,
            }
        return True, drift_dict

    except Exception as exc:  # noqa: BLE001
        # Catch *any* exception (ImportError, RuntimeError, etc.) so the
        # pipeline never crashes due to NannyML issues.
        logger.debug("NannyML drift check unavailable: %s", exc)
        return False, {}


def _check_drift_scipy(
    reference_df: pd.DataFrame,
    batch_df: pd.DataFrame,
) -> Dict[str, Dict[str, Any]]:
    """Fallback drift detection using the two-sample Kolmogorov–Smirnov test.

    The KS test compares the empirical CDFs of two samples and returns a
    test statistic (maximum absolute difference between the CDFs) and a
    p-value.  A small p-value (< 0.05) indicates that the two samples are
    unlikely to have been drawn from the same distribution.

    This is a simpler alternative to NannyML — it does not learn adaptive
    thresholds, but it is always available via scipy and provides a
    reasonable drift signal for numeric columns.

    **Guard rail:** if either sample has fewer than 20 non-NaN values, the
    KS test is unreliable, so we skip the column and report no drift.

    Parameters
    ----------
    reference_df : pd.DataFrame
        Training-era reference data.
    batch_df : pd.DataFrame
        Incoming batch to evaluate.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Keyed by column name.  Each value dict contains:
        - ``statistic`` (float | None): KS test statistic (D).
        - ``p_value`` (float | None): two-sided p-value.
        - ``drifted`` (bool): ``True`` when ``p_value < 0.05``.
    """
    # Lazy import — scipy is always in the conda env but we keep the
    # import local to avoid loading it when NannyML succeeds.
    from scipy.stats import ks_2samp

    drift_dict: Dict[str, Dict[str, Any]] = {}
    for col in _DRIFT_NUMERIC_COLS:
        if col not in reference_df.columns or col not in batch_df.columns:
            continue
        # Coerce to numeric and drop NaNs — the KS test requires clean
        # numeric arrays.
        ref_vals = pd.to_numeric(reference_df[col], errors="coerce").dropna().to_numpy()
        batch_vals = pd.to_numeric(batch_df[col], errors="coerce").dropna().to_numpy()
        # Guard: KS test is unreliable with very small samples.
        if len(ref_vals) < 20 or len(batch_vals) < 20:
            drift_dict[col] = {
                "statistic": None,
                "p_value": None,
                "drifted": False,
            }
            continue
        # ks_2samp returns (statistic, p_value).  The statistic is the
        # maximum absolute difference between the two empirical CDFs.
        stat, p_value = ks_2samp(ref_vals, batch_vals)
        drift_dict[col] = {
            "statistic": round(float(stat), 6),
            "p_value": round(float(p_value), 6),
            # Standard significance level α = 0.05.
            "drifted": p_value < 0.05,
        }
    return drift_dict


def _build_summary(
    missingness: Dict[str, Dict[str, Any]],
    unseen: Dict[str, Dict[str, Any]],
    drift: Dict[str, Dict[str, Any]],
) -> str:
    """Produce a one-paragraph human-readable summary of soft-check results.

    Aggregates flagged columns from all three soft-check categories
    (missingness, unseen categories, drift) into a single summary string.
    If nothing is flagged, returns a reassuring "all passed" message.

    This summary is stored in the integrity report and logged to the
    console, making it easy for operators to see at a glance what (if
    anything) triggered a warning.

    Parameters
    ----------
    missingness : Dict[str, Dict[str, Any]]
        Output of ``_check_missingness()``.
    unseen : Dict[str, Dict[str, Any]]
        Output of ``_check_unseen_categories()``.
    drift : Dict[str, Dict[str, Any]]
        Output of ``_check_drift_nannyml()`` or ``_check_drift_scipy()``.

    Returns
    -------
    str
        Human-readable summary sentence.
    """
    parts: List[str] = []

    flagged_miss = [c for c, v in missingness.items() if v["flagged"]]
    if flagged_miss:
        parts.append(
            f"Missingness spike in {len(flagged_miss)} column(s): "
            f"{', '.join(flagged_miss)}"
        )

    flagged_unseen = [c for c, v in unseen.items() if v["flagged"]]
    if flagged_unseen:
        parts.append(
            f"Unseen categories in {len(flagged_unseen)} column(s): "
            f"{', '.join(flagged_unseen)}"
        )

    drifted_cols = [c for c, v in drift.items() if v.get("drifted")]
    if drifted_cols:
        parts.append(
            f"Distribution drift detected in {len(drifted_cols)} column(s): "
            f"{', '.join(drifted_cols)}"
        )

    if not parts:
        return "All soft checks passed — no warnings."
    return "; ".join(parts) + "."


def run_soft_checks(
    reference_df: pd.DataFrame,
    batch_df: pd.DataFrame,
) -> Tuple[bool, Dict[str, Any]]:
    """Run statistical soft checks comparing *batch_df* against *reference_df*.

    Soft checks are **warn-only** — they never block the pipeline.  Their
    purpose is to surface data-quality signals that downstream decision
    functions (``should_retrain``, ``should_promote``) can use to adjust
    their thresholds.

    Three categories of checks are performed:

    1. **Missingness** — per-column NaN-rate comparison (see
       ``_check_missingness``).
    2. **Unseen categories** — categorical values in the batch that were
       absent from the reference (see ``_check_unseen_categories``).
    3. **Distribution drift** — univariate drift on key numeric columns,
       using NannyML if available, otherwise scipy KS test (see
       ``_check_drift_nannyml`` and ``_check_drift_scipy``).

    Parameters
    ----------
    reference_df : pd.DataFrame
        Training-era reference data (the baseline for comparison).
    batch_df : pd.DataFrame
        Incoming batch to evaluate.

    Returns
    -------
    has_warnings : bool
        ``True`` when at least one soft check is flagged.
    report : dict
        Structured report with keys ``missingness``, ``unseen_categories``,
        ``drift``, and ``summary``.
    """
    missingness = _check_missingness(reference_df, batch_df)
    unseen = _check_unseen_categories(reference_df, batch_df)

    # Graceful degradation: try NannyML first (production-grade drift
    # detection with adaptive thresholds); fall back to scipy KS test
    # if NannyML is unavailable or errors out.
    nannyml_ok, drift = _check_drift_nannyml(reference_df, batch_df)
    if not nannyml_ok:
        logger.info("Falling back to scipy KS test for drift detection")
        drift = _check_drift_scipy(reference_df, batch_df)

    summary = _build_summary(missingness, unseen, drift)

    # Aggregate: has_warnings is True if ANY sub-check flagged anything.
    # This single boolean is what flow.py passes to should_retrain() and
    # should_promote() as the integrity_warn parameter.
    has_warnings = (
        any(v["flagged"] for v in missingness.values())
        or any(v["flagged"] for v in unseen.values())
        or any(v.get("drifted", False) for v in drift.values())
    )

    report: Dict[str, Any] = {
        "missingness": missingness,
        "unseen_categories": unseen,
        "drift": drift,
        "summary": summary,
    }
    return has_warnings, report


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_integrity_checks(
    reference_df: pd.DataFrame,
    batch_df: pd.DataFrame,
) -> Tuple[bool, bool, Dict[str, Any]]:
    """Execute the full two-layer integrity gate (Step B).

    This is the main entry point called by ``flow.py``.  It orchestrates
    both layers sequentially:

    1. Run hard checks on the batch (Layer 1).
    2. **Only if hard checks pass**, run soft checks comparing the batch
       against the reference (Layer 2).  If hard checks fail, soft checks
       are skipped entirely — there is no value in computing drift
       statistics on structurally broken data.

    The ``overall_status`` field in the returned report encodes the
    three-way outcome:

    - ``"pass"``   — all hard rules pass AND no soft warnings.
    - ``"warn"``   — all hard rules pass BUT soft checks flagged warnings.
      The pipeline continues, but ``should_retrain()`` and
      ``should_promote()`` will tighten their thresholds.
    - ``"reject"`` — one or more hard rules failed.  The pipeline
      short-circuits and skips training / promotion entirely.

    Parameters
    ----------
    reference_df : pd.DataFrame
        Training-era reference data used as the baseline for soft checks.
    batch_df : pd.DataFrame
        Raw incoming batch DataFrame (before feature engineering).

    Returns
    -------
    hard_pass : bool
        ``True`` when all hard rules pass.
    has_soft_warnings : bool
        ``True`` when at least one soft check fires (only meaningful when
        *hard_pass* is ``True``).
    report : dict
        Combined report with keys ``hard_check_passed``, ``hard_failures``,
        ``soft_warnings``, ``soft_report``, and ``overall_status``.
    """
    hard_pass, hard_failures = run_hard_checks(batch_df)

    if hard_pass:
        # Hard checks passed — proceed to statistical soft checks.
        has_soft_warnings, soft_report = run_soft_checks(reference_df, batch_df)
    else:
        # Hard checks failed — skip soft checks (data is too broken).
        has_soft_warnings = False
        soft_report = None

    # overall_status encodes the three-way outcome used by flow.py to
    # decide whether to continue, retrain with caution, or abort.
    if not hard_pass:
        overall_status = "reject"
    elif has_soft_warnings:
        overall_status = "warn"
    else:
        overall_status = "pass"

    report: Dict[str, Any] = {
        "hard_check_passed": hard_pass,
        "hard_failures": hard_failures,
        "soft_warnings": has_soft_warnings,
        "soft_report": soft_report,
        "overall_status": overall_status,
    }

    logger.info("Integrity gate: %s", overall_status)
    if hard_failures:
        for f in hard_failures:
            logger.warning("  HARD FAIL: %s", f)
    if soft_report and has_soft_warnings:
        logger.warning("  SOFT: %s", soft_report.get("summary", ""))

    return hard_pass, has_soft_warnings, report
