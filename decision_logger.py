"""
Module: decision_logger.py

Purpose:
This module provides a standardized way to log decisions made throughout the MLOps pipeline.
Structured decision logging is crucial for auditability, transparency, and debugging of automated ML workflows.
Each significant gate (e.g., data integrity, model evaluation, retraining, model promotion) in the pipeline
produces a ``decision.json`` artifact. These artifacts serve as an immutable record of why a particular
action was taken (or not taken) at a specific stage.

How ``decision.json`` artifacts are used:
- **Auditability**: Provides a clear, timestamped record of all decisions, their criteria, and evidence.
- **Transparency**: Explains the rationale behind automated actions, making the pipeline's behavior understandable.
- **Debugging**: Helps in tracing issues by reviewing the decisions made at each stage of a pipeline run.
- **Reproducibility**: Contributes to the overall reproducibility of pipeline runs by documenting decision points.

Role of MLflow tags:
In addition to logging the full ``decision.json`` artifact, key aspects of each decision (``decision_stage``
and ``decision_action``) are logged as MLflow tags. This allows for quick filtering and overview of pipeline
runs directly within the MLflow UI without needing to download and inspect the full JSON artifact.

Functions:
- `make_decision()`: The core builder function for creating a standardized decision dictionary.
- `log_decision_to_mlflow()`: Persists a single decision as an MLflow artifact and sets relevant MLflow tags.
- `make_integrity_decision()`: Convenience wrapper for creating decisions related to the data integrity gate.
- `make_retrain_decision()`: Convenience wrapper for creating decisions related to the model evaluation/retraining gate.
- `make_promotion_decision()`: Convenience wrapper for creating decisions related to the model promotion gate.
- `log_all_decisions()`: Persists a list of multiple decisions as a single combined MLflow artifact.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone

import mlflow

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Core decision builder
# ---------------------------------------------------------------------------


def make_decision(
    stage: str,
    action: str,
    criteria: dict,
    evidence: dict,
    reason: str,
) -> dict:
    """Creates a standardized decision record, capturing the outcome of a pipeline gate.

    This function is the core builder for all decision artifacts. It ensures that every
    decision made within the MLOps pipeline adheres to a consistent schema, which is vital
    for auditability, transparency, and debugging.

    Args:
        stage (str):
            The name of the pipeline gate or stage where the decision was made.
            Examples: ``"integrity_gate"``, ``"model_gate"``, ``"retrain"``,
            ``"promotion"``.
        action (str):
            The specific outcome or action taken as a result of the decision.
            Examples: ``"reject_batch"``, ``"pass"``, ``"retrain"``,
            ``"promote"``, ``"skip_retrain"``, ``"reject_candidate"``.
        criteria (dict):
            A dictionary detailing the rules, thresholds, or conditions that were evaluated
            to arrive at the decision. This provides context on *what* was checked.
        evidence (dict):
            A dictionary containing the actual observed values, metrics, or data points
            that were used as evidence against the defined criteria. This shows *what was found*.
        reason (str):
            A human-readable explanation or summary of *why* the particular action was taken.
            This is crucial for understanding the decision without deep-diving into raw data.

    Returns:
        dict:
            A complete decision record dictionary with the following structure:
            - ``timestamp`` (str): UTC timestamp of when the decision was made (ISO 8601 format).
            - ``stage`` (str): The pipeline stage where the decision occurred.
            - ``action`` (str): The action taken based on the decision.
            - ``criteria`` (dict): The criteria used for evaluation.
            - ``evidence`` (dict): The evidence observed during evaluation.
            - ``reason`` (str): The human-readable reason for the decision.
    """
    # Capture the current UTC timestamp for the decision, ensuring consistency.
    decision = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "action": action,
        "criteria": criteria,
        "evidence": evidence,
        "reason": reason,
    }
    # Log the decision for immediate visibility in console/logs.
    logger.info("Decision [%s] → %s: %s", stage, action, reason)
    return decision


# ---------------------------------------------------------------------------
# 2. Log a single decision to MLflow
# ---------------------------------------------------------------------------


def log_decision_to_mlflow(
    decision: dict,
    artifact_name: str = "decision.json",
) -> None:
    """Writes a decision record to a temporary JSON file and logs it as an MLflow artifact.

    This function ensures that each individual decision made during a pipeline run is
    persisted within the MLflow tracking server, making it accessible for review and audit.
    To facilitate quick overview and filtering in the MLflow UI, two key fields from the
    decision record (``stage`` and ``action``) are also logged as MLflow tags.

    The process involves:
    1. Creating a temporary directory and file to store the JSON representation of the decision.
    2. Writing the decision dictionary to this temporary file.
    3. Logging the temporary file as an MLflow artifact using ``mlflow.log_artifact()``.
    4. Setting MLflow run tags (``decision_stage`` and ``decision_action``) for easy searchability.
    5. Ensuring the temporary file and directory are cleaned up regardless of success or failure.

    Args:
        decision (dict):
            A decision record dictionary, typically produced by :func:`make_decision`
            or one of its convenience wrappers.
        artifact_name (str, optional):
            The filename to use when logging the artifact to MLflow.
            Defaults to ``"decision.json"``.
    """
    # Create a temporary directory and file to store the decision JSON.
    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, artifact_name)
    try:
        # Write the decision dictionary to the temporary JSON file.
        with open(tmp_path, "w") as fh:
            json.dump(decision, fh, indent=2, default=str)

        # Log the temporary JSON file as an MLflow artifact.
        mlflow.log_artifact(tmp_path)
        # Set MLflow tags for quick filtering and visibility in the UI.
        mlflow.set_tag("decision_stage", decision["stage"])
        mlflow.set_tag("decision_action", decision["action"])
        logger.info("Logged decision artifact '%s' to MLflow.", artifact_name)
    finally:
        # Ensure cleanup of the temporary file and directory.
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        if os.path.isdir(tmp_dir):
            os.rmdir(tmp_dir)


# ---------------------------------------------------------------------------
# 3. Integrity-gate convenience function
# ---------------------------------------------------------------------------


def make_integrity_decision(
    hard_pass: bool,
    hard_failures: list,
    soft_warnings: bool,
    soft_report: dict,
) -> dict:
    """Builds a decision record specifically for the data integrity gate.

    This function encapsulates the logic for determining the action and reason
    based on the outcomes of hard and soft integrity checks. It then maps these
    outcomes to the generic :func:`make_decision` function to create a standardized
    decision record.

    Args:
        hard_pass (bool):
            ``True`` if the incoming data batch successfully passed all critical
            (hard) integrity checks. If ``False``, the batch is rejected.
        hard_failures (list):
            A list of strings, each describing a specific failure encountered during
            the hard integrity checks. This list will be empty if ``hard_pass`` is ``True``.
        soft_warnings (bool):
            ``True`` if any non-critical (soft) warnings were raised, typically from
            data drift detection tools like NannyML. If ``True``, the batch still passes
            but with a warning.
        soft_report (dict):
            A summary dictionary containing details from the soft integrity checks,
            such as drift detection reports. This provides evidence for soft warnings.

    Returns:
        dict:
            A complete decision record dictionary, formatted by :func:`make_decision`,
            ready for logging with :func:`log_decision_to_mlflow`.
            The ``stage`` will be ``"integrity_gate"``.
    """
    # Determine the action and reason based on integrity check results.
    if not hard_pass:
        # If hard checks fail, the batch is rejected.
        action = "reject_batch"
        reason = (
            f"Hard integrity checks failed: {hard_failures}"
        )
    elif soft_warnings:
        # If hard checks pass but soft warnings exist, the batch passes with caution.
        action = "pass"
        reason = (
            "Batch passed hard checks but has soft warnings — "
            f"proceeding with caution. Warnings: {soft_report}"
        )
    else:
        # If all checks pass without warnings, the batch passes cleanly.
        action = "pass"
        reason = "Batch passed all integrity checks with no warnings."

    # Define the criteria that were used for the integrity checks.
    # These are typically configured thresholds from the `config` module.
    criteria = {
        "max_trip_distance": config.MAX_TRIP_DISTANCE,
        "max_fare_amount": config.MAX_FARE_AMOUNT,
        "max_tip_amount": config.MAX_TIP_AMOUNT,
        "min_trip_distance": config.MIN_TRIP_DISTANCE,
        "max_passenger_count": config.MAX_PASSENGER_COUNT,
        "max_duration_min": config.MAX_DURATION_MIN,
        "missingness_spike_threshold": config.MISSINGNESS_SPIKE_THRESHOLD,
        "unseen_categorical_threshold": config.UNSEEN_CATEGORICAL_THRESHOLD,
    }

    # Compile the evidence from the integrity checks.
    evidence = {
        "hard_pass": hard_pass,
        "hard_failures": hard_failures,
        "soft_warnings": soft_warnings,
        "soft_report_summary": soft_report,
    }

    # Call the generic make_decision function to create the final decision record.
    return make_decision(
        stage="integrity_gate",
        action=action,
        criteria=criteria,
        evidence=evidence,
        reason=reason,
    )


# ---------------------------------------------------------------------------
# 4. Retrain-gate convenience function
# ---------------------------------------------------------------------------


def make_retrain_decision(
    retrain_needed: bool,
    rmse_champion_on_batch: float,
    rmse_champion_on_ref: float,
    rmse_increase_pct: float,
    reason: str,
    rmse_baseline: float | None = None,
) -> dict:
    """Builds a decision record for the model evaluation and potential retraining gate.

    This function determines whether a model retraining is necessary based on the
    performance of the champion model on the new batch compared to its performance
    on the reference data.  It then uses :func:`make_decision` to create a
    standardized record of this evaluation.

    Args:
        retrain_needed (bool):
            ``True`` if the champion model's performance (RMSE) has degraded beyond
            the acceptable threshold, indicating a need for retraining. ``False`` otherwise.
        rmse_champion_on_batch (float):
            The RMSE of the current champion model when evaluated on the latest
            incoming data batch.
        rmse_champion_on_ref (float):
            The RMSE of the current champion model when evaluated on the reference
            dataset.  This is the performance baseline for degradation detection.
        rmse_increase_pct (float):
            The relative increase in the champion model's RMSE from reference to
            batch.  This metric is a key indicator for model degradation.
        reason (str):
            A human-readable explanation for the decision, detailing why retraining is
            or is not recommended.
        rmse_baseline (float | None):
            Optional RMSE of a naive mean-prediction baseline on the batch data.
            Logged for informational purposes only.

    Returns:
        dict:
            A complete decision record dictionary, formatted by :func:`make_decision`,
            ready for logging with :func:`log_decision_to_mlflow`.
            The ``stage`` will be ``"model_gate"``.
    """
    # Build evidence dict with the new comparison metrics.
    evidence = {
        "rmse_champion_on_batch": rmse_champion_on_batch,
        "rmse_champion_on_ref": rmse_champion_on_ref,
        "rmse_increase_pct": rmse_increase_pct,
    }
    if rmse_baseline is not None:
        evidence["rmse_baseline"] = rmse_baseline

    # Call the generic make_decision function to create the final decision record.
    # The action is either "retrain" or "skip_retrain" based on `retrain_needed`.
    return make_decision(
        stage="model_gate",
        action="retrain" if retrain_needed else "skip_retrain",
        criteria={
            # The key criterion for this decision is the RMSE degradation threshold.
            "rmse_degradation_threshold": config.RMSE_DEGRADATION_THRESHOLD,
        },
        evidence=evidence,
        reason=reason,
    )


# ---------------------------------------------------------------------------
# 5. Promotion-gate convenience function
# ---------------------------------------------------------------------------


def make_promotion_decision(
    promote: bool,
    rmse_candidate: float,
    rmse_champion: float,
    rmse_candidate_on_ref: float,
    rmse_champion_on_ref: float,
    reason: str,
) -> dict:
    """Builds a decision record for the model promotion gate.

    This function evaluates whether a candidate model should replace the current champion
    model based on their performance metrics (RMSE) on both the current batch and a
    reference dataset. It then constructs a standardized decision record using
    :func:`make_decision`.

    Args:
        promote (bool):
            ``True`` if the candidate model's performance warrants its promotion to champion.
            ``False`` if the candidate should be rejected.
        rmse_candidate (float):
            The Root Mean Squared Error (RMSE) of the candidate model when evaluated
            on the current incoming data batch.
        rmse_champion (float):
            The RMSE of the current champion model when evaluated on the current
            incoming data batch.
        rmse_candidate_on_ref (float):
            The RMSE of the candidate model when evaluated on a historical reference dataset.
            This helps assess the candidate's stability and generalization.
        rmse_champion_on_ref (float):
            The RMSE of the current champion model when evaluated on the historical
            reference dataset. Used for comparison against the candidate on known data.
        reason (str):
            A human-readable explanation for the promotion decision, detailing why the
            candidate was promoted or rejected.

    Returns:
        dict:
            A complete decision record dictionary, formatted by :func:`make_decision`,
            ready for logging with :func:`log_decision_to_mlflow`.
            The ``stage`` will be ``"promotion"``.
    """
    # Calculate the percentage improvement of the candidate over the champion on the current batch.
    if rmse_champion > 0:
        improvement_pct = (rmse_champion - rmse_candidate) / rmse_champion
    else:
        improvement_pct = 0.0

    # Calculate the percentage regression of the candidate compared to the champion on the reference dataset.
    # This helps detect if the candidate performs significantly worse on historical data.
    if rmse_champion_on_ref > 0:
        ref_regression_pct = (
            (rmse_candidate_on_ref - rmse_champion_on_ref) / rmse_champion_on_ref
        )
    else:
        ref_regression_pct = 0.0

    # Call the generic make_decision function to create the final decision record.
    # The action is either "promote" or "reject_candidate" based on `promote`.
    return make_decision(
        stage="promotion",
        action="promote" if promote else "reject_candidate",
        criteria={
            # Key criteria for promotion include minimum improvement and tolerance for reference regression.
            "min_improvement": config.MIN_IMPROVEMENT,
            "reference_regression_tolerance": config.REFERENCE_REGRESSION_TOLERANCE,
        },
        evidence={
            # Evidence includes all RMSE values and the calculated percentage changes.
            "rmse_candidate": rmse_candidate,
            "rmse_champion": rmse_champion,
            "rmse_candidate_on_ref": rmse_candidate_on_ref,
            "rmse_champion_on_ref": rmse_champion_on_ref,
            "improvement_pct": round(improvement_pct, 6),
            "reference_regression_pct": round(ref_regression_pct, 6),
        },
        reason=reason,
    )


# ---------------------------------------------------------------------------
# 6. Log all decisions as a combined artifact
# ---------------------------------------------------------------------------


def log_all_decisions(
    decisions: list[dict],
    artifact_name: str = "all_decisions.json",
) -> None:
    """Logs a list of decision records as a single combined MLflow artifact.

    This function is particularly useful for creating a comprehensive audit trail
    of all decisions made throughout an entire pipeline run. By combining multiple
    individual decision records into one artifact, it provides a holistic view
    of the pipeline's behavior and outcomes in a single, easily accessible file.

    Args:
        decisions (list[dict]):
            A list of decision record dictionaries. Each dictionary should ideally
            be produced by :func:`make_decision` or one of its specialized wrappers
            (e.g., :func:`make_integrity_decision`).
        artifact_name (str, optional):
            The filename to use when logging the combined artifact to MLflow.
            Defaults to ``"all_decisions.json"``.
    """
    tmp_path: str | None = None
    try:
        # Create a temporary directory to store the combined JSON artifact.
        tmp_dir = tempfile.mkdtemp()
        tmp_path = os.path.join(tmp_dir, artifact_name)
        # Write the list of decisions to the temporary JSON file.
        with open(tmp_path, "w") as fh:
            json.dump(decisions, fh, indent=2, default=str)

        # Log the combined JSON file as an MLflow artifact.
        mlflow.log_artifact(tmp_path)
        logger.info(
            "Logged %d decisions as combined artifact '%s'.",
            len(decisions),
            artifact_name,
        )
    finally:
        # Ensure cleanup of the temporary file and directory.
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
