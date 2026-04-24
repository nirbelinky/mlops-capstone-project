"""
Module: flow.py

This module defines the main Metaflow orchestrator for the MLOps capstone project,
``GreenTaxiTipFlow``. It implements a complete monitoring and retraining pipeline
for a machine learning model that predicts green taxi tips.

Overall Workflow:
1.  **Start**: Initializes MLflow and sets up flow-specific artifacts.
2.  **Load Data**: Fetches the reference dataset and the new batch of data.
3.  **Integrity Gate**: Performs data integrity checks on the new batch. If critical issues are found,
    the pipeline can short-circuit here.
4.  **Feature Engineering**: Applies necessary feature transformations to both reference and batch data.
5.  **Load Champion**: Loads the current champion model from the MLflow Model Registry. If no champion
    exists (first run), a bootstrap model is trained and registered.
6.  **Evaluate Champion**: Evaluates the champion model's performance on the new batch and decides
    whether retraining is needed based on performance degradation.
7.  **Retrain**: If retraining is recommended, a new candidate model is trained on the combined
    reference and new batch data.
8.  **Candidate Gate**: Evaluates the newly trained candidate model against the champion and decides
    whether to promote the candidate to become the new champion.
9.  **End**: Logs all decisions made throughout the pipeline and provides a summary of the run.

Role of Metaflow:
Metaflow provides the framework for building and orchestrating this machine learning pipeline.
-   **Steps**: Each stage of the pipeline is defined as a Metaflow step, ensuring modularity and reusability.
-   **Artifacts**: Data and state are passed between steps using Metaflow artifacts (e.g., ``self.ref_raw``,
    ``self.batch_eng``), ensuring data lineage and reproducibility.
-   **Resume**: Metaflow's ability to resume flows from any step aids in development and debugging.

Role of MLflow:
MLflow is used for experiment tracking, model management, and artifact logging.
-   **Tracking**: Logs parameters, metrics, and artifacts for each pipeline run, providing a comprehensive
    history of experiments.
-   **Registry**: Manages model versions and aliases (e.g., ``@champion``), facilitating model deployment
    and lifecycle management.

Linearized Flow Structure:
The flow is designed with a linear topology (each step has exactly one ``self.next()`` call) to satisfy
Metaflow's structural requirements. Conditional logic (e.g., skipping retraining if not needed, rejecting
a batch due to integrity issues) is handled within individual steps by checking status flags (e.g.,
``self.hard_pass``, ``self.retrain_needed``) and conditionally executing logic or setting artifacts
for downstream steps.

Decision Logging:
Every critical decision point in the pipeline (integrity gate, evaluation, promotion) produces a structured
decision record using the :mod:`decision_logger` module. These records are logged as MLflow artifacts,
providing a complete and auditable trail of the pipeline's automated judgments.
"""

from __future__ import annotations

import sys
import os
import logging
import tempfile
from pathlib import Path

# Ensure sibling modules are importable regardless of working directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import mlflow
from mlflow import MlflowClient
from metaflow import FlowSpec, Parameter, step, current
from sklearn.model_selection import cross_val_score

import config
import feature_engineering
import integrity_checks
import model_utils
import decision_logger

logging.basicConfig(
    level=logging.INFO,
    format="%(name)-25s  %(levelname)-8s  %(message)s",
)

# Suppress noisy MLflow / Alembic initialisation messages that repeat in every
# Metaflow step (each step is a separate subprocess that re-initialises MLflow).
model_utils.suppress_mlflow_logs()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _log_step_banner(message: str) -> None:
    """Print a decorated step banner to the log.

    Produces output like::

        ============================================================
        STEP 1/8: LOAD DATA — Reading reference and batch parquet files
        ============================================================

    Args:
        message: The text to display between the separator lines.
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info(message)
    logger.info("=" * 60)


def _ensure_mlflow(run_id: str | None = None):
    """Ensures MLflow tracking is correctly configured and optionally resumes a run.

    This helper function is called at the beginning of each Metaflow step to
    (re)initialize the MLflow client and ensure that subsequent MLflow logging
    operations are associated with the correct experiment and run.

    Args:
        run_id (str | None, optional):
            If provided, MLflow will attempt to resume the specified run.
            If ``None``, a new run will be started implicitly by subsequent
            logging calls, or an existing active run will be used.
            Defaults to ``None``.
    """
    # Initialize MLflow tracking URI and experiment name from config.
    # This ensures all runs are logged to the correct location.
    model_utils.init_mlflow()
    # If a run_id is provided, resume that specific MLflow run.
    # This is crucial for Metaflow steps to continue logging to the same run.
    if run_id is not None:
        mlflow.start_run(run_id=run_id)


class GreenTaxiTipFlow(FlowSpec):
    """Metaflow orchestrator for the Green Taxi Tip Prediction MLOps pipeline.

    This flow defines the complete end-to-end process for monitoring a deployed
    machine learning model, performing data integrity checks, evaluating model
    performance on new data, retraining a candidate model if necessary, and
    managing model promotion to production via the MLflow Model Registry.

    The flow is structured as a series of Metaflow steps, with data and state
    passed between steps as Metaflow artifacts (``self.<artifact_name>``).
    Each significant decision point is logged using the :mod:`decision_logger`
    module to ensure auditability and transparency.
    """

    # ── Parameters ────────────────────────────────────────────────────
    reference_path = Parameter(
        "reference-path",
        help="Path to the reference dataset in Parquet format. This dataset represents the historical data on which the champion model was initially trained or last validated. It is used for data integrity checks, feature engineering, and evaluating candidate models for stability.",
    )
    batch_path = Parameter(
        "batch-path",
        help="Path to the new incoming data batch in Parquet format. This is the data that the pipeline will process, check for integrity, and use to evaluate the champion model's performance.",
    )
    model_name = Parameter(
        "model-name",
        default="green_taxi_tip_model",
        help="The name of the registered model in the MLflow Model Registry. This name is used to retrieve the current champion model and to register new candidate models.",
    )
    min_improvement = Parameter(
        "min-improvement",
        default=0.01,
        type=float,
        help="Minimum relative RMSE improvement (as a fraction, e.g., 0.01 for 1%) required for a candidate model to be considered for promotion over the current champion. This threshold prevents promotion of models that offer only marginal gains.",
    )
    simulate_failure = Parameter(
        "simulate-failure",
        default=False,
        type=bool,
        help="If True, raise an intentional exception in the retrain step (for demo purposes).",
    )

    # ── Step: start ───────────────────────────────────────────────────

    @step
    def start(self):
        """Initializes MLflow and prepares the decision log for the pipeline run.

        This is the first step of the Metaflow, responsible for setting up the MLflow
        tracking environment and initializing flow-specific artifacts that will be
        used and updated throughout the pipeline.

        Logic Explanation:
        1.  **MLflow Initialization**: Calls ``model_utils.init_mlflow()`` to set the
            MLflow tracking URI and experiment name, ensuring all subsequent MLflow
            operations are correctly associated.
        2.  **Main MLflow Run**: Starts a new MLflow run for the entire pipeline.
            The ``run_id`` of this main run is stored as a Metaflow artifact
            (``self.mlflow_run_id``) so that all subsequent steps can log to the
            same parent run. The run is immediately ended and will be re-opened
            (resumed) by each step as needed.
        3.  **Artifact Initialization**: Initializes several Metaflow artifacts:
            -   ``self.decisions``: An empty list to accumulate decision records from various gates.
            -   ``self.is_bootstrap``: Flag to indicate if an initial model is being trained.
            -   ``self.promoted``: Flag to track if a candidate model was promoted.
            -   ``self.retrain_needed``: Flag to indicate if retraining is recommended.
            -   ``self.hard_pass``: Assumed ``True`` initially, set to ``False`` if integrity checks fail.
            -   ``self.candidate_version``: Stores the version of a newly trained candidate model.
        4.  **Console Output**: Prints a summary header for the pipeline run.
        5.  **Transition**: Moves to the ``load_data`` step.

        Args:
            self: The Metaflow FlowSpec instance.

        Returns:
            None: This step transitions to the next step using ``self.next()``.
        """
        _log_step_banner("STEP 0/8: START — Initializing MLflow and flow parameters")

        # Initialize MLflow tracking URI and experiment name.
        model_utils.init_mlflow(self.model_name)

        # Start the main MLflow run and persist its ID as an artifact
        # The run_id is stored as a Metaflow artifact to allow subsequent steps
        # to log to this same parent run.
        run = mlflow.start_run(run_name="capstone_pipeline")
        self.mlflow_run_id = run.info.run_id
        # End the run immediately; each step will resume it as needed.
        mlflow.end_run()  # close immediately; we'll re-open per step

        # Initialize flow-specific artifacts.
        self.decisions: list = []
        self.is_bootstrap = False
        self.promoted = False
        self.retrain_needed = False
        self.hard_pass = True  # assume pass until proven otherwise
        self.candidate_version = None

        # Print a header for the console output.
        logger.info(f"  Reference : {self.reference_path}")
        logger.info(f"  Batch     : {self.batch_path}")
        logger.info(f"  Model     : {self.model_name}")

        # Transition to the next step in the pipeline.
        self.next(self.load_data)

    # ── Step A: load_data ─────────────────────────────────────────────

    @step
    def load_data(self):
        """Loads reference and new batch data, and logs initial metadata to MLflow.

        This step is responsible for reading the input Parquet files specified by
        ``self.reference_path`` and ``self.batch_path``. It stores the raw DataFrames
        as Metaflow artifacts (``self.ref_raw`` and ``self.batch_raw``) for use in
        subsequent steps.

        Logic Explanation:
        1.  **Load Data**: Reads the Parquet files into pandas DataFrames.
        2.  **Print Shapes**: Outputs the shapes of the loaded DataFrames to the console.
        3.  **Resume MLflow Run**: Calls ``_ensure_mlflow()`` to resume the main MLflow run
            (identified by ``self.mlflow_run_id``) to ensure all logging is associated
            with the current pipeline execution.
        4.  **Log Parameters**: Logs the input paths (``reference_path``, ``batch_path``)
            and the ``model_name`` as MLflow parameters.
        5.  **Log Metrics**: Logs the number of rows and columns for both the reference
            and batch datasets as MLflow metrics.
        6.  **Log Date Range (if applicable)**: If a datetime column (defined in ``config.DATETIME_COL``)
            exists in the batch data, it extracts the minimum and maximum dates and logs
            them as MLflow tags (``batch_date_min``, ``batch_date_max``). This provides
            useful context about the time period covered by the new data.
        7.  **End MLflow Run**: Ensures the MLflow run is ended, releasing resources.
        8.  **Transition**: Moves to the ``integrity_gate`` step.

        Args:
            self: The Metaflow FlowSpec instance.

        Returns:
            None: This step transitions to the next step using ``self.next()``.
        """
        _log_step_banner("STEP 1/8: LOAD DATA — Reading reference and batch parquet files")

        # Load the raw reference and batch data from the specified Parquet paths.
        self.ref_raw = pd.read_parquet(self.reference_path)
        self.batch_raw = pd.read_parquet(self.batch_path)

        # Print the shapes of the loaded dataframes for immediate feedback.
        logger.info(f"Reference shape: {self.ref_raw.shape}")
        logger.info(f"Batch shape    : {self.batch_raw.shape}")

        # Resume the main MLflow run to log metadata specific to this step.
        _ensure_mlflow(self.mlflow_run_id)
        try:
            # Log input parameters and basic data statistics.
            mlflow.log_param("reference_path", self.reference_path)
            mlflow.log_param("batch_path", self.batch_path)
            mlflow.log_param("model_name", self.model_name)
            mlflow.log_metric("ref_rows", self.ref_raw.shape[0])
            mlflow.log_metric("ref_cols", self.ref_raw.shape[1])
            mlflow.log_metric("batch_rows", self.batch_raw.shape[0])
            mlflow.log_metric("batch_cols", self.batch_raw.shape[1])

            # If a datetime column is present, log the date range of the batch data as MLflow tags.
            if config.DATETIME_COL in self.batch_raw.columns:
                dt = pd.to_datetime(
                    self.batch_raw[config.DATETIME_COL], errors="coerce"
                )
                dt_valid = dt.dropna()
                if len(dt_valid) > 0:
                    mlflow.set_tag("batch_date_min", str(dt_valid.min()))
                    mlflow.set_tag("batch_date_max", str(dt_valid.max()))
        finally:
            # Ensure the MLflow run is always ended.
            mlflow.end_run()

        # Transition to the next step: integrity_gate.
        self.next(self.integrity_gate)

    # ── Step B: integrity_gate ────────────────────────────────────────

    @step
    def integrity_gate(self):
        """Runs hard and soft integrity checks on the new data batch.

        This step is a critical gate in the pipeline, ensuring the quality and validity
        of the incoming data. It performs both 'hard' checks (e.g., schema validation,
        missing critical values) that can halt the pipeline, and 'soft' checks (e.g.,
        data drift detection) that may issue warnings but allow the pipeline to proceed.

        Logic Explanation:
        1.  **Run Integrity Checks**: Calls ``integrity_checks.run_integrity_checks()``
            with the raw reference and batch data. This function returns a boolean
            indicating if hard checks passed, a boolean for soft warnings, and a detailed
            report.
        2.  **Update Artifacts**: Stores the results of the checks in Metaflow artifacts:
            ``self.hard_pass`` (boolean) and ``self.integrity_warn`` (boolean).
        3.  **Resume MLflow Run**: Resumes the main MLflow run.
        4.  **Log Decision**: Uses ``decision_logger.make_integrity_decision()`` to create
            a structured decision record based on the integrity check results. This decision
            is then logged as an MLflow artifact (``integrity_decision.json``) and appended
            to ``self.decisions``.
        5.  **Log MLflow Tags**: Sets MLflow tags (``integrity_hard_pass``,
            ``integrity_soft_warnings``, ``integrity_status``) for quick visibility in the UI.
        6.  **Console Output**: Prints a summary of the integrity check outcome to the console.
        7.  **Conditional Logic**: Although the pipeline proceeds linearly to the next step
            (``feature_engineering``), downstream steps will check ``self.hard_pass`` to
            determine if they should execute or skip their operations.
        8.  **End MLflow Run**: Ensures the MLflow run is ended.
        9.  **Transition**: Moves to the ``feature_engineering`` step.

        Args:
            self: The Metaflow FlowSpec instance.

        Returns:
            None: This step transitions to the next step using ``self.next()``.
        """
        _log_step_banner("STEP 2/8: INTEGRITY GATE — Running hard rules + soft checks on raw batch")

        # Resume the main MLflow run to log decision and tags.
        _ensure_mlflow(self.mlflow_run_id)
        try:
            # Execute both hard and soft integrity checks using the `integrity_checks` module.
            self.hard_pass, self.integrity_warn, report = (
                integrity_checks.run_integrity_checks(self.ref_raw, self.batch_raw)
            )

            # Create a structured decision record for the integrity gate.
            integrity_decision = decision_logger.make_integrity_decision(
                hard_pass=self.hard_pass,
                hard_failures=report["hard_failures"],
                soft_warnings=self.integrity_warn,
                soft_report=report.get("soft_report") or {},
            )
            # Log the decision as an MLflow artifact and add it to the flow's decision list.
            decision_logger.log_decision_to_mlflow(
                integrity_decision, artifact_name="integrity_decision.json"
            )
            self.decisions.append(integrity_decision)

            # Set MLflow tags to provide a quick overview of the integrity check results.
            mlflow.set_tag("integrity_hard_pass", str(self.hard_pass))
            mlflow.set_tag("integrity_soft_warnings", str(self.integrity_warn))
            mlflow.set_tag("integrity_status", report["overall_status"])
        finally:
            # Ensure the MLflow run is always ended.
            mlflow.end_run()

        # Provide console feedback based on the integrity check results.
        if not self.hard_pass:
            logger.info("❌  Batch REJECTED by hard integrity checks.")
            for f in report["hard_failures"]:
                logger.info(f"    • {f}")
        elif self.integrity_warn:
            logger.info("⚠️  Batch passed hard checks but has soft warnings.")
        else:
            logger.info("✅  Batch passed all integrity checks.")

        # Proceed to the next step. Downstream steps will check `self.hard_pass`
        # to decide whether to execute their logic or skip.
        self.next(self.feature_engineering)

    # ── Step C: feature_engineering ───────────────────────────────────

    @step
    def feature_engineering(self):
        """Applies deterministic feature transformations to reference and batch data.

        This step transforms the raw data into a format suitable for model inference
        or training. It uses the ``feature_engineering`` module to apply consistent
        feature engineering steps to both the reference and new batch datasets.

        Logic Explanation:
        1.  **Conditional Skip**: Checks ``self.hard_pass``. If ``False`` (batch rejected
            by integrity gate), it skips feature engineering, sets engineered data
            artifacts to ``None``, and transitions to ``load_champion``.
        2.  **Engineer Features**: Calls ``feature_engineering.engineer_features()``
            to apply transformations to ``self.ref_raw`` and ``self.batch_raw``.
            The results are stored as Metaflow artifacts: ``self.ref_eng`` and
            ``self.batch_eng``.
        3.  **Print Shapes**: Outputs the shapes of the engineered DataFrames to the console.
        4.  **Resume MLflow Run**: Resumes the main MLflow run.
        5.  **Log Feature Spec**: Logs the feature specification (details of engineered
            features) as an MLflow artifact (``feature_spec.json``). This ensures
            reproducibility and understanding of the features used.
        6.  **Log Metrics**: Logs the number of rows in the engineered reference and
            batch datasets as MLflow metrics.
        7.  **End MLflow Run**: Ensures the MLflow run is ended.
        8.  **Transition**: Moves to the ``load_champion`` step.

        Args:
            self: The Metaflow FlowSpec instance.

        Returns:
            None: This step transitions to the next step using ``self.next()``.
        """
        _log_step_banner("STEP 3/8: FEATURE ENGINEERING — Transforming raw data to model-ready features")

        # If the batch failed hard integrity checks, skip feature engineering.
        if not self.hard_pass:
            logger.info("⏭  Skipping feature engineering (batch rejected).")
            self.ref_eng = None
            self.batch_eng = None
            self.next(self.load_champion)
            return

        # Apply feature engineering to both reference and batch raw data.
        self.ref_eng = feature_engineering.engineer_features(self.ref_raw)
        self.batch_eng = feature_engineering.engineer_features(self.batch_raw)

        # Print the shapes of the engineered dataframes for verification.
        logger.info(f"Reference engineered shape: {self.ref_eng.shape}")
        logger.info(f"Batch engineered shape    : {self.batch_eng.shape}")

        # Resume the main MLflow run to log feature engineering metadata.
        _ensure_mlflow(self.mlflow_run_id)
        try:
            # Log the feature specification for traceability.
            mlflow.log_dict(
                feature_engineering.get_feature_spec(), "feature_spec.json"
            )
            # Log the number of rows in the engineered datasets as metrics.
            mlflow.log_metric("ref_eng_rows", self.ref_eng.shape[0])
            mlflow.log_metric("batch_eng_rows", self.batch_eng.shape[0])
        finally:
            # Ensure the MLflow run is always ended.
            mlflow.end_run()

        # Transition to the next step: load_champion.
        self.next(self.load_champion)

    # ── Step D: load_champion ─────────────────────────────────────────

    @step
    def load_champion(self):
        """Loads the current champion model from MLflow Model Registry or bootstraps a new one.

        This step is responsible for retrieving the currently designated champion model.
        If no champion model is found (e.g., on the very first run of the pipeline),
        it trains an initial model using the reference data and registers it as the
        first champion.

        Logic Explanation:
        1.  **Conditional Skip**: Checks ``self.hard_pass``. If ``False`` (batch rejected
            by integrity gate), it skips loading/bootstrapping, sets champion artifacts
            to ``None``, and transitions to ``evaluate_champion``.
        2.  **Resume MLflow Run**: Resumes the main MLflow run.
        3.  **Load Champion**: Calls ``model_utils.load_champion_model()`` to attempt
            to load the model aliased as ``@champion`` from the MLflow Model Registry.
        4.  **Bootstrap Logic (if no champion)**:
            -   If no champion is found, it sets ``self.is_bootstrap = True``.
            -   It then trains a new model using the engineered reference data
                (``self.ref_eng``) via ``model_utils.train_model()``.
            -   This initial model is logged and registered within a *nested* MLflow run
                to keep the bootstrap training separate but linked to the main pipeline run.
            -   The newly registered model is then aliased as ``@champion`` in the MLflow
                Model Registry, and relevant tags (``role: champion``, ``bootstrap: true``)
                are set.
            -   The version of this bootstrap model is stored in ``self.champion_version``
                and its URI in ``self.champion_model_uri``.
        5.  **Existing Champion Logic (if champion exists)**:
            -   If a champion is found, its version and URI are stored in
                ``self.champion_version`` and ``self.champion_model_uri`` respectively.
            -   ``self.is_bootstrap`` remains ``False``.
        6.  **Console Output**: Prints messages indicating whether a champion was loaded or bootstrapped.
        7.  **End MLflow Run**: Ensures the MLflow run is ended.
        8.  **Transition**: Moves to the ``evaluate_champion`` step.

        Args:
            self: The Metaflow FlowSpec instance.

        Returns:
            None: This step transitions to the next step using ``self.next()``.
        """
        _log_step_banner("STEP 4/8: LOAD CHAMPION — Loading current champion model from registry")

        # If the batch failed hard integrity checks, skip loading/bootstrapping a champion.
        if not self.hard_pass:
            logger.info("⏭  Skipping load_champion (batch rejected).")
            self.champion_version = None
            self.champion_model_uri = None
            self.cv_rmse_ref = None
            self.next(self.evaluate_champion)
            return

        # Default: no CV RMSE (non-bootstrap path computes ref RMSE in evaluate_champion)
        self.cv_rmse_ref = None

        # Resume the main MLflow run to log model loading/registration details.
        _ensure_mlflow(self.mlflow_run_id)
        try:
            # Attempt to load the current champion model from the MLflow Model Registry.
            champion_model, champion_version = model_utils.load_champion_model(
                self.model_name
            )

            feature_cols = feature_engineering.get_feature_columns()

            if champion_model is not None:
                # If a champion model exists, store its version and URI.
                self.champion_version = champion_version
                self.champion_model_uri = f"models:/{self.model_name}@champion"
                self.is_bootstrap = False
                logger.info(f"✅  Loaded champion model v{champion_version}")
            else:
                # If no champion exists, this is the first run, so bootstrap an initial model.
                logger.info("🆕  No champion found — bootstrapping initial model.")
                self.is_bootstrap = True

                # Prepare full reference data for training and CV evaluation.
                X_ref = self.ref_eng[feature_cols]
                y_ref = self.ref_eng[config.TARGET_COL]

                # 1. Compute CV RMSE (out-of-sample reference performance estimate).
                #    This trains 5 temporary models internally — we only keep the RMSE.
                cv_model = model_utils.train_model(X_ref, y_ref)
                cv_scores = cross_val_score(
                    cv_model, X_ref, y_ref,
                    cv=5,
                    scoring='neg_root_mean_squared_error',
                    n_jobs=-1,
                )
                self.cv_rmse_ref = float(-cv_scores.mean())
                logger.info("Bootstrap CV RMSE on reference (5-fold): %.4f", self.cv_rmse_ref)

                # 2. Train the FINAL champion on ALL reference data (no holdout).
                initial_model = model_utils.train_model(X_ref, y_ref)

                # Log and register the bootstrap model within a nested MLflow run.
                # This keeps the bootstrap training run separate but linked to the main pipeline run.
                with mlflow.start_run(
                    run_name="bootstrap_train", nested=True
                ) as bootstrap_run:
                    mlflow.log_params(model_utils.DEFAULT_PARAMS)
                    mlflow.log_metric("bootstrap_train_rows", len(X_ref))
                    mlflow.log_metric("bootstrap_cv_rmse", self.cv_rmse_ref)
                    run_id = model_utils.log_model_to_mlflow(
                        initial_model, X_ref, self.model_name
                    )

                # Set the @champion alias on the newly registered bootstrap version.
                client = MlflowClient()
                versions = client.search_model_versions(
                    f"name='{self.model_name}'"
                )
                # Find the latest version, which should be the one just registered.
                latest_version = max(versions, key=lambda v: int(v.version))
                bootstrap_version = int(latest_version.version)

                client.set_registered_model_alias(
                    self.model_name, "champion", str(bootstrap_version)
                )
                # Tag the bootstrap model version with its role and bootstrap status.
                client.set_model_version_tag(
                    self.model_name, str(bootstrap_version), "role", "champion"
                )
                client.set_model_version_tag(
                    self.model_name,
                    str(bootstrap_version),
                    "bootstrap",
                    "true",
                )
                # Store the fair out-of-sample reference RMSE (CV estimate)
                # so that evaluate_champion can compare against it instead of
                # re-evaluating on the training data (which would be in-sample).
                client.set_model_version_tag(
                    self.model_name,
                    str(bootstrap_version),
                    "ref_rmse",
                    str(self.cv_rmse_ref),
                )

                # Store the bootstrap model's version and URI as flow artifacts.
                self.champion_version = bootstrap_version
                self.champion_model_uri = f"models:/{self.model_name}@champion"
                logger.info(
                    f"✅  Bootstrap model registered as v{bootstrap_version} (@champion)"
                )
        finally:
            # Ensure the MLflow run is always ended.
            mlflow.end_run()

        # Transition to the next step: evaluate_champion.
        self.next(self.evaluate_champion)

    # ── Step E: evaluate_champion ─────────────────────────────────────

    @step
    def evaluate_champion(self):
        """Evaluates the champion model on both batch and reference data, decides on retraining.

        This step assesses how well the current champion model performs on the latest
        incoming data **and** on the reference data.  By comparing the champion's RMSE
        on the new batch against its RMSE on the reference data (where it was originally
        trained/validated), the pipeline can detect performance degradation caused by
        data drift or concept drift.

        Logic Explanation:
        1.  **Conditional Skip**: Checks ``self.hard_pass``. If ``False`` (batch rejected
            by integrity gate), it skips evaluation, sets metrics artifacts to ``None``,
            and transitions to ``retrain``.
        2.  **Resume MLflow Run**: Resumes the main MLflow run.
        3.  **Prepare Evaluation Data**: Extracts feature columns (``X_eval``) and the target
            column (``y_eval``) from the engineered batch data (``self.batch_eng``).
        4.  **Reload Champion Model**: Loads the champion model using its URI (``self.champion_model_uri``)
            from the MLflow Model Registry.
        5.  **Evaluate Champion on Batch**: Calls ``model_utils.evaluate_model()`` to calculate
            performance metrics (RMSE, MAE, R², Median AE) for the champion model on the new
            batch.  Results stored in ``self.champion_metrics``.
        6.  **Evaluate Champion on Reference**: Evaluates the champion model on the reference
            engineered data to establish its baseline performance.  The reference RMSE is stored
            in ``self.rmse_champion_on_ref`` for use in the retrain decision and downstream
            ``candidate_gate`` step.
        7.  **Compute Baseline RMSE**: Calculates a naive baseline RMSE for informational
            logging (stored in ``self.rmse_baseline``).
        8.  **Log Metrics**: Logs champion RMSE on batch, champion RMSE on reference, MAE,
            R², and baseline RMSE to MLflow.
        9.  **Compute Degradation**: Calculates the percentage increase in champion RMSE from
            reference to batch.  This is the key degradation indicator.
        10. **Retrain Decision**: Calls ``model_utils.should_retrain()`` comparing batch RMSE
            vs reference RMSE.  If ``self.is_bootstrap`` is ``True``, retraining is skipped.
        11. **Log Retrain Decision**: Uses ``decision_logger.make_retrain_decision()`` to create
            a structured decision record, logged as ``retrain_decision.json``.
        12. **Log MLflow Tag**: Sets ``retrain_recommended`` tag.
        13. **Console Output**: Prints evaluation results and the retraining recommendation.
        14. **End MLflow Run**: Ensures the MLflow run is ended.
        15. **Transition**: Moves to the ``retrain`` step.

        Args:
            self: The Metaflow FlowSpec instance.

        Returns:
            None: This step transitions to the next step using ``self.next()``.
        """
        _log_step_banner("STEP 5/8: EVALUATE CHAMPION — Computing RMSE on batch & ref, deciding retrain")

        # If the batch failed hard integrity checks, skip model evaluation.
        if not self.hard_pass:
            logger.info("⏭  Skipping evaluate_champion (batch rejected).")
            self.champion_metrics = None
            self.rmse_baseline = None
            self.rmse_champion_on_ref = None
            self.next(self.retrain)
            return

        # Resume the main MLflow run to log evaluation metrics and decisions.
        _ensure_mlflow(self.mlflow_run_id)
        try:
            feature_cols = feature_engineering.get_feature_columns()

            # Prepare evaluation data from the engineered batch.
            X_eval = self.batch_eng[feature_cols]
            y_eval = self.batch_eng[config.TARGET_COL]

            X_ref = self.ref_eng[feature_cols]
            y_ref = self.ref_eng[config.TARGET_COL]

            # Reload the champion model from the MLflow Model Registry.
            champion_model = mlflow.pyfunc.load_model(self.champion_model_uri)

            # Evaluate the champion model on the new batch data.
            self.champion_metrics = model_utils.evaluate_model(
                champion_model, X_eval, y_eval
            )

            # Determine reference RMSE for the champion.
            # We prefer the stored ``ref_rmse`` tag on the champion model
            # version because it is always a fair out-of-sample estimate:
            #   • Bootstrap champion → CV RMSE (5-fold cross-validation)
            #   • Promoted candidate → candidate_ref_rmse (evaluated on
            #     reference data the candidate was NOT trained on)
            # Falling back to live evaluation would be in-sample for the
            # bootstrap champion (trained on the same reference data),
            # producing an artificially low baseline and triggering
            # spurious retrains.
            client = MlflowClient()
            mv = client.get_model_version_by_alias(self.model_name, "champion")
            stored_ref_rmse = mv.tags.get("ref_rmse")

            if stored_ref_rmse is not None:
                self.rmse_champion_on_ref = float(stored_ref_rmse)
                logger.info(
                    "Using stored ref_rmse tag from champion v%s: %.4f",
                    mv.version,
                    self.rmse_champion_on_ref,
                )
            elif self.is_bootstrap and hasattr(self, 'cv_rmse_ref') and self.cv_rmse_ref is not None:
                # Fallback for bootstrap run within the same flow execution
                self.rmse_champion_on_ref = self.cv_rmse_ref
                logger.info("Using bootstrap CV RMSE for reference: %.4f", self.rmse_champion_on_ref)
            else:
                # Last resort: evaluate champion on reference (may be in-sample)
                champion_ref_metrics = model_utils.evaluate_model(
                    champion_model, X_ref, y_ref
                )
                self.rmse_champion_on_ref = champion_ref_metrics["rmse"]
                logger.warning(
                    "No stored ref_rmse tag found — using live evaluation: %.4f "
                    "(may be in-sample if champion was trained on reference data)",
                    self.rmse_champion_on_ref,
                )

            # Compute a naive baseline RMSE for informational logging.
            self.rmse_baseline = model_utils.compute_baseline_rmse(y_eval)

            # Log champion model performance metrics to MLflow.
            mlflow.log_metric("rmse_champion", self.champion_metrics["rmse"])
            mlflow.log_metric("mae_champion", self.champion_metrics["mae"])
            mlflow.log_metric("r2_champion", self.champion_metrics["r2"])
            mlflow.log_metric("rmse_champion_on_ref", self.rmse_champion_on_ref)
            mlflow.log_metric("rmse_baseline", self.rmse_baseline)

            # Calculate the percentage increase in RMSE from reference to batch.
            # This is the key degradation metric: how much worse does the champion
            # perform on new data compared to the data it was trained on?
            if self.rmse_champion_on_ref > 0:
                rmse_increase_pct = (
                    (self.champion_metrics["rmse"] - self.rmse_champion_on_ref)
                    / self.rmse_champion_on_ref
                )
            else:
                rmse_increase_pct = 0.0
            mlflow.log_metric("rmse_increase_pct", rmse_increase_pct)

            # Print evaluation results to the console.
            logger.info(f"Champion RMSE on batch : {self.champion_metrics['rmse']:.4f}")
            logger.info(f"Champion RMSE on ref   : {self.rmse_champion_on_ref:.4f}")
            logger.info(f"Baseline RMSE (naive)  : {self.rmse_baseline:.4f}")
            logger.info(f"RMSE degradation %     : {rmse_increase_pct:.2%}")

            # Determine if retraining is needed, with special handling for bootstrap runs.
            if self.is_bootstrap:
                self.retrain_needed = False
                retrain_reason = "Bootstrap run — no retrain needed."
                logger.info("🆕  Bootstrap run — skipping retrain.")
            else:
                self.retrain_needed, retrain_reason = model_utils.should_retrain(
                    rmse_champion_on_batch=self.champion_metrics["rmse"],
                    rmse_champion_on_ref=self.rmse_champion_on_ref,
                    integrity_warn=self.integrity_warn,
                )

            # Create and log a structured decision record for the retraining gate.
            retrain_decision = decision_logger.make_retrain_decision(
                retrain_needed=self.retrain_needed,
                rmse_champion_on_batch=self.champion_metrics["rmse"],
                rmse_champion_on_ref=self.rmse_champion_on_ref,
                rmse_increase_pct=rmse_increase_pct,
                reason=retrain_reason,
                rmse_baseline=self.rmse_baseline,
            )
            decision_logger.log_decision_to_mlflow(
                retrain_decision, artifact_name="retrain_decision.json"
            )
            self.decisions.append(retrain_decision)

            # Set an MLflow tag to indicate whether retraining was recommended.
            mlflow.set_tag("retrain_recommended", str(self.retrain_needed))

            # Provide console feedback on the retraining decision.
            if self.retrain_needed:
                logger.info("🔄  Retrain recommended — proceeding to retrain step.")
            elif not self.is_bootstrap:
                logger.info("✅  Champion performance acceptable — no retrain needed.")
        finally:
            # Ensure the MLflow run is always ended.
            mlflow.end_run()

        # Transition to the next step: retrain.
        self.next(self.retrain)

    # ── Step F: retrain ───────────────────────────────────────────────

    @step
    def retrain(self):
        """Trains a new candidate model on combined reference and batch data.

        This step is executed if the ``evaluate_champion`` step determined that
        retraining is necessary. It combines the historical reference data with the
        new incoming batch data to train a new model, which then becomes a candidate
        for promotion.

        Logic Explanation:
        1.  **Conditional Skip**: Checks ``self.hard_pass`` and ``self.retrain_needed``.
            If the batch was rejected or retraining is not needed, it skips training,
            sets candidate artifacts to ``None``, and transitions to ``candidate_gate``.
        2.  **Resume MLflow Run**: Resumes the main MLflow run.
        3.  **Prepare Training Data**: Concatenates the engineered reference data
            (``self.ref_eng``) and engineered batch data (``self.batch_eng``) to create
            an expanded training dataset. Extracts features (``X_train``) and target
            (``y_train``).
        4.  **Train Candidate Model**: Calls ``model_utils.train_model()`` to train a new
            model on the combined dataset.
        5.  **Evaluate Candidate on Batch**: Evaluates the newly trained candidate model
            on the current batch data (``self.batch_eng``) to get ``self.candidate_metrics``.
        6.  **Evaluate Candidate on Reference**: Evaluates the candidate model on the
            original reference data (``self.ref_eng``) to assess its stability and
            ensure it hasn't regressed on known good data. Results stored in
            ``self.candidate_ref_metrics``.
        7.  **Console Output**: Prints RMSE of the candidate on both batch and reference data.
        8.  **Log Candidate in Nested Run**: The candidate model and its training parameters
            and metrics are logged within a *nested* MLflow run (``candidate_train``).
            This keeps the training details separate but linked to the main pipeline run.
            The model is also registered in the MLflow Model Registry.
        9.  **Identify and Tag Candidate Version**: Retrieves the version of the newly
            registered candidate model and sets MLflow Model Version tags (``role: candidate``,
            ``validation_status: pending``).
        10. **Log Candidate Metrics to Parent Run**: Key metrics of the candidate model
            (RMSE, MAE, R2 on batch, RMSE on reference) are also logged to the main
            pipeline MLflow run for easy comparison.
        11. **Console Output**: Confirms successful registration of the candidate model.
        12. **End MLflow Run**: Ensures the MLflow run is ended.
        13. **Transition**: Moves to the ``candidate_gate`` step.

        Args:
            self: The Metaflow FlowSpec instance.

        Returns:
            None: This step transitions to the next step using ``self.next()``.
        """
        _log_step_banner("STEP 6/8: RETRAIN — Training candidate model on combined data")

        # Check for simulated failure (demo purposes).
        # This allows demonstrating Metaflow's resume capability by intentionally
        # crashing the retrain step regardless of whether retrain is needed.
        # On resume (origin_run_id is set), skip the simulated failure so the
        # pipeline can complete successfully.
        if self.simulate_failure:
            if current.origin_run_id is not None:
                logger.info("🔄  Resumed run — skipping simulated failure.")
            else:
                raise RuntimeError(
                    "💥 Simulated failure in retrain step! "
                    "To resume: remove --simulate-failure and run: python flow.py resume"
                )

        # If the batch was rejected or retraining is not needed, skip this step.
        if not self.hard_pass or not self.retrain_needed:
            logger.info("⏭  Skipping retrain (not needed).")
            self.candidate_metrics = None
            self.candidate_ref_metrics = None
            self.candidate_version = None
            self.candidate_run_id = None
            self.next(self.candidate_gate)
            return

        # ── Idempotency guard for repeated resumes ─────────────────────
        # Metaflow's `resume` always re-executes failed steps from the
        # origin run, even if a previous resume already completed them
        # successfully.  This guard prevents duplicate model registration
        # by checking whether a candidate was already registered by a
        # prior resume of the same origin run.
        if current.origin_run_id is not None:
            model_utils.init_mlflow()
            client = MlflowClient()
            existing_versions = client.search_model_versions(
                f"name='{self.model_name}'"
            )
            for mv in existing_versions:
                if mv.tags.get("metaflow_origin_run_id") == current.origin_run_id:
                    logger.info(
                        "♻️  Candidate v%s already registered by a previous "
                        "resume of origin run %s — skipping duplicate retrain.",
                        mv.version,
                        current.origin_run_id,
                    )
                    self.candidate_version = int(mv.version)
                    self.candidate_run_id = mv.run_id
                    # Reconstruct candidate metrics from the existing model
                    # version's MLflow run so downstream steps have them.
                    prev_run = client.get_run(mv.run_id)
                    self.candidate_metrics = {
                        "rmse": prev_run.data.metrics.get("candidate_batch_rmse", 0.0),
                        "mae": prev_run.data.metrics.get("candidate_batch_mae", 0.0),
                        "r2": prev_run.data.metrics.get("candidate_batch_r2", 0.0),
                        "median_ae": prev_run.data.metrics.get("candidate_batch_median_ae", 0.0),
                    }
                    self.candidate_ref_metrics = {
                        "rmse": prev_run.data.metrics.get("candidate_ref_rmse", 0.0),
                        "mae": prev_run.data.metrics.get("candidate_ref_mae", 0.0),
                        "r2": prev_run.data.metrics.get("candidate_ref_r2", 0.0),
                        "median_ae": prev_run.data.metrics.get("candidate_ref_median_ae", 0.0),
                    }
                    self.next(self.candidate_gate)
                    return

        # Resume the main MLflow run to log training details and candidate model.
        _ensure_mlflow(self.mlflow_run_id)
        try:
            feature_cols = feature_engineering.get_feature_columns()

            # Combine engineered reference and batch data to create the training dataset.
            combined = pd.concat(
                [self.ref_eng, self.batch_eng], ignore_index=True
            )
            X_train = combined[feature_cols]
            y_train = combined[config.TARGET_COL]

            logger.info(f"Training on combined data: {combined.shape[0]} rows")

            # Tune hyperparameters using Optuna Bayesian search.
            logger.info("🔍  Running Optuna hyperparameter tuning (%d trials)…",
                        config.OPTUNA_N_TRIALS)
            best_params = model_utils.tune_hyperparams(X_train, y_train)
            mlflow.log_params({f"optuna_{k}": v for k, v in best_params.items()})

            # Train the new candidate model with the tuned parameters.
            candidate = model_utils.train_model(X_train, y_train, params=best_params)

            # Evaluate the candidate model on the new batch data.
            X_eval = self.batch_eng[feature_cols]
            y_eval = self.batch_eng[config.TARGET_COL]
            self.candidate_metrics = model_utils.evaluate_model(
                candidate, X_eval, y_eval
            )

            # Evaluate the candidate model on the reference data for stability checks.
            X_ref = self.ref_eng[feature_cols]
            y_ref = self.ref_eng[config.TARGET_COL]
            self.candidate_ref_metrics = model_utils.evaluate_model(
                candidate, X_ref, y_ref
            )

            # Print candidate model performance to console.
            logger.info(f"Candidate RMSE on batch : {self.candidate_metrics['rmse']:.4f}")
            logger.info(f"Candidate RMSE on ref   : {self.candidate_ref_metrics['rmse']:.4f}")

            # Log the candidate model and its metrics within a nested MLflow run.
            with mlflow.start_run(
                run_name="candidate_train", nested=True
            ) as candidate_run:
                mlflow.log_params(model_utils.DEFAULT_PARAMS)
                mlflow.log_metric("train_rows", len(X_train))

                # Log detailed metrics for the candidate model.
                for prefix, metrics in [
                    ("candidate_batch", self.candidate_metrics),
                    ("candidate_ref", self.candidate_ref_metrics),
                ]:
                    for k, v in metrics.items():
                        mlflow.log_metric(f"{prefix}_{k}", v)

                # Log and register the candidate model in the MLflow Model Registry.
                candidate_run_id = model_utils.log_model_to_mlflow(
                    candidate, X_train, self.model_name
                )
                self.candidate_run_id = candidate_run_id

            # Retrieve the version of the newly registered candidate model.
            client = MlflowClient()
            versions = client.search_model_versions(
                f"name='{self.model_name}'"
            )
            latest_version = max(versions, key=lambda v: int(v.version))
            self.candidate_version = int(latest_version.version)

            # Tag the candidate model version in the MLflow Model Registry.
            client.set_model_version_tag(
                self.model_name,
                str(self.candidate_version),
                "role",
                "candidate",
            )
            client.set_model_version_tag(
                self.model_name,
                str(self.candidate_version),
                "validation_status",
                "pending",
            )
            # Tag with the Metaflow origin run ID for idempotency on
            # subsequent resumes (see guard above).
            origin_id = current.origin_run_id or current.run_id
            client.set_model_version_tag(
                self.model_name,
                str(self.candidate_version),
                "metaflow_origin_run_id",
                origin_id,
            )

            # Log key candidate metrics to the parent MLflow run for easy comparison.
            mlflow.log_metric("rmse_candidate", self.candidate_metrics["rmse"])
            mlflow.log_metric("mae_candidate", self.candidate_metrics["mae"])
            mlflow.log_metric("r2_candidate", self.candidate_metrics["r2"])
            mlflow.log_metric(
                "rmse_candidate_on_ref", self.candidate_ref_metrics["rmse"]
            )

            logger.info(f"✅  Candidate registered as v{self.candidate_version}")
        finally:
            # Ensure the MLflow run is always ended.
            mlflow.end_run()

        # Transition to the next step: candidate_gate.
        self.next(self.candidate_gate)

    # ── Step G: candidate_gate ────────────────────────────────────────

    @step
    def candidate_gate(self):
        """Decide whether to promote the candidate model to champion (Step G).

        This step implements the **promotion gate** — the final quality check
        before a newly trained candidate replaces the current champion in the
        MLflow Model Registry.

        Promotion logic (delegated to ``model_utils.should_promote``):
        * Compares candidate vs. champion RMSE on the *batch* data.
        * Compares candidate vs. champion RMSE on the *reference* data to
          detect overfitting or instability.
        * Considers whether soft integrity warnings were raised earlier.

        If promoted:
        * The ``@champion`` alias is flipped to the candidate version via
          ``model_utils.promote_champion``.
        * Batch predictions are generated with the newly promoted model and
          logged as a ``predictions.parquet`` artifact.

        If rejected:
        * The candidate version is tagged ``validation_status=rejected`` in
          the Model Registry, along with the rejection reason.

        Artifacts logged to MLflow:
            ``promotion_decision.json`` — structured promotion decision record.
            ``predictions.parquet``     — batch predictions (promotion only).

        Tags set:
            ``promotion_recommended`` — ``"True"`` or ``"False"``.
            ``promoted_version``      — version number (promotion only).
            ``previous_champion_version`` — old champion (promotion only).

        Metaflow artifacts set:
            self.promoted (bool): Whether the candidate was promoted.
            self.decisions (list): Appended with the promotion decision dict.
        """
        _log_step_banner("STEP 7/8: CANDIDATE GATE — Evaluating promotion criteria (P1-P4)")

        # ── Guard: skip if batch failed integrity or no retrain occurred ──
        if not self.hard_pass or not self.retrain_needed:
            logger.info("⏭  Skipping candidate_gate (no candidate to evaluate).")
            self.next(self.end)
            return

        # Resume the existing MLflow run for this pipeline execution
        _ensure_mlflow(self.mlflow_run_id)
        try:
            feature_cols = feature_engineering.get_feature_columns()

            # ── Reuse champion-on-reference RMSE from evaluate_champion step ──
            # This was already computed and stored as self.rmse_champion_on_ref
            # in the evaluate_champion step, avoiding redundant model inference.
            rmse_champion_on_ref = self.rmse_champion_on_ref

            # ── Promotion decision via business-rule helper ──
            promote, promote_reason = model_utils.should_promote(
                rmse_candidate=self.candidate_metrics["rmse"],
                rmse_champion=self.champion_metrics["rmse"],
                rmse_candidate_on_ref=self.candidate_ref_metrics["rmse"],
                rmse_champion_on_ref=rmse_champion_on_ref,
                integrity_warn=self.integrity_warn,
            )

            # ── Persist promotion decision as an MLflow artifact ──
            promotion_decision = decision_logger.make_promotion_decision(
                promote=promote,
                rmse_candidate=self.candidate_metrics["rmse"],
                rmse_champion=self.champion_metrics["rmse"],
                rmse_candidate_on_ref=self.candidate_ref_metrics["rmse"],
                rmse_champion_on_ref=rmse_champion_on_ref,
                reason=promote_reason,
            )
            decision_logger.log_decision_to_mlflow(
                promotion_decision, artifact_name="promotion_decision.json"
            )
            # Append to the running list for the final summary artifact
            self.decisions.append(promotion_decision)

            # Record promotion outcome as MLflow tag + metric
            mlflow.set_tag("promotion_recommended", str(promote))
            mlflow.log_metric("rmse_champion_on_ref", rmse_champion_on_ref)

            client = MlflowClient()

            if promote:
                # ── Promotion path: flip alias and generate predictions ──
                logger.info("🏆  Promoting candidate to champion!")
                self.promoted = True

                # Atomically move the @champion alias to the candidate version
                model_utils.promote_champion(
                    model_name=self.model_name,
                    new_version=self.candidate_version,
                    old_version=self.champion_version,
                    reason=promote_reason,
                )

                # Store the candidate's fair reference RMSE on the new
                # champion version so evaluate_champion can use it in
                # future runs (avoids in-sample evaluation).
                client.set_model_version_tag(
                    self.model_name,
                    str(self.candidate_version),
                    "ref_rmse",
                    str(self.candidate_ref_metrics["rmse"]),
                )

                # Tag the run with version lineage for traceability
                mlflow.set_tag("promoted_version", str(self.candidate_version))
                mlflow.set_tag(
                    "previous_champion_version", str(self.champion_version)
                )

                # ── Generate batch predictions with the newly promoted model ──
                new_champion_uri = f"models:/{self.model_name}@champion"
                new_champion = mlflow.pyfunc.load_model(new_champion_uri)
                X_batch = self.batch_eng[feature_cols].copy()
                # Location IDs are already int (converted in feature_engineering).
                preds = new_champion.predict(X_batch)

                # Build a DataFrame with predictions alongside original features
                pred_df = self.batch_eng.copy()
                pred_df["predicted_tip"] = preds

                # Write predictions to a temp Parquet file and log as artifact
                tmp_dir = tempfile.mkdtemp()
                pred_path = os.path.join(tmp_dir, "predictions.parquet")
                pred_df.to_parquet(pred_path, index=False)
                mlflow.log_artifact(pred_path)
                # Clean up temporary files to avoid disk clutter
                if os.path.exists(pred_path):
                    os.remove(pred_path)
                if os.path.isdir(tmp_dir):
                    os.rmdir(tmp_dir)

                logger.info(f"📊  Logged {len(pred_df)} batch predictions as artifact.")
            else:
                # ── Rejection path: tag the candidate version as rejected ──
                logger.info("❌  Candidate NOT promoted.")
                self.promoted = False

                # Mark the candidate version in the registry so it is clearly
                # identifiable as a failed promotion attempt.
                client.set_model_version_tag(
                    self.model_name,
                    str(self.candidate_version),
                    "validation_status",
                    "rejected",
                )
                client.set_model_version_tag(
                    self.model_name,
                    str(self.candidate_version),
                    "rejection_reason",
                    promote_reason,
                )
        finally:
            # Always close the MLflow run, even on error
            mlflow.end_run()

        self.next(self.end)

    # ── Step: end ─────────────────────────────────────────────────────

    @step
    def end(self):
        """Log all accumulated decisions and print a human-readable pipeline summary.

        This is the **terminal step** of the Metaflow DAG.  It performs two
        final duties:

        1. **Persist the combined decision log** — all decision dicts collected
           in ``self.decisions`` throughout the pipeline are written to a single
           ``all_decisions.json`` artifact via
           ``decision_logger.log_all_decisions``.  This provides a single
           audit-trail file that captures every gate outcome in one place.

        2. **Print a summary banner** — a concise, human-readable recap of
           how many decisions were logged and what the overall pipeline outcome
           was (e.g., batch rejected, champion promoted, no retrain needed).

        The MLflow run is resumed one last time to log the combined artifact,
        then closed via ``mlflow.end_run()`` in a ``finally`` block to
        guarantee cleanup even on error.

        Metaflow artifacts used:
            self.decisions (list[dict]): Accumulated decision records.
            self.hard_pass (bool): Whether the batch passed hard integrity.
            self.is_bootstrap (bool): Whether this was a bootstrap run.
            self.promoted (bool): Whether the candidate was promoted.
            self.retrain_needed (bool): Whether retraining was triggered.
            self.champion_metrics (dict | None): Champion eval metrics.
            self.champion_version (str): Previous champion version number.
            self.candidate_version (str): Candidate version number.
        """
        _log_step_banner("STEP 8/8: END — Logging decisions and printing summary")

        # Resume the MLflow run one final time to log the combined artifact
        _ensure_mlflow(self.mlflow_run_id)
        try:
            # Write all decision records as a single JSON artifact for auditing
            if self.decisions:
                decision_logger.log_all_decisions(self.decisions)
        finally:
            # Guarantee the MLflow run is closed regardless of errors
            mlflow.end_run()

        # ── Human-readable summary banner ──
        _log_step_banner("  Pipeline Summary")
        logger.info(f"  Decisions logged : {len(self.decisions)}")
        for d in self.decisions:
            logger.info(f"    [{d['stage']}] → {d['action']}")

        # Determine and display the overall pipeline outcome
        if not self.hard_pass:
            logger.info("  Outcome: Batch rejected at integrity gate.")
        elif self.is_bootstrap:
            logger.info("  Outcome: Bootstrap — initial champion registered.")
        elif self.promoted:
            logger.info(
                f"  Outcome: Champion promoted "
                f"(v{self.champion_version} → v{self.candidate_version})."
            )
        elif self.retrain_needed:
            logger.info("  Outcome: Candidate trained but NOT promoted.")
        elif self.champion_metrics is not None:
            logger.info("  Outcome: Champion performance acceptable — no retrain.")
        else:
            logger.info("  Outcome: Pipeline completed.")
        logger.info("=" * 60)


if __name__ == "__main__":
    GreenTaxiTipFlow()
