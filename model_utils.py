"""
model_utils.py — Model training, evaluation, registry, and promotion utilities.

Implements **Steps D–G** of the capstone design doc
(``08_mlops_capstone_project/design_doc.md``):

  **D. Train** — fit an XGBoost regressor on the current batch's
      feature-engineered data.
  **E. Evaluate** — compute regression metrics (RMSE, MAE, R², Median AE)
      on a held-out evaluation split.
  **F. Register** — create a new model *version* in the MLflow Model
      Registry and attach metadata tags.
  **G. Promote** — decide whether the newly trained *candidate* should
      replace the current *champion* model.

Champion / Challenger pattern
-----------------------------
The pipeline maintains at most two "roles" for registered model versions:

- **champion** — the production model, identified by the ``@champion``
  alias in the MLflow Model Registry.  All inference requests are served
  by this version.
- **candidate** (challenger) — a freshly trained model that has not yet
  been promoted.  It is registered as a new version but does *not*
  receive the ``@champion`` alias until it passes all promotion criteria
  (P1–P4 in ``should_promote``).

On the very first run (the **bootstrap case**) there is no champion yet.
``load_champion_model`` detects this and returns ``(None, None)``, which
tells ``flow.py`` to skip the retrain-decision step and go straight to
training a new model.

MLflow Model Registry mechanics
--------------------------------
- **Registered model** — a named container (``config.MODEL_NAME``) that
  holds an ordered sequence of *versions*.
- **Version** — an immutable snapshot of a trained model artifact, created
  by ``mlflow.register_model()`` or ``mlflow.sklearn.log_model(...,
  registered_model_name=...)``.  Each version gets an auto-incrementing
  integer ID.
- **Alias** — a mutable pointer (like a Git tag) that can be moved from
  one version to another.  We use ``@champion`` as the single alias.
  ``promote_champion()`` atomically moves this alias to the new version.
- **Tags** — key-value metadata attached to a version.  We use tags to
  record ``role``, ``promoted_at``, ``demoted_at``, ``promotion_reason``,
  ``trained_on_batches``, ``eval_batch_id``, ``validation_status``, and
  ``decision_reason``.

All constants and thresholds are imported from ``config.py``.
Registry patterns follow ``07_model_registry_deployment/072_train_register.py``
and ``07_model_registry_deployment/073_flip_aliases.py``.
"""

from __future__ import annotations

import math
import logging
from datetime import datetime, timezone

import numpy as np
import xgboost as xgb
import optuna

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    median_absolute_error,
    r2_score,
)

import config

logger = logging.getLogger(__name__)

# ── Default XGBoost hyper-parameters ──────────────────────────────────────

DEFAULT_PARAMS: dict = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
}


# ---------------------------------------------------------------------------
# 1. MLflow initialisation
# ---------------------------------------------------------------------------


def suppress_mlflow_logs() -> None:
    """Suppress noisy MLflow / Alembic log messages.

    Each Metaflow step runs as a separate subprocess that re-initialises
    MLflow, so these suppressions must be re-applied in every process.
    Centralising them here avoids duplicating the logger list in multiple
    modules.
    """
    logging.getLogger("alembic").setLevel(logging.WARNING)
    logging.getLogger("mlflow.store.db.utils").setLevel(logging.WARNING)
    logging.getLogger("mlflow.utils.requirements_utils").setLevel(logging.WARNING)
    logging.getLogger("mlflow.models.model").setLevel(logging.ERROR)
    logging.getLogger("mlflow.models.signature").setLevel(logging.ERROR)
    logging.getLogger("mlflow.tracking._model_registry.client").setLevel(logging.WARNING)
    logging.getLogger("mlflow.tracking").setLevel(logging.WARNING)


def init_mlflow(model_name: str | None = None) -> str:
    """Configure the MLflow tracking URI and experiment.

    This function sets up the connection to the MLflow tracking server
    and ensures that all subsequent runs are logged under a consistent
    experiment name.  It's typically called once at the start of a pipeline.

    Parameters
    ----------
    model_name : str | None
        This parameter is accepted for API consistency with other functions
        that require ``model_name``, but it is not used within this function.
        Callers can safely pass ``config.MODEL_NAME`` without branching logic.

    Returns
    -------
    str
        The experiment ID for the configured MLflow experiment.

    Notes
    -----
    - The tracking URI is loaded from ``config.MLFLOW_TRACKING_URI``.
      This can be a local path (e.g., ``sqlite:///mlruns.db``) or a remote
      server address.
    - The experiment name is loaded from ``config.MLFLOW_EXPERIMENT_NAME``.
      If an experiment with this name does not exist, MLflow creates it.
    """
    suppress_mlflow_logs()

    # Set the MLflow tracking URI. This tells MLflow where to store tracking data.
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)

    # Handle case where experiment was soft-deleted via the MLflow UI.
    # MLflow marks deleted experiments as lifecycle_stage="deleted" but keeps
    # them in the DB.  Calling set_experiment() on a deleted name raises
    # MlflowException, so we restore it first.
    client = MlflowClient()
    exp = client.get_experiment_by_name(config.MLFLOW_EXPERIMENT_NAME)
    if exp and exp.lifecycle_stage == "deleted":
        logger.info(
            "Restoring soft-deleted experiment '%s' (id=%s)",
            config.MLFLOW_EXPERIMENT_NAME,
            exp.experiment_id,
        )
        client.restore_experiment(exp.experiment_id)

    # Set the active MLflow experiment. All runs will be logged under this experiment.
    experiment = mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
    logger.info(
        "MLflow initialised — experiment=%s  id=%s",
        config.MLFLOW_EXPERIMENT_NAME,
        experiment.experiment_id,
    )
    return experiment.experiment_id


# ---------------------------------------------------------------------------
# 2. Training
# ---------------------------------------------------------------------------


def train_model(X_train, y_train, params: dict | None = None):
    """Train an XGBoost regressor and return the fitted model.

    This function trains an XGBoost regression model using the provided
    training data and hyperparameters. It merges default parameters with
    any user-provided overrides to ensure a consistent model configuration.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features. This DataFrame should contain the pre-processed
        features ready for model training.
    y_train : pd.Series | np.ndarray
        Training target values (``tip_amount``). This is the variable
        the model will learn to predict.
    params : dict | None
        Optional dictionary of XGBoost hyperparameters to override the
        ``DEFAULT_PARAMS``. This allows for flexible tuning of the model.

    Returns
    -------
    xgb.XGBRegressor
        The fitted XGBoost regression model, ready for evaluation or prediction.

    Notes
    -----
    - The ``DEFAULT_PARAMS`` are defined globally in this module and provide
      a sensible starting point for the XGBoost model.
    - ``enable_categorical=True`` is crucial for XGBoost to correctly handle
      categorical features (e.g., ``PULocationID``, ``DOLocationID``) without
      requiring explicit one-hot encoding, which can be memory-intensive.
    """
    # Merge default parameters with any provided overrides. This ensures that
    # all necessary parameters are present, with user-defined values taking
    # precedence.
    merged = {**DEFAULT_PARAMS, **(params or {})}
    # Initialize the XGBoost Regressor with the merged parameters.
    # Location IDs are already int (converted in feature_engineering),
    # so enable_categorical is not needed.
    model = xgb.XGBRegressor(**merged)
    # Fit the model to the training data.
    model.fit(X_train, y_train)
    logger.info("Model trained — params=%s", merged)
    return model


def tune_hyperparams(
    X_train,
    y_train,
    n_trials: int | None = None,
    cv_folds: int | None = None,
) -> dict:
    """Find optimal XGBoost hyperparameters using Optuna Bayesian search.

    Runs an Optuna study that evaluates candidate hyperparameter sets via
    cross-validation.  The search space covers the most impactful XGBoost
    parameters: ``n_estimators``, ``max_depth``, ``learning_rate``,
    ``subsample``, and ``colsample_bytree``.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series | np.ndarray
        Training target values.
    n_trials : int | None
        Number of Optuna trials.  Defaults to ``config.OPTUNA_N_TRIALS``.
    cv_folds : int | None
        Number of cross-validation folds.  Defaults to ``config.OPTUNA_CV_FOLDS``.

    Returns
    -------
    dict
        The best hyperparameter set found, ready to pass to
        ``train_model(X, y, params=best_params)``.
    """
    from sklearn.model_selection import cross_val_score

    n_trials = n_trials or config.OPTUNA_N_TRIALS
    cv_folds = cv_folds or config.OPTUNA_CV_FOLDS

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "random_state": 42,
        }
        model = xgb.XGBRegressor(**params)
        scores = cross_val_score(
            model, X_train, y_train,
            cv=cv_folds,
            scoring="neg_root_mean_squared_error",
        )
        return -scores.mean()  # Optuna minimizes

    # Suppress Optuna's verbose trial-by-trial logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    best = study.best_params
    logger.info(
        "Optuna tuning complete — %d trials, best RMSE=%.4f, params=%s",
        n_trials,
        study.best_value,
        best,
    )
    return best


# ---------------------------------------------------------------------------
# 3. Evaluation
# ---------------------------------------------------------------------------


def evaluate_model(model, X_eval, y_eval) -> dict:
    """Compute regression metrics on an evaluation set.

    This function calculates several standard regression metrics to assess
    the performance of a trained model on a given evaluation dataset.
    The metrics computed are Root Mean Squared Error (RMSE), Mean Absolute
    Error (MAE), R-squared (R²), and Median Absolute Error (Median AE).

    Parameters
    ----------
    model : fitted estimator
        A trained machine learning model object that exposes a ``.predict()``
        method, capable of generating predictions on new data.
    X_eval : pd.DataFrame
        Evaluation features. This DataFrame contains the input features
        used to make predictions.
    y_eval : pd.Series | np.ndarray
        True target values. These are the actual outcomes against which
        the model's predictions are compared.

    Returns
    -------
    dict
        A dictionary containing the computed regression metrics:
        - ``"rmse"`` (float): Root Mean Squared Error.
        - ``"mae"`` (float): Mean Absolute Error.
        - ``"r2"`` (float): R-squared score.
        - ``"median_ae"`` (float): Median Absolute Error.

    Notes
    -----
    - **RMSE** penalizes large errors more heavily, useful when large errors
      are particularly undesirable.
    - **MAE** provides a linear scoring that is less sensitive to outliers
      than RMSE.
    - **R²** indicates the proportion of the variance in the dependent
      variable that is predictable from the independent variables.
    - **Median AE** is robust to outliers, as it uses the median of absolute
      differences.
    """
    # Generate predictions using the provided model and evaluation features.
    # Location IDs are already int (converted in feature_engineering).
    y_pred = model.predict(X_eval)
    # Calculate various regression metrics.
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_eval, y_pred))),  # Root Mean Squared Error
        "mae": float(mean_absolute_error(y_eval, y_pred)),          # Mean Absolute Error
        "r2": float(r2_score(y_eval, y_pred)),                      # R-squared
        "median_ae": float(median_absolute_error(y_eval, y_pred)),  # Median Absolute Error
    }
    logger.info("Evaluation metrics: %s", metrics)
    return metrics


# ---------------------------------------------------------------------------
# 4. Baseline RMSE
# ---------------------------------------------------------------------------


def compute_baseline_rmse(y_eval) -> float:
    """Calculate the Root Mean Squared Error (RMSE) for a naive mean-prediction baseline.

    This baseline model simply predicts the mean of the target variable for
    all instances. It serves as a simple, easily interpretable benchmark
    against which the performance of more complex models can be compared.
    A good model should significantly outperform this baseline.

    Parameters
    ----------
    y_eval : array-like
        True target values (e.g., ``tip_amount``) from the evaluation set.
        Can be a pandas Series, numpy array, or list.

    Returns
    -------
    float
        The RMSE value when every prediction is the mean of ``y_eval``.

    Notes
    -----
    - The mean-prediction baseline is useful for understanding the inherent
      predictability of the target variable. If a complex model cannot beat
      this simple baseline, it suggests issues with the model, features, or data.
    - ``np.asarray(y_eval, dtype=float)`` ensures numerical stability and
      consistency, handling various input array-like types.
    """
    # Convert y_eval to a numpy array of floats for consistent numerical operations.
    y_eval = np.asarray(y_eval, dtype=float)
    # Create an array where every prediction is the mean of the true target values.
    mean_pred = np.full_like(y_eval, y_eval.mean())
    # Calculate the RMSE between the true values and the mean predictions.
    baseline_rmse = float(np.sqrt(mean_squared_error(y_eval, mean_pred)))
    logger.info("Baseline (mean-prediction) RMSE: %.4f", baseline_rmse)
    return baseline_rmse


# ---------------------------------------------------------------------------
# 5. Load champion model
# ---------------------------------------------------------------------------


def load_champion_model(model_name: str) -> tuple:
    """Load the current champion model from the MLflow Model Registry.

    This function attempts to load the model version currently aliased as
    ``@champion`` for a given registered model name. It handles two main
    scenarios:

    1. **Champion exists**: The ``@champion`` alias is found, and the
       corresponding model version is loaded. The model object and its
       version number are returned.
    2. **Bootstrap case**: No ``@champion`` alias is found (e.g., on the
       very first run of the pipeline). In this case, ``(None, None)`` is
       returned, signaling that a new model needs to be trained from scratch.

    It also gracefully handles deserialization errors (e.g., due to an
    XGBoost version mismatch between the saved model and the current
    environment), treating them as a bootstrap case to prevent pipeline crashes.

    Parameters
    ----------
    model_name : str
        The name of the registered model in MLflow (e.g.,
        ``config.MODEL_NAME``).

    Returns
    -------
    tuple[mlflow.pyfunc.PyFuncModel | None, int | None]
        A tuple containing:
        - ``pyfunc_model`` (mlflow.pyfunc.PyFuncModel | None): The loaded
          champion model wrapped as an MLflow PyFuncModel, or ``None`` if
          no champion is found or an error occurs.
        - ``version_number`` (int | None): The version number of the loaded
          champion model, or ``None``.
    """
    try:
        # Construct the MLflow URI for the champion model using its alias.
        model_uri = f"models:/{model_name}@champion"
        # Attempt to load the model using mlflow.pyfunc.load_model.
        model = mlflow.pyfunc.load_model(model_uri)

        # Retrieve the model version details using MlflowClient to get the
        # explicit version number associated with the 'champion' alias.
        client = MlflowClient()
        mv = client.get_model_version_by_alias(model_name, "champion")
        version = int(mv.version)

        logger.info("Loaded champion model v%d from %s", version, model_uri)
        return model, version
    except MlflowException:
        # This exception is raised if the 'champion' alias does not exist,
        # indicating a bootstrap scenario where no champion has been promoted yet.
        logger.info("No champion alias found for '%s' — bootstrap case.", model_name)
        return None, None
    except Exception as exc:
        # Catch any other exceptions during model loading (e.g., environment
        # mismatches, dependency issues). Treat these as a bootstrap case
        # to allow the pipeline to retrain a new model rather than failing.
        logger.warning(
            "Failed to load champion model for '%s' due to %s: %s — "
            "treating as bootstrap case (will retrain from scratch).",
            model_name,
            type(exc).__name__,
            exc,
        )
        return None, None


# ---------------------------------------------------------------------------
# 6. Register model
# ---------------------------------------------------------------------------


def register_model(run_id: str, model_name: str, tags: dict):
    """Register a model version from a completed run and apply tags.

    This function takes a model artifact logged during an MLflow run and
    registers it as a new version under a specified registered model name.
    It also applies a set of custom tags to this new model version, which
    are crucial for tracking metadata, lineage, and decision-making within
    the MLOps pipeline.

    Parameters
    ----------
    run_id : str
        The MLflow Run ID associated with the run that logged the model
        artifact. This ID is used to construct the URI to the model.
    model_name : str
        The name of the registered model in MLflow under which this new
        version will be stored (e.g., ``config.MODEL_NAME``).
    tags : dict
        A dictionary of key-value pairs to be set as version-level tags
        for the newly registered model.  Expected keys include:
        - ``role``: e.g., "candidate", "champion", "previous_champion"
        - ``trained_on_batches``: IDs of data batches used for training.
        - ``eval_batch_id``: ID of the data batch used for evaluation.
        - ``validation_status``: e.g., "pass", "fail", "warn".
        - ``decision_reason``: Human-readable explanation for promotion/rejection.

    Returns
    -------
    mlflow.entities.model_registry.ModelVersion
        The newly created ``ModelVersion`` object, containing details like
        its version number, creation timestamp, and associated run ID.

    Notes
    -----
    - The ``model_uri`` is constructed using the ``run_id`` and the default
      artifact path "model" where ``mlflow.sklearn.log_model`` stores the model.
    - ``MlflowClient`` is used to set tags because ``mlflow.register_model``
      does not directly support version-level tags during registration.
    - All tag values are converted to strings to ensure compatibility with
      MLflow's tagging system.
    """
    # Construct the URI to the model artifact within the specified MLflow run.
    model_uri = f"runs:/{run_id}/model"
    # Register the model as a new version under the given model name.
    # This creates a new ModelVersion object.
    mv = mlflow.register_model(model_uri, model_name)

    # Initialize an MlflowClient to interact with the MLflow Model Registry.
    client = MlflowClient()
    # Apply the provided tags to the newly registered model version.
    # Tags are crucial for tracking model metadata and pipeline decisions.
    for key, value in tags.items():
        client.set_model_version_tag(model_name, mv.version, key, str(value))

    logger.info(
        "Registered %s v%s  (run=%s, tags=%s)",
        model_name,
        mv.version,
        run_id,
        list(tags.keys()),
    )
    return mv


# ---------------------------------------------------------------------------
# 7. Retrain decision
# ---------------------------------------------------------------------------


def should_retrain(
    rmse_champion_on_batch: float,
    rmse_champion_on_ref: float,
    integrity_warn: bool = False,
) -> tuple[bool, str]:
    """Decide whether the champion needs retraining on the new batch.

    This function implements the logic for deciding if the current champion
    model needs to be retrained.  The decision is based on comparing the
    champion's RMSE on the **new batch** against its RMSE on the
    **reference data** (the data it was originally trained/validated on).
    A significant increase indicates that the model's performance has
    degraded — likely due to data drift — and retraining is warranted.

    The integrity warnings from the data integrity gate can influence the
    degradation threshold, making the system more sensitive when data
    quality is questionable.

    Decision logic
    --------------
    1. Compute the relative RMSE increase::

           rmse_increase_pct = (batch − ref) / ref

    2. Determine the effective threshold from
       ``config.RMSE_DEGRADATION_THRESHOLD``.  If ``integrity_warn`` is
       ``True``, the threshold is halved (×0.5) to be more conservative.

    3. If ``rmse_increase_pct > threshold`` → retrain.

    Parameters
    ----------
    rmse_champion_on_batch : float
        The RMSE of the current champion model when evaluated on the new
        incoming data batch.
    rmse_champion_on_ref : float
        The RMSE of the current champion model when evaluated on the
        reference dataset (the data it was trained or last validated on).
        This serves as the performance baseline for degradation detection.
    integrity_warn : bool
        A boolean flag indicating whether soft integrity checks (from
        ``integrity_checks.py``) detected any warnings in the new data
        batch.  If ``True``, the retraining threshold becomes stricter.

    Returns
    -------
    tuple[bool, str]
        A tuple where:
        - The first element (bool) is ``True`` if retraining is needed,
          ``False`` otherwise.
        - The second element (str) is a human-readable reason for the
          decision.
    """
    # Calculate the relative increase in RMSE from reference to batch.
    # A positive value means the champion performs worse on the new batch
    # than on the reference data it was trained/validated on.
    if rmse_champion_on_ref > 0:
        rmse_increase_pct = (
            (rmse_champion_on_batch - rmse_champion_on_ref) / rmse_champion_on_ref
        )
    else:
        rmse_increase_pct = 0.0

    # Determine the effective degradation threshold.
    # If integrity warnings are present, the threshold is halved to make the
    # model more sensitive to performance drops, as data quality is uncertain.
    threshold = config.RMSE_DEGRADATION_THRESHOLD
    if integrity_warn:
        threshold *= 0.5   # Lower the bar when data looks shaky

    # If the champion's RMSE increased beyond the threshold, retrain.
    if rmse_increase_pct > threshold:
        reason = (
            f"Champion RMSE degraded by {rmse_increase_pct:.2%} "
            f"(batch={rmse_champion_on_batch:.4f} vs "
            f"ref={rmse_champion_on_ref:.4f}, "
            f"threshold={threshold:.2%}) — retraining required."
        )
        logger.warning(reason)
        return True, reason

    # If the increase is within the acceptable range, no retraining is needed.
    reason = (
        f"Champion RMSE within acceptable range "
        f"(batch={rmse_champion_on_batch:.4f} vs "
        f"ref={rmse_champion_on_ref:.4f}, "
        f"degradation={rmse_increase_pct:.2%}, "
        f"threshold={threshold:.2%}) — no retrain."
    )
    logger.info(reason)
    return False, reason


# ---------------------------------------------------------------------------
# 8. Promotion decision
# ---------------------------------------------------------------------------


def should_promote(
    rmse_candidate: float,
    rmse_champion: float,
    rmse_candidate_on_ref: float,
    rmse_champion_on_ref: float,
    integrity_warn: bool,
) -> tuple[bool, str]:
    """Apply promotion criteria P1–P4 to decide if the candidate replaces the champion.

    This function evaluates a newly trained candidate model against the current
    champion model based on a set of predefined criteria (P1-P4). A candidate
    must satisfy all criteria to be promoted to champion.

    Criteria
    --------
    **P1 — Validity**: Ensures that all input RMSE metrics (candidate on new
    batch, champion on new batch, candidate on reference, champion on reference)
    are valid finite numbers (not ``None`` or ``NaN``). If any metric is invalid,
    promotion is blocked.

    **P2 — Improvement**: The candidate model must demonstrate a significant
    improvement over the champion model on the *new data batch*. Specifically,
    its RMSE must be lower than the champion's RMSE by at least
    ``config.MIN_IMPROVEMENT`` (e.g., 1 %):

        ``rmse_candidate < rmse_champion * (1 - config.MIN_IMPROVEMENT)``

    **P3 — Stability**: The candidate model must not show significant regression
    compared to the champion model on the *reference (training-era) dataset*.
    This prevents promoting models that perform well on new data but have
    "forgotten" how to predict on older, representative data. The candidate's
    RMSE on the reference set must be within a tolerance of the champion's:

        ``rmse_candidate_on_ref <= rmse_champion_on_ref * (1 + config.REFERENCE_REGRESSION_TOLERANCE)``

    **P4 — Integrity**: If ``integrity_warn`` is ``True`` (meaning soft data
    integrity checks flagged warnings for the new data batch), promotion is
    blocked. This is a safety mechanism to prevent promoting a model based
    on potentially shaky or anomalous input data, ensuring data quality is
    confirmed before a model swap.

    Parameters
    ----------
    rmse_candidate : float
        The RMSE of the candidate model on the new evaluation batch.
    rmse_champion : float
        The RMSE of the current champion model on the same new evaluation batch.
    rmse_candidate_on_ref : float
        The RMSE of the candidate model when evaluated on the reference
        (training-era) holdout dataset.
    rmse_champion_on_ref : float
        The RMSE of the current champion model when evaluated on the reference
        (training-era) holdout dataset.
    integrity_warn : bool
        A boolean flag indicating whether soft integrity checks flagged warnings
        for the new data batch. If ``True``, promotion is blocked.

    Returns
    -------
    tuple[bool, str]
        A tuple where:
        - The first element (bool) is ``True`` if the candidate should be
          promoted, ``False`` otherwise.
        - The second element (str) is a human-readable reason for the decision.
    """
    # P1 — Validity Check: Ensure all RMSE metrics are valid numbers.
    # If any metric is None or NaN, it indicates a problem in evaluation or data,
    # and promotion should be blocked to prevent deploying an unreliable model.
    values = [rmse_candidate, rmse_champion, rmse_candidate_on_ref, rmse_champion_on_ref]
    if any(v is None for v in values) or any(math.isnan(v) for v in values):
        reason = "P1 FAIL: one or more metrics are None or NaN — promotion blocked."
        logger.warning(reason)
        return False, reason

    # P4 — Integrity Check: Block promotion if data quality warnings are present.
    # This is checked early to prioritize data quality. If the incoming data
    # is questionable, promoting a new model based on it is risky.
    if integrity_warn:
        reason = (
            "P4 FAIL: integrity warnings present — promotion blocked "
            "until data quality is confirmed."
        )
        logger.warning(reason)
        return False, reason

    # P2 — Improvement Check: Candidate must be significantly better on new data.
    # The candidate's RMSE must be lower than the champion's by at least
    # config.MIN_IMPROVEMENT (e.g., 1%). This ensures that only truly better
    # models are promoted.
    improvement_limit = rmse_champion * (1.0 - config.MIN_IMPROVEMENT)
    if rmse_candidate >= improvement_limit:
        reason = (
            f"P2 FAIL: candidate RMSE ({rmse_candidate:.4f}) does not beat "
            f"champion ({rmse_champion:.4f}) by >= {config.MIN_IMPROVEMENT:.2%} "
            f"(need < {improvement_limit:.4f})."
        )
        logger.info(reason)
        return False, reason

    # P3 — Stability Check: Candidate must not regress too much on reference data.
    # This prevents models that overfit to recent data or have forgotten older
    # patterns. The candidate's RMSE on the reference set should not exceed
    # the champion's by more than config.REFERENCE_REGRESSION_TOLERANCE.
    regression_limit = rmse_champion_on_ref * (1.0 + config.REFERENCE_REGRESSION_TOLERANCE)
    if rmse_candidate_on_ref > regression_limit:
        reason = (
            f"P3 FAIL: candidate reference RMSE ({rmse_candidate_on_ref:.4f}) "
            f"regresses beyond tolerance ({regression_limit:.4f}, "
            f"tol={config.REFERENCE_REGRESSION_TOLERANCE:.2%})."
        )
        logger.info(reason)
        return False, reason

    # If all criteria are met, the candidate is deemed worthy of promotion.
    reason = (
        f"All promotion criteria passed — candidate RMSE {rmse_candidate:.4f} "
        f"beats champion {rmse_champion:.4f} (improvement limit {improvement_limit:.4f}), "
        f"reference RMSE {rmse_candidate_on_ref:.4f} within tolerance "
        f"({regression_limit:.4f})."
    )
    logger.info(reason)
    return True, reason


# ---------------------------------------------------------------------------
# 9. Promote champion
# ---------------------------------------------------------------------------


def promote_champion(
    model_name: str,
    new_version: int,
    old_version: int | None,
    reason: str,
) -> None:
    """Move the ``@champion`` alias to a new model version.

    This function performs the critical operation of promoting a new model
    version to become the ``@champion`` in the MLflow Model Registry. This
    involves updating aliases and setting appropriate tags to maintain a clear
    history of model roles and promotion events.

    Parameters
    ----------
    model_name : str
        The name of the registered model in MLflow (e.g.,
        ``config.MODEL_NAME``).
    new_version : int
        The version number of the candidate model that is to be promoted
        to champion.
    old_version : int | None
        The version number of the model that was previously the champion.
        This will be ``None`` in the bootstrap case (first promotion).
    reason : str
        A human-readable string explaining why the new model version is being
        promoted. This reason is stored as a tag for auditability.

    Notes
    -----
    - The ``@champion`` alias is atomically moved to ``new_version``.
    - The ``new_version`` is tagged with ``role: champion``, ``promoted_at``,
      and ``promotion_reason``.
    - If an ``old_version`` existed, it is demoted by tagging it with
      ``role: previous_champion`` and ``demoted_at``.
    - All interactions with the MLflow Model Registry are done via ``MlflowClient``.
    """
    client = MlflowClient()
    now_iso = datetime.now(timezone.utc).isoformat()

    # Step 1: Point the @champion alias to the new model version.
    # This is the core promotion step, making the new model the active production model.
    client.set_registered_model_alias(model_name, "champion", str(new_version))

    # Step 2: Tag the newly promoted champion model version with its role and promotion details.
    client.set_model_version_tag(model_name, str(new_version), "role", "champion")
    client.set_model_version_tag(model_name, str(new_version), "promoted_at", now_iso)
    client.set_model_version_tag(model_name, str(new_version), "promotion_reason", reason)

    logger.info("Promoted %s v%d to @champion.", model_name, new_version)

    # Step 3: If an old champion existed, demote it by updating its tags.
    # This maintains a clear history of which model was champion and when it was replaced.
    if old_version is not None:
        client.set_model_version_tag(
            model_name, str(old_version), "role", "previous_champion"
        )
        client.set_model_version_tag(
            model_name, str(old_version), "demoted_at", now_iso
        )
        logger.info("Demoted %s v%d to previous_champion.", model_name, old_version)


# ---------------------------------------------------------------------------
# 10. Log model to MLflow
# ---------------------------------------------------------------------------


def log_model_to_mlflow(model, X_train, model_name: str) -> str:
    """Log the trained model as an artifact in the active MLflow run.

    This function logs a trained model to the active MLflow run as an artifact.
    It leverages ``mlflow.sklearn.log_model``, which is compatible with
    XGBoost models (as XGBoost implements the scikit-learn API).
    Crucially, it also attaches an ``input_example`` to the logged model,
    enabling MLflow to automatically infer the model's signature.

    Parameters
    ----------
    model : fitted estimator
        The trained model object (e.g., an ``xgb.XGBRegressor`` instance).
        It must be a fitted estimator that can be saved by MLflow's sklearn
        flavor.
    X_train : pd.DataFrame
        A DataFrame containing the training features. A small subset (e.g.,
        ``X_train.head(5)``) is used as an ``input_example`` to help MLflow
        infer the model's input schema and data types. This is vital for
        downstream model serving and deployment.
    model_name : str
        The name of the registered model. By passing this to
        ``registered_model_name``, the model is not only logged as an artifact
        but also automatically registered as a new version in the MLflow Model
        Registry in a single API call.

    Returns
    -------
    str
        The ``run_id`` of the active MLflow run where the model was logged.

    Notes
    -----
    - The ``name="model"`` specifies the subdirectory within the
      MLflow run's artifact URI where the model will be saved.
    - Providing an ``input_example`` is a best practice for MLflow models,
      as it allows for automatic signature inference, which improves model
      governance and deployment reliability.
    - The ``registered_model_name`` parameter streamlines the process of
      logging and registering a model, ensuring it appears in the Model
      Registry immediately after the run completes.
    """
    # Log the model using MLflow's sklearn flavor. This saves the model
    # artifact and registers it in the MLflow Model Registry.
    # Location IDs are already int (converted in feature_engineering).
    example = X_train.head(5).copy()
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        name="model",
        input_example=example,
        registered_model_name=model_name,
    )
    # Retrieve the run_id of the active MLflow run.
    run_id = mlflow.active_run().info.run_id
    logger.info(
        "Model logged — run_id=%s  model_uri=%s",
        run_id,
        model_info.model_uri,
    )
    return run_id
