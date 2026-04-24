# MLOps Capstone: Green Taxi Tip Prediction — Monitoring & Retraining Pipeline

A manually-run, end-to-end MLOps pipeline that monitors an XGBoost tip-prediction
model for NYC Green Taxi trips, decides whether to retrain, and automatically
promotes a better candidate to champion — all orchestrated by **Metaflow** with
full audit logging in **MLflow**.

Each run ingests a new monthly batch, applies hard + soft (NannyML) integrity
checks, evaluates the current champion, and — when performance degrades —
retrains a candidate, validates it against the champion on both batch and
reference data, and flips the `@champion` alias in the MLflow Model Registry.

---

## Project Structure

| File | Description |
|---|---|
| [`flow.py`](flow.py) | Main Metaflow flow — Steps A–G (load → integrity → features → champion → evaluate → retrain → promote) |
| [`config.py`](config.py) | Central configuration: column names, feature lists, thresholds, MLflow settings |
| [`feature_engineering.py`](feature_engineering.py) | Deterministic feature transforms (time features, log-transforms, credit-card filter) |
| [`integrity_checks.py`](integrity_checks.py) | Hard rules (schema, ranges) + NannyML soft checks (missingness, unseen categoricals) |
| [`model_utils.py`](model_utils.py) | Training, evaluation, registry helpers, promotion logic |
| [`decision_logger.py`](decision_logger.py) | Structured decision audit trail (`decision.json` artifacts) |
| [`download_data.py`](download_data.py) | Downloads reference & batch parquet files from the NYC TLC site |
| [`design_doc.md`](design_doc.md) | Full design specification for the capstone |
| `data/` | Parquet data files (created by `download_data.py`) |

---

## Setup

```bash
# 1. Create / activate the conda environment
conda env create -f environment.yml   # first time only
conda activate 22971-mlflow

# 2. Install additional dependencies (Metaflow + NannyML)
pip install metaflow nannyml

# 3. Download data
python download_data.py
#    Downloads:
#      data/reference_2024-01.parquet   (reference / baseline)
#      data/batch_2024-02.parquet       (similar season — no drift expected)
#      data/batch_2024-06.parquet       (summer — drift / degradation likely)

# 4. Start the MLflow UI (in a separate terminal)
mlflow ui --backend-store-uri sqlite:///mlflow_tracking/mlflow.db --port 5000
```

Open <http://127.0.0.1:5000> to browse experiments.

---

## How to Run

### Bootstrap run (first time — creates the initial champion)

```bash
python flow.py run \
  --reference-path data/reference_2024-01.parquet \
  --batch-path data/batch_2024-02.parquet
```

### Subsequent run (with a different batch)

```bash
python flow.py run \
  --reference-path data/reference_2024-01.parquet \
  --batch-path data/batch_2024-06.parquet
```

### Resume after failure

```bash
python flow.py resume
```

Metaflow resumes from the last failed step; previously completed steps are
**not** re-executed.

---

## Required Demo Runs

The design doc requires **three separate runs** to demonstrate the pipeline:

### Run 1 — Baseline (no retrain, no promotion)

Run with a batch that is close to the reference distribution (e.g.
`batch_2024-02.parquet`). The champion performs well, so no retrain is
triggered and no promotion occurs.

**Expected evidence in MLflow:**

- `retrain_recommended = false`
- `promotion_recommended` tag absent (step never reached)
- `decision.json` explaining the outcome

### Run 2 — Retrain + Promotion (degradation triggers retrain)

Run with a batch that exhibits drift (e.g. `batch_2024-06.parquet`). The
champion's RMSE degrades beyond the threshold, a candidate is trained on
combined data, and — if it beats the champion — it is promoted.

**Expected evidence in MLflow:**

- `retrain_recommended = true`
- `promotion_recommended = true`
- New model version in the Model Registry with `@champion` alias
- `decision.json` artifacts for integrity, retrain, and promotion decisions

### Run 3 — Failure + Resume (workflow robustness)

Inject an exception in a mid-flow step (e.g. add `raise RuntimeError("test")`
at the top of the `retrain` step), run the flow, then remove the exception and
resume:

```bash
# 1. Edit flow.py — add `raise RuntimeError("injected")` in the retrain step
# 2. Run the flow (it will fail at retrain)
python flow.py run \
  --reference-path data/reference_2024-01.parquet \
  --batch-path data/batch_2024-06.parquet

# 3. Remove the injected exception from flow.py
# 4. Resume from the failed step
python flow.py resume
```

**Expected evidence:**

- The flow resumes from `retrain`, not from `start`
- Previously completed steps are not re-executed
- Final decisions and artifacts reflect the successful resumed execution

---

## Where to Look in the MLflow UI

| Item | Value |
|---|---|
| **Experiment name** | `green_taxi_capstone` |
| **Key metrics** | `rmse_champion`, `rmse_baseline`, `rmse_increase_pct`, `rmse_candidate` |
| **Key artifacts** | `integrity_decision.json`, `retrain_decision.json`, `promotion_decision.json`, `all_decisions.json`, `feature_spec.json`, `predictions.parquet` |
| **Key tags** | `retrain_recommended`, `promotion_recommended`, `integrity_hard_pass`, `integrity_soft_warnings` |
| **Model Registry** | Model name: `green_taxi_tip_model` — look for the `@champion` alias |

---

## Configuration

All tuneable thresholds live in [`config.py`](config.py):

| Constant | Default | Purpose |
|---|---|---|
| `RMSE_DEGRADATION_THRESHOLD` | `0.10` (10 %) | RMSE increase that triggers retraining |
| `MIN_IMPROVEMENT` | `0.01` (1 %) | Candidate must beat champion by this margin to be promoted |
| `REFERENCE_REGRESSION_TOLERANCE` | `0.05` (5 %) | Candidate must not regress on reference data by more than this |
| `MISSINGNESS_SPIKE_THRESHOLD` | `0.10` (10 %) | NannyML soft-gate: missingness increase vs reference |
| `UNSEEN_CATEGORICAL_THRESHOLD` | `0.05` (5 %) | NannyML soft-gate: unseen category rate vs reference |
| `MAX_TRIP_DISTANCE` | `200` miles | Hard integrity: maximum allowed trip distance |
| `MAX_FARE_AMOUNT` | `500` dollars | Hard integrity: maximum allowed fare |
| `MAX_TIP_AMOUNT` | `200` dollars | Hard integrity: maximum allowed tip |

Adjust these values to experiment with different sensitivity levels for the
integrity, retrain, and promotion gates.
