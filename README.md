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
| [`watcher.py`](watcher.py) | Polling script — watches `data/inbox/` for new batches, runs the flow, moves processed files to `data/reference/` (Stretch A) |
| [`design_doc.md`](design_doc.md) | Full design specification for the capstone |
| `data/reference/` | Reference parquet files — the expanding baseline window |
| `data/inbox/` | New batch files waiting to be processed |

---

## Setup

```bash
# 1. Create / activate the conda environment
conda env create -f environment.yml   # first time only
conda activate 22971-mlflow-nir-belinky-01

# 2. Install additional dependencies (Metaflow + NannyML)
pip install metaflow nannyml

# 3. Download data
python download_data.py
#    Downloads:
#      data/reference/2024-01.parquet   (reference / baseline — winter)
#      data/reference/2024-02.parquet   (reference / baseline — winter)
#      data/inbox/2024-03.parquet       (similar season — no drift expected)
#      data/inbox/2024-06.parquet       (summer — drift / degradation likely)

# 4. Start the MLflow UI (in a separate terminal)
mlflow ui --backend-store-uri sqlite:///mlflow_tracking/mlflow.db --port 5000
```

Open <http://127.0.0.1:5000> to browse experiments.

---

## How to Run

### Bootstrap run (first time — creates the initial champion)

```bash
python flow.py run \
  --reference-path data/reference \
  --batch-path data/inbox/2024-03.parquet
```

### Subsequent run (with a different batch)

```bash
python flow.py run \
  --reference-path data/reference \
  --batch-path data/inbox/2024-06.parquet
```

### Resume after failure

```bash
python flow.py resume
```

Metaflow resumes from the last failed step; previously completed steps are
**not** re-executed.

---

## Required Demo Runs

The design doc requires **three separate runs** to demonstrate the pipeline.
Below are two ways to run them: manually (step by step) or using the watcher
for automation.

### Run 1 — Baseline (no retrain, no promotion)

Process a batch from the same season as the reference (March 2024). The
champion performs well (~6% degradation, below the 10% threshold), so no
retrain is triggered and no promotion occurs.

**Expected evidence in MLflow:**

- `retrain_recommended = false`
- `promotion_recommended` tag absent (step never reached)
- `decision.json` explaining the outcome

### Run 2 — Retrain + Promotion (degradation triggers retrain)

Process a summer batch (June 2024) that exhibits seasonal drift. The
champion's RMSE degrades beyond the 10% threshold, a candidate is trained on
combined data, and — if it beats the champion — it is promoted.

**Expected evidence in MLflow:**

- `retrain_recommended = true`
- `promotion_recommended = true`
- New model version in the Model Registry with `@champion` alias
- `decision.json` artifacts for integrity, retrain, and promotion decisions

### Run 3 — Failure + Resume (workflow robustness)

`flow.py` has a built-in `--simulate-failure` parameter that raises a
`RuntimeError` in the `retrain` step — no source editing needed. Pass the flag
on the first run to trigger the failure, then resume without it (the parameter
defaults to `False`, so the resumed run skips the simulated failure
automatically).

**Expected evidence:**

- The flow resumes from `retrain`, not from `start`
- Previously completed steps are not re-executed
- Final decisions and artifacts reflect the successful resumed execution

### Option A — Manual demo (step by step)

```bash
# 0. Fresh start
python cleanup.py
python download_data.py

# Run 1 — Bootstrap + baseline (March batch, no retrain)
python flow.py run \
  --reference-path data/reference \
  --batch-path data/inbox/2024-03.parquet

# Run 2 — Same batch again (non-bootstrap, confirms no retrain at ~6%)
python flow.py run \
  --reference-path data/reference \
  --batch-path data/inbox/2024-03.parquet

# Move March into reference so June sees an expanded training window
mv data/inbox/2024-03.parquet data/reference/

# Run 3 — Summer batch (drift → retrain + promote)
python flow.py run \
  --reference-path data/reference \
  --batch-path data/inbox/2024-06.parquet

# Run 4 — Failure + Resume (no reset needed, June is still in inbox)
python flow.py run \
  --reference-path data/reference \
  --batch-path data/inbox/2024-06.parquet \
  --simulate-failure True
python flow.py resume
```

> **Note:** The manual `mv` step between Run 2 and Run 3 promotes the
> processed batch into the reference directory. This is what the watcher
> automates (see Option B). In manual mode, `flow.py` never moves files,
> so June stays in `data/inbox/` and can be reused for Run 4.

### Option B — Automated demo (using the watcher)

```bash
# 0. Fresh start
python cleanup.py
python download_data.py

# Run 1 — Bootstrap (creates initial champion)
python flow.py run \
  --reference-path data/reference \
  --batch-path data/inbox/2024-03.parquet

# Runs 2 + 3 — Watcher processes all inbox files automatically:
#   • March → no retrain (6% < 10%) → moved to data/reference/
#   • June  → retrain + promote     → moved to data/reference/
python watcher.py

# Run 4 — Failure + Resume (no reset needed)
# Copy June back to inbox (the watcher moved it to reference)
cp data/reference/2024-06.parquet data/inbox/
python flow.py run \
  --reference-path data/reference \
  --batch-path data/inbox/2024-06.parquet \
  --simulate-failure True
python flow.py resume
```

> **Note:** The watcher automatically moves successfully processed batches
> from `data/inbox/` to `data/reference/`, expanding the reference window
> for future runs. For Run 4, we copy June back to inbox since the watcher
> already moved it.

---

## Automated Batch Processing (Stretch A)

The [`watcher.py`](watcher.py) script automates the flow by watching `data/inbox/`
for new `.parquet` files. For each file found, it runs the pipeline and — on
success — moves the batch into `data/reference/` so it becomes part of the
expanding reference window for future runs.

### One-shot mode (process all pending files and exit)

```bash
python watcher.py
```

### Continuous polling mode

```bash
# Check every 60 seconds for new files
python watcher.py --poll-interval 60
```

### With cron (e.g. every 15 minutes)

```bash
*/15 * * * * cd /path/to/project && python watcher.py >> watcher.log 2>&1
```

### Dry-run mode (preview without executing)

```bash
python watcher.py --dry-run
```

### Data lifecycle

```
data/inbox/2024-06.parquet          ← new batch arrives here
     │
     ▼
flow.py run --reference-path data/reference
            --batch-path data/inbox/2024-06.parquet
     │
     ├─ success → move to data/reference/2024-06.parquet
     │            (becomes part of future reference)
     │
     └─ failure → leave in data/inbox/
                  (will be retried on next watcher run)
```

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
| `RMSE_DEGRADATION_THRESHOLD` | `0.10` (10 %) | RMSE increase (vs stored ref_rmse) that triggers retraining |
| `MIN_IMPROVEMENT` | `0.01` (1 %) | Candidate must beat champion by this margin to be promoted |
| `REFERENCE_REGRESSION_TOLERANCE` | `0.05` (5 %) | Candidate must not regress on reference data by more than this |
| `MISSINGNESS_SPIKE_THRESHOLD` | `0.10` (10 %) | NannyML soft-gate: missingness increase vs reference |
| `UNSEEN_CATEGORICAL_THRESHOLD` | `0.05` (5 %) | NannyML soft-gate: unseen category rate vs reference |
| `MAX_TRIP_DISTANCE` | `200` miles | Hard integrity: maximum allowed trip distance |
| `MAX_FARE_AMOUNT` | `500` dollars | Hard integrity: maximum allowed fare |
| `MAX_TIP_AMOUNT` | `200` dollars | Hard integrity: maximum allowed tip |

Adjust these values to experiment with different sensitivity levels for the
integrity, retrain, and promotion gates.
