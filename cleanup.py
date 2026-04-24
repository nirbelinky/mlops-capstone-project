"""
cleanup.py — Reset the capstone project to a clean slate.

Purpose
-------
During development you will run the pipeline many times.  Each run creates
Metaflow state (the ``.metaflow/`` directory), local MLflow run artifacts
(the ``mlruns/`` directory), Python bytecode caches (``__pycache__/``), and
rows inside the shared MLflow SQLite database (experiments, runs, registered
models, model versions, and aliases).

This script removes **all** of that generated state so you can start fresh,
while leaving source code, documentation, data files, and the database file
itself intact.  It is safe to run repeatedly — if an artifact has already
been deleted, the script simply skips it.

What gets deleted
-----------------
* ``.metaflow/``          — Metaflow run metadata and artifacts
* ``mlruns/``             — Local MLflow artifact store
* ``__pycache__/``        — Python bytecode caches (this dir + subdirs)
* MLflow DB contents      — Registered model aliases, model versions,
                            the registered model entry, and all runs
                            belonging to the ``green_taxi_capstone``
                            experiment inside ``../mlflow_tracking/mlflow.db``

What is preserved
-----------------
* Source code (``.py``, ``.md``)
* Data files (``data/*.parquet``)
* ``.gitignore``
* The MLflow database **file** (only its contents are cleaned)

Usage
-----
::

    python cleanup.py          # interactive — asks for confirmation
    python cleanup.py --force  # non-interactive — skips the prompt

Implementation notes
--------------------
* Directory removal uses ``shutil.rmtree(path, ignore_errors=True)`` so
  missing directories never cause failures.
* MLflow database cleanup uses ``mlflow.MlflowClient()`` and wraps every
  call in try/except so the script succeeds even on a first run when the
  experiment or model does not yet exist.
* The script imports ``MLFLOW_TRACKING_URI``, ``MODEL_NAME``, and
  ``MLFLOW_EXPERIMENT_NAME`` from :mod:`config` to stay in sync with the
  rest of the project.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

import mlflow
from mlflow.exceptions import MlflowException

from config import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI, MODEL_NAME

# ---------------------------------------------------------------------------
# Directories to remove (relative to this script's location)
# ---------------------------------------------------------------------------
# Each entry is a tuple of (human-readable label, directory path).
_DIRS_TO_REMOVE: list[tuple[str, str]] = [
    ("Metaflow state", ".metaflow"),
    ("MLflow local artifacts", "mlruns"),
]


# ===========================================================================
# Helper functions
# ===========================================================================

def _script_dir() -> str:
    """Return the directory that contains this script (absolute path)."""
    return os.path.dirname(os.path.abspath(__file__))


def _remove_directory(label: str, path: str) -> str:
    """
    Remove a directory tree and return a one-line status message.

    Parameters
    ----------
    label : str
        Human-readable name shown in the summary (e.g. "Metaflow state").
    path : str
        Absolute or relative path to the directory.

    Returns
    -------
    str
        A status line such as ``"✓ Metaflow state — removed"`` or
        ``"– Metaflow state — not found (skipped)"``.
    """
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
        return f"  ✓ {label} — removed"
    return f"  – {label} — not found (skipped)"


def _remove_pycache(base_dir: str) -> str:
    """
    Walk *base_dir* and remove every ``__pycache__/`` directory found.

    Returns
    -------
    str
        A status line reporting how many cache directories were removed.
    """
    count = 0
    for root, dirs, _files in os.walk(base_dir):
        if "__pycache__" in dirs:
            shutil.rmtree(os.path.join(root, "__pycache__"), ignore_errors=True)
            count += 1
    if count:
        return f"  ✓ Python caches — removed {count} __pycache__ dir(s)"
    return "  – Python caches — none found (skipped)"


def _clean_mlflow_db() -> list[str]:
    """
    Connect to the MLflow tracking store and delete capstone-related data.

    The function performs the following steps (each wrapped in its own
    try/except so that earlier failures do not prevent later cleanup):

    1. Delete all registered-model **aliases** for ``MODEL_NAME``.
    2. Delete all **model versions** for ``MODEL_NAME``.
    3. Delete the **registered model** entry for ``MODEL_NAME``.
    4. Delete all **runs** in the ``MLFLOW_EXPERIMENT_NAME`` experiment.

    Returns
    -------
    list[str]
        Status lines for each sub-step.
    """
    messages: list[str] = []

    # Point the client at the shared SQLite database.
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.MlflowClient()

    # ---- 1. Delete registered-model aliases --------------------------------
    try:
        model_details = client.get_registered_model(MODEL_NAME)
        aliases_deleted = 0
        # Iterate over all versions and collect their aliases.
        for mv in client.search_model_versions(f"name='{MODEL_NAME}'"):
            for alias in mv.aliases:
                try:
                    client.delete_registered_model_alias(MODEL_NAME, alias)
                    aliases_deleted += 1
                except MlflowException:
                    pass
        if aliases_deleted:
            messages.append(f"  ✓ Model aliases — deleted {aliases_deleted} alias(es)")
        else:
            messages.append("  – Model aliases — none found (skipped)")
    except MlflowException:
        messages.append("  – Model aliases — model not registered (skipped)")

    # ---- 2. Delete model versions ------------------------------------------
    try:
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        versions_deleted = 0
        for mv in versions:
            try:
                client.delete_model_version(MODEL_NAME, mv.version)
                versions_deleted += 1
            except MlflowException:
                pass
        if versions_deleted:
            messages.append(f"  ✓ Model versions — deleted {versions_deleted} version(s)")
        else:
            messages.append("  – Model versions — none found (skipped)")
    except MlflowException:
        messages.append("  – Model versions — model not registered (skipped)")

    # ---- 3. Delete the registered model itself -----------------------------
    try:
        client.delete_registered_model(MODEL_NAME)
        messages.append(f"  ✓ Registered model '{MODEL_NAME}' — deleted")
    except MlflowException:
        messages.append(f"  – Registered model '{MODEL_NAME}' — not found (skipped)")

    # ---- 4. Delete all runs in the experiment ------------------------------
    try:
        experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
        if experiment is None:
            messages.append(f"  – Experiment '{MLFLOW_EXPERIMENT_NAME}' — not found (skipped)")
        else:
            # If the experiment was soft-deleted (e.g. via the MLflow UI),
            # restore it first so we can search its runs and so the next
            # pipeline run can use it without errors.
            if experiment.lifecycle_stage == "deleted":
                client.restore_experiment(experiment.experiment_id)
                messages.append("  ✓ Experiment — restored soft-deleted experiment")

            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                max_results=5000,
            )
            runs_deleted = 0
            for run in runs:
                try:
                    client.delete_run(run.info.run_id)
                    runs_deleted += 1
                except MlflowException:
                    pass
            if runs_deleted:
                messages.append(
                    f"  ✓ Experiment runs — deleted {runs_deleted} run(s) "
                    f"from '{MLFLOW_EXPERIMENT_NAME}'"
                )
            else:
                messages.append(
                    f"  – Experiment runs — no runs in '{MLFLOW_EXPERIMENT_NAME}' (skipped)"
                )
    except MlflowException as exc:
        messages.append(f"  – Experiment runs — error ({exc}) (skipped)")

    return messages


# ===========================================================================
# Main entry point
# ===========================================================================

def main() -> None:
    """
    Parse CLI arguments, optionally prompt for confirmation, then run all
    cleanup steps and print a summary.
    """
    parser = argparse.ArgumentParser(
        description="Reset the capstone project by removing generated artifacts.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip the confirmation prompt.",
    )
    args = parser.parse_args()

    # ---- Confirmation prompt -----------------------------------------------
    if not args.force:
        answer = input(
            "This will delete all MLflow runs, models, and Metaflow state. "
            "Continue? [y/N] "
        )
        if answer.strip().lower() != "y":
            print("Aborted.")
            sys.exit(0)

    base = _script_dir()
    summary: list[str] = []

    # ---- 1. Remove directories ---------------------------------------------
    print("\n🧹  Cleaning up capstone project artifacts …\n")
    print("Directories:")
    for label, rel_path in _DIRS_TO_REMOVE:
        msg = _remove_directory(label, os.path.join(base, rel_path))
        summary.append(msg)
        print(msg)

    # ---- 2. Remove __pycache__ dirs ----------------------------------------
    msg = _remove_pycache(base)
    summary.append(msg)
    print(msg)

    # ---- 3. Clean MLflow database ------------------------------------------
    print("\nMLflow database:")
    try:
        db_messages = _clean_mlflow_db()
    except Exception as exc:  # noqa: BLE001
        db_messages = [f"  ⚠ MLflow cleanup failed: {exc}"]
    summary.extend(db_messages)
    for msg in db_messages:
        print(msg)

    # ---- Summary -----------------------------------------------------------
    print("\n✅  Cleanup complete.\n")


if __name__ == "__main__":
    main()
