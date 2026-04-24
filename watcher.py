"""
watcher.py — Polling script that watches for new batch files and triggers the flow.

Purpose
-------
This script implements **Stretch A (Automation / event triggering)** from the
capstone design doc.  It watches the ``data/inbox/`` directory for new Parquet
files and, for each one found, invokes the Metaflow pipeline.  After a
successful run the batch file is moved into ``data/reference/`` so it becomes
part of the expanding reference window for future runs.

The Metaflow flow itself remains pure and manual — this script adds automation
*around* it, not inside it, as recommended by the design doc.

Data lifecycle
--------------
::

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

How to run
----------
One-shot (process all pending files and exit)::

    python watcher.py

Continuous polling (check every N seconds)::

    python watcher.py --poll-interval 60

With cron (e.g. every 15 minutes)::

    */15 * * * * cd /path/to/project && python watcher.py >> watcher.log 2>&1

Options::

    --inbox          Directory to watch for new .parquet files (default: data/inbox)
    --reference      Directory where processed batches are moved (default: data/reference)
    --poll-interval  Seconds between checks; 0 = run once and exit (default: 0)
    --dry-run        Show what would happen without actually running the flow
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-12s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger("watcher")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Resolve paths relative to this script's location so the watcher works
# regardless of the current working directory.
PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_INBOX = PROJECT_DIR / "data" / "inbox"
DEFAULT_REFERENCE = PROJECT_DIR / "data" / "reference"
FLOW_SCRIPT = PROJECT_DIR / "flow.py"


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def find_pending_batches(inbox: Path) -> list[Path]:
    """Return a sorted list of .parquet files in the inbox directory.

    Files are sorted alphabetically (which, given YYYY-MM naming, also
    means chronological order).  This ensures batches are processed in
    the order they were generated.

    Args:
        inbox: Path to the inbox directory.

    Returns:
        Sorted list of Path objects for each ``.parquet`` file found.
    """
    if not inbox.is_dir():
        return []
    return sorted(inbox.glob("*.parquet"))


def run_flow(batch_path: Path, reference_dir: Path) -> bool:
    """Invoke the Metaflow pipeline for a single batch file.

    Args:
        batch_path:    Path to the batch ``.parquet`` file.
        reference_dir: Path to the reference directory (read by the flow
                       via ``pd.read_parquet``).

    Returns:
        ``True`` if the flow completed successfully (exit code 0),
        ``False`` otherwise.
    """
    cmd = [
        sys.executable,
        str(FLOW_SCRIPT),
        "run",
        "--reference-path", str(reference_dir),
        "--batch-path", str(batch_path),
    ]
    logger.info("▶  Running: %s", " ".join(cmd))

    result = subprocess.run(cmd, cwd=str(PROJECT_DIR))
    return result.returncode == 0


def promote_batch(batch_path: Path, reference_dir: Path) -> Path:
    """Move a successfully processed batch into the reference directory.

    After the flow completes successfully, the batch data is considered
    "known good" and becomes part of the expanding reference window for
    future runs.

    Args:
        batch_path:    Path to the batch file in ``inbox/``.
        reference_dir: Path to the ``reference/`` directory.

    Returns:
        The new path of the file inside ``reference/``.
    """
    dest = reference_dir / batch_path.name
    shutil.move(str(batch_path), str(dest))
    logger.info("📦  Moved %s → %s", batch_path.name, dest)
    return dest


def process_inbox(
    inbox: Path,
    reference_dir: Path,
    dry_run: bool = False,
) -> dict[str, list[str]]:
    """Process all pending batch files in the inbox.

    For each ``.parquet`` file found:
    1. Run the Metaflow flow with the file as ``--batch-path``.
    2. On success, move the file to ``reference/``.
    3. On failure, leave the file in ``inbox/`` for retry.

    Args:
        inbox:         Path to the inbox directory.
        reference_dir: Path to the reference directory.
        dry_run:       If ``True``, log what would happen but don't
                       actually run the flow or move files.

    Returns:
        A dict with keys ``"succeeded"`` and ``"failed"``, each
        containing a list of filenames.
    """
    pending = find_pending_batches(inbox)
    results: dict[str, list[str]] = {"succeeded": [], "failed": []}

    if not pending:
        logger.info("📭  No new batches in %s", inbox)
        return results

    logger.info("📬  Found %d pending batch(es) in %s", len(pending), inbox)

    for batch_path in pending:
        logger.info("")
        logger.info("=" * 60)
        logger.info("Processing: %s", batch_path.name)
        logger.info("=" * 60)

        if dry_run:
            logger.info("🔍  [dry-run] Would run flow for %s", batch_path.name)
            logger.info("🔍  [dry-run] Would move to %s/", reference_dir)
            results["succeeded"].append(batch_path.name)
            continue

        success = run_flow(batch_path, reference_dir)

        if success:
            promote_batch(batch_path, reference_dir)
            results["succeeded"].append(batch_path.name)
            logger.info("✅  %s processed successfully", batch_path.name)
        else:
            results["failed"].append(batch_path.name)
            logger.warning(
                "❌  Flow failed for %s — leaving in inbox for retry",
                batch_path.name,
            )

    # Print summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("Watcher Summary")
    logger.info("=" * 60)
    logger.info("  Succeeded: %d", len(results["succeeded"]))
    logger.info("  Failed:    %d", len(results["failed"]))
    for name in results["succeeded"]:
        logger.info("    ✅ %s", name)
    for name in results["failed"]:
        logger.info("    ❌ %s", name)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Watch data/inbox/ for new batches and trigger the MLOps flow.",
    )
    parser.add_argument(
        "--inbox",
        type=Path,
        default=DEFAULT_INBOX,
        help="Directory to watch for new .parquet files (default: data/inbox)",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=DEFAULT_REFERENCE,
        help="Directory where processed batches are moved (default: data/reference)",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=0,
        help="Seconds between checks; 0 = run once and exit (default: 0)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without running the flow or moving files",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the watcher script."""
    args = parse_args()

    # Ensure directories exist
    args.inbox.mkdir(parents=True, exist_ok=True)
    args.reference.mkdir(parents=True, exist_ok=True)

    if args.poll_interval <= 0:
        # One-shot mode: process and exit
        process_inbox(args.inbox, args.reference, dry_run=args.dry_run)
    else:
        # Continuous polling mode
        logger.info(
            "👀  Watching %s every %ds (Ctrl+C to stop)",
            args.inbox,
            args.poll_interval,
        )
        try:
            while True:
                process_inbox(args.inbox, args.reference, dry_run=args.dry_run)
                time.sleep(args.poll_interval)
        except KeyboardInterrupt:
            logger.info("\n🛑  Watcher stopped.")


if __name__ == "__main__":
    main()
