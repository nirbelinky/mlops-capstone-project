"""
download_data.py — Download NYC Green Taxi TLC trip-record data (Parquet format).

Purpose
-------
This script is the **first step** in the MLOps capstone pipeline.  Before any
model training, feature engineering, or drift detection can happen, we need raw
taxi-trip data on disk.  The NYC Taxi & Limousine Commission (TLC) publishes
monthly Parquet files on a public CloudFront CDN; this script fetches exactly
the three months the capstone project requires and saves them locally under a
``data/`` directory next to this file.

Why these three specific months?
--------------------------------
The capstone project is designed to demonstrate **data-drift detection** and
**automated retraining**.  To make that concrete we pick:

* **January 2024** — the *reference* (baseline) dataset.  The model is trained
  on this month, so all drift comparisons are made against it.
* **February 2024** — a *batch* from the same winter season.  Because the
  travel patterns are similar to January, we expect **little or no drift**,
  which lets us verify the pipeline's "no-retrain" path.
* **June 2024** — a *batch* from summer.  Seasonal changes in travel behaviour
  (tourism, weather, daylight hours) are likely to cause **measurable drift**,
  triggering the pipeline's retraining logic.

How to run
----------
From the ``08_mlops_capstone_project/`` directory::

    python download_data.py

The script is **idempotent**: if a file already exists on disk it is skipped,
so re-running is safe and fast.  A visual progress bar is printed to the
terminal while each file downloads.

Dependencies
------------
Only the Python standard library is used (``os``, ``ssl``, ``sys``,
``urllib.request``), so no extra packages need to be installed.
"""

# ---------------------------------------------------------------------------
# Standard-library imports — no third-party packages required.
# ---------------------------------------------------------------------------
import os       # File-system operations: path construction, directory creation, file existence checks
import ssl      # TLS/SSL context creation for HTTPS downloads
import sys      # Low-level stdout access for the progress-bar animation
import urllib.request  # High-level URL retrieval (urlretrieve) with progress-hook support

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# BASE_URL is the URL template for the NYC TLC trip-record Parquet files.
# The ``{year_month}`` placeholder is filled in at download time with values
# like ``"2024-01"``.  The domain ``d37ci6vzurychx.cloudfront.net`` is the
# official CloudFront CDN that the TLC uses to distribute its open data.
# Using a CDN means downloads are fast and geographically distributed.
BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year_month}.parquet"

# FILES is the manifest of datasets to download.  Each entry is a dict with:
#   • "year_month" — the YYYY-MM string plugged into BASE_URL to form the
#     remote URL (must match the TLC's naming convention).
#   • "local_name" — the relative path (under DATA_DIR) where the file is
#     saved.  Files are organised into subdirectories by role:
#       - ``reference/`` — baseline data the model was trained on.
#       - ``inbox/``     — new batches waiting to be processed by the flow.
#     The watcher script (``watcher.py``) moves successfully processed
#     batches from ``inbox/`` into ``reference/`` so the reference window
#     expands over time.
#
# The order matters only cosmetically (download sequence); the downstream
# pipeline identifies files by path, not by position in this list.
FILES = [
    {"year_month": "2024-01", "local_name": "reference/2024-01.parquet"},
    {"year_month": "2024-02", "local_name": "reference/2024-02.parquet"},
    {"year_month": "2024-03", "local_name": "inbox/2024-03.parquet"},
    {"year_month": "2024-06", "local_name": "inbox/2024-06.parquet"},
]

# DATA_DIR is the absolute path to the ``data/`` subdirectory that sits next
# to this script.  We derive it dynamically so the script works regardless of
# the current working directory (e.g. when invoked from the repo root via
# ``python 08_mlops_capstone_project/download_data.py``).
#
# Breakdown:
#   __file__                → path to *this* Python file
#   os.path.abspath(...)    → resolve to an absolute path (handles symlinks)
#   os.path.dirname(...)    → strip the filename, leaving the parent directory
#   os.path.join(...,"data")→ append "data" to get the target directory
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


# ---------------------------------------------------------------------------
# Helper: progress bar callback
# ---------------------------------------------------------------------------

def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    """Print a simple progress bar to stdout during a file download.

    This function is passed as the ``reporthook`` argument to
    ``urllib.request.urlretrieve``.  ``urlretrieve`` calls it once after every
    network block is read, supplying three positional arguments that describe
    how much data has been transferred so far.

    Args:
        block_num:  The number of blocks transferred so far (starts at 0).
                    Multiplied by ``block_size`` to get bytes downloaded.
        block_size: The size of each block in bytes (typically 8 KiB, but
                    this is controlled by the HTTP library).
        total_size: The total size of the file in bytes as reported by the
                    server's ``Content-Length`` header.  If the server does
                    not provide this header, ``total_size`` is **-1**, and
                    we fall back to showing only the amount downloaded.

    Returns:
        None.  Output is written directly to ``sys.stdout``.

    Why ``sys.stdout.write`` + ``flush`` instead of ``print``?
        We use a carriage return (``\\r``) at the start of the line to
        *overwrite* the previous progress output in-place, creating the
        illusion of an animated progress bar.  ``print()`` appends a newline
        by default, which would scroll the terminal instead.  Calling
        ``flush()`` ensures the output appears immediately rather than being
        buffered.
    """
    # Calculate total bytes downloaded so far
    downloaded = block_num * block_size

    if total_size > 0:
        # --- Known file size: show a percentage-based progress bar ---

        # Clamp percentage to 100% because the last block may overshoot
        pct = min(100.0, downloaded / total_size * 100)

        # Convert bytes to megabytes for human-readable display
        mb_down = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)

        # Build a fixed-width bar of 40 characters using Unicode block chars:
        #   █ (U+2588) for the filled portion
        #   ░ (U+2591) for the remaining portion
        bar_len = 40
        filled = int(bar_len * pct / 100)
        bar = "█" * filled + "░" * (bar_len - filled)

        # Write the bar, percentage, and MB counts on a single overwritten line
        sys.stdout.write(f"\r  |{bar}| {pct:5.1f}%  ({mb_down:.1f}/{mb_total:.1f} MB)")
    else:
        # --- Unknown file size (no Content-Length header) ---
        # We can't show a percentage, so just show how much has been fetched.
        mb_down = downloaded / (1024 * 1024)
        sys.stdout.write(f"\r  Downloaded {mb_down:.1f} MB ...")

    # Force the output to appear immediately (Python buffers stdout by default)
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Helper: SSL context factory
# ---------------------------------------------------------------------------

def _get_ssl_context() -> ssl.SSLContext:
    """Create and return an SSL context for HTTPS connections.

    In most environments the default SSL context (which verifies server
    certificates against the system's trusted CA bundle) works fine.  However,
    in some corporate or academic networks a **proxy** re-signs TLS traffic
    with its own certificate authority, causing ``ssl.SSLError`` when Python
    tries to verify the server's certificate against the *system* CA store.

    This helper tries the secure path first and, only if that fails, falls
    back to an **unverified** context (equivalent to ``curl -k``).  The
    fallback is acceptable here because we are downloading *public, read-only*
    open data — there is no sensitive information at risk.

    Returns:
        ssl.SSLContext: A context suitable for passing to an HTTPS handler.

    Note:
        In practice, ``download_all()`` below bypasses this function and
        directly creates an unverified context for simplicity.  This helper
        is retained as a more cautious alternative that could replace the
        inline context creation if stricter certificate handling is desired.
    """
    try:
        # Attempt to create a context that validates certificates using the
        # operating system's default trusted CA bundle.
        ctx = ssl.create_default_context()
        # If no exception was raised, the CA bundle loaded successfully —
        # return the secure context.
        return ctx
    except ssl.SSLError:
        # Certificate verification setup failed (e.g. missing CA bundle).
        # Fall through to the unverified fallback below.
        pass

    # Fallback: create a default context but disable hostname checking and
    # certificate verification entirely.  This makes the connection
    # vulnerable to man-in-the-middle attacks, but is acceptable for
    # downloading public open data when no other option is available.
    ctx = ssl.create_default_context()
    ctx.check_hostname = False          # Don't verify that the cert matches the hostname
    ctx.verify_mode = ssl.CERT_NONE     # Don't verify the certificate chain at all
    return ctx


# ---------------------------------------------------------------------------
# Main download logic
# ---------------------------------------------------------------------------

def download_all() -> None:
    """Download all configured Parquet files into ``DATA_DIR``.

    This is the main entry point of the script.  It iterates over the ``FILES``
    manifest, constructs the remote URL for each entry, and downloads the file
    to the local ``data/`` directory.  Files that already exist on disk are
    skipped to make the function **idempotent** — you can safely call it
    multiple times without re-downloading data you already have.

    Workflow:
        1. Create the ``data/`` directory if it doesn't exist yet.
        2. Install a global HTTPS opener that skips SSL certificate
           verification (needed behind some corporate proxies / VPNs).
        3. For each file in ``FILES``:
           a. Build the full remote URL from ``BASE_URL``.
           b. Build the local destination path under ``DATA_DIR``.
           c. If the file already exists, print its size and skip it.
           d. Otherwise, download it with a progress bar.
           e. On failure, remove any partially-written file and re-raise
              the exception so the caller knows something went wrong.

    Returns:
        None.  Side effects: files are written to disk and status messages
        are printed to stdout.

    Raises:
        Exception: Re-raises any exception from ``urllib.request.urlretrieve``
            (e.g. ``urllib.error.URLError`` for network errors, ``OSError``
            for disk-full conditions).  Partial files are cleaned up before
            the exception propagates.
    """
    # Ensure the target directory exists.  ``exist_ok=True`` means "don't
    # raise an error if the directory is already there" — this is what makes
    # the function safe to call repeatedly.
    os.makedirs(DATA_DIR, exist_ok=True)

    # --- SSL workaround ---
    # Create an *unverified* SSL context.  This is the nuclear option: it
    # disables all certificate checks.  We do this because:
    #   • The data is public and read-only — no secrets are transmitted.
    #   • Many university / corporate networks use TLS-intercepting proxies
    #     whose certificates are not in Python's CA bundle, causing downloads
    #     to fail with certificate-verification errors.
    # By installing this context as the *global* opener, every subsequent
    # ``urllib.request`` call in this process will use it automatically.
    ctx = ssl._create_unverified_context()

    # Build a custom URL opener that uses our permissive SSL context and
    # install it as the global default.  After this call, even
    # ``urllib.request.urlretrieve`` (which doesn't accept a context argument
    # directly) will use the unverified context.
    opener = urllib.request.build_opener(
        urllib.request.HTTPSHandler(context=ctx)
    )
    urllib.request.install_opener(opener)

    # --- Iterate over the download manifest ---
    for entry in FILES:
        # Construct the full remote URL by substituting the year-month into
        # the BASE_URL template (e.g. "2024-01" → ".../green_tripdata_2024-01.parquet")
        url = BASE_URL.format(year_month=entry["year_month"])

        # Construct the full local file path (e.g. ".../data/reference/2024-01.parquet")
        dest = os.path.join(DATA_DIR, entry["local_name"])

        # Ensure the subdirectory (e.g. data/reference/, data/inbox/) exists.
        os.makedirs(os.path.dirname(dest), exist_ok=True)

        # --- Idempotency check ---
        # If the file already exists we assume a previous run completed
        # successfully and skip the download.  We print the file size so the
        # user can verify it looks reasonable.
        if os.path.exists(dest):
            size_mb = os.path.getsize(dest) / (1024 * 1024)
            print(f"⏭  {entry['local_name']} already exists ({size_mb:.1f} MB) – skipping")
            continue

        # --- Download the file ---
        print(f"⬇  Downloading {entry['local_name']} …")
        print(f"   {url}")
        try:
            # ``urlretrieve`` downloads the URL to a local file.  The
            # ``reporthook`` callback (_progress_hook) is called after every
            # block is read, giving us the animated progress bar.
            urllib.request.urlretrieve(url, dest, reporthook=_progress_hook)

            # Print a confirmation with the final file size
            size_mb = os.path.getsize(dest) / (1024 * 1024)
            print(f"\n   ✅ Saved ({size_mb:.1f} MB)")
        except Exception as exc:
            # --- Error handling ---
            # If anything goes wrong (network timeout, DNS failure, disk full,
            # etc.) we print the error, clean up any partially-written file
            # (so a future re-run doesn't mistakenly think the download
            # succeeded), and re-raise the exception.
            print(f"\n   ❌ Failed: {exc}")

            # Remove partial file to avoid leaving corrupt data on disk.
            # A future re-run will then correctly attempt the download again.
            if os.path.exists(dest):
                os.remove(dest)
            raise

    # Summary line printed after all files have been processed
    print("\nDone – all files are in:", DATA_DIR)


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

# The standard Python idiom: when this file is executed directly
# (``python download_data.py``), call ``download_all()``.  When it is
# *imported* by another module (e.g. for testing or orchestration), this
# block is skipped, allowing the caller to invoke ``download_all()``
# explicitly or to use the helper functions in isolation.
if __name__ == "__main__":
    download_all()
