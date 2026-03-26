"""
Download PTB-XL v1.0.3 from PhysioNet.

PTB-XL is open-access — no account required.
~1.8 GB total (500 Hz records).  ~400 MB for 100 Hz only.

Usage:
    python training/ptbxl_download.py

Output:
    data/ptb-xl/   (contains ptbxl_database.csv, scp_statements.csv,
                    records500/, records100/, LICENSE.txt)
"""

import os
import subprocess
import sys

OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "ptb-xl",
)

PHYSIONET_URL = "https://physionet.org/files/ptb-xl/1.0.3/"


def _check_wget():
    try:
        subprocess.run(["wget", "--version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def download_wfdb():
    """Download using wfdb.dl_database (preferred — handles resume automatically)."""
    import wfdb
    print(f"Downloading PTB-XL via wfdb to {OUT_DIR} ...")
    print("This will download ~1.8 GB.  May take 10–30 minutes depending on connection.")
    os.makedirs(OUT_DIR, exist_ok=True)
    wfdb.dl_database("ptb-xl", OUT_DIR)
    print(f"\nDownload complete.  Files saved to: {OUT_DIR}")


def download_wget():
    """Fallback: recursive wget download."""
    print(f"Downloading PTB-XL via wget to {OUT_DIR} ...")
    print("This will download ~1.8 GB.  May take 10–30 minutes.")
    os.makedirs(OUT_DIR, exist_ok=True)
    cmd = [
        "wget", "-r", "-N", "-c", "-np",
        "--directory-prefix", OUT_DIR,
        "--cut-dirs", "3",
        PHYSIONET_URL,
    ]
    subprocess.run(cmd, check=True)
    print(f"\nDownload complete.  Files saved to: {OUT_DIR}")


def verify_download():
    """Check that the essential files are present."""
    required = [
        "ptbxl_database.csv",
        "scp_statements.csv",
    ]
    missing = [f for f in required if not os.path.exists(os.path.join(OUT_DIR, f))]
    if missing:
        print(f"WARNING: Missing files after download: {missing}")
        return False

    # Check at least one record directory exists
    for d in ("records500", "records100"):
        if os.path.isdir(os.path.join(OUT_DIR, d)):
            n = sum(1 for _ in os.scandir(os.path.join(OUT_DIR, d)))
            print(f"  {d}/: {n} sub-folders found")
            break
    else:
        print("WARNING: No records500/ or records100/ directory found.")
        return False

    print("Verification passed — PTB-XL download looks complete.")
    return True


def main():
    if os.path.exists(os.path.join(OUT_DIR, "ptbxl_database.csv")):
        print(f"PTB-XL already present at {OUT_DIR}")
        verify_download()
        return

    # wget is much faster than wfdb.dl_database (avoids per-record enumeration)
    if _check_wget():
        download_wget()
    else:
        try:
            download_wfdb()
        except Exception as e:
            print(
                f"\nDownload failed ({e}).\n"
                "Please install wget and retry:  brew install wget\n"
                f"Or manually:  wget -r -N -c -np {PHYSIONET_URL} -P {OUT_DIR}"
            )
            sys.exit(1)

    verify_download()


if __name__ == "__main__":
    main()
