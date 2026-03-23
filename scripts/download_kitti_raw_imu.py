#!/usr/bin/env python3
"""SUP-04: Download KITTI Raw oxts (IMU/GPS) data.

Downloads only the oxts subdirectory for KITTI Raw drives that correspond
to KITTI Odometry sequences.  Each drive is ~10-50 MB (oxts only).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path

# KITTI Odometry → Raw mapping (official correspondence)
# Source: https://www.cvlibs.net/datasets/kitti/eval_odometry.php
ODOM_TO_RAW = {
    "00": ("2011_10_03", "0027"),
    "01": ("2011_10_03", "0042"),
    "02": ("2011_10_03", "0034"),
    "03": ("2011_09_26", "0067"),
    "04": ("2011_09_30", "0016"),
    "05": ("2011_09_30", "0018"),
    "06": ("2011_09_30", "0020"),
    "07": ("2011_09_30", "0027"),
    "08": ("2011_09_30", "0028"),
    "09": ("2011_09_30", "0033"),
    "10": ("2011_09_30", "0034"),
}

RAW_ROOT = Path(os.path.expanduser("~/data/kitti_raw"))

# KITTI Raw data download URL template (extract = unsynced, sync = synced)
URL_TEMPLATE = (
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/"
    "{date}_drive_{drive}/{date}_drive_{drive}_extract.zip"
)


def download_oxts(seq: str, timeout_s: int = 900) -> bool:
    """Download and extract oxts data for a KITTI Odometry sequence.

    Returns True if successful.
    """
    if seq not in ODOM_TO_RAW:
        print(f"  No Raw mapping for Odometry seq {seq}")
        return False

    date, drive = ODOM_TO_RAW[seq]
    drive_name = f"{date}_drive_{drive}_extract"
    oxts_dir = RAW_ROOT / date / f"{date}_drive_{drive}_extract" / "oxts"

    if oxts_dir.exists() and list(oxts_dir.glob("data/*.txt")):
        print(f"  [{seq}] Already exists: {oxts_dir}")
        return True

    url = URL_TEMPLATE.format(date=date, drive=drive)
    zip_path = RAW_ROOT / f"{drive_name}.zip"
    RAW_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"  [{seq}] Downloading {url}")
    try:
        result = subprocess.run(
            ["wget", "-c", "-q", "--show-progress", "-O", str(zip_path), url],
            timeout=timeout_s,
            check=False,
        )
        if result.returncode != 0:
            print(f"  [{seq}] Download failed (wget exit {result.returncode})")
            return False
    except subprocess.TimeoutExpired:
        print(f"  [{seq}] Download timed out after {timeout_s}s — partial file kept for resume")
        return False
    except FileNotFoundError:
        # wget not available, try curl
        try:
            result = subprocess.run(
                ["curl", "-C", "-", "-L", "-o", str(zip_path), url],
                timeout=timeout_s,
                check=False,
            )
            if result.returncode != 0:
                print(f"  [{seq}] Download failed (curl exit {result.returncode})")
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print(f"  [{seq}] Neither wget nor curl available")
            return False

    # Extract only oxts directory
    print(f"  [{seq}] Extracting oxts from {zip_path.name}...")
    extract_dir = RAW_ROOT / date
    extract_dir.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_path) as zf:
            oxts_members = [m for m in zf.namelist() if "/oxts/" in m]
            if not oxts_members:
                print(f"  [{seq}] No oxts directory found in zip")
                return False
            for member in oxts_members:
                zf.extract(member, extract_dir)
        print(f"  [{seq}] Extracted {len(oxts_members)} files to {extract_dir}")
    except zipfile.BadZipFile:
        print(f"  [{seq}] Bad zip file")
        return False
    finally:
        zip_path.unlink(missing_ok=True)

    return oxts_dir.exists()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download KITTI Raw IMU data")
    parser.add_argument(
        "--sequences", default="00,05",
        help="Comma-separated Odometry sequence ids",
    )
    parser.add_argument(
        "--timeout", type=int, default=900,
        help="Download timeout per file in seconds",
    )
    args = parser.parse_args()

    sequences = [s.strip() for s in args.sequences.split(",")]
    results = {}

    for seq in sequences:
        print(f"\n=== Sequence {seq} ===")
        ok = download_oxts(seq, timeout_s=args.timeout)
        results[seq] = "OK" if ok else "FAIL"

    print(f"\n{'='*40}")
    print("Download results:")
    for seq, status in results.items():
        print(f"  {seq}: {status}")

    if all(v == "OK" for v in results.values()):
        print("\nAll downloads successful.")
        sys.exit(0)
    else:
        print("\nSome downloads failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
