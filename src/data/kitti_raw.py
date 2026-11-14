"""KITTI Raw drives behind the KITTI Odometry sequences, and their OxTS logs.

The OxTS RT3003 GNSS/INS log of a raw drive is the only independent position
source for an odometry sequence: it georeferences the map
(``src/export/georef.py``) and tells where the ground truth comes from an
RTK-integer fix (``scripts/loop_closure_study.py gps``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# KITTI Odometry → Raw mapping (official correspondence)
ODOM_TO_RAW: dict[str, tuple[str, str]] = {
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


def load_oxts_sequence(oxts_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load all oxts data files from a KITTI Raw drive.

    Args:
        oxts_dir: Path to ``<drive>/oxts/`` directory containing
            ``data/`` and ``timestamps.txt``.

    Returns:
        Tuple of ``(data, timestamps)`` where:
        - data: ``(N, 30)`` float64 array of oxts measurements
        - timestamps: ``(N,)`` float64 array of timestamps in seconds
          (relative to first frame)
    """
    data_dir = oxts_dir / "data"
    ts_path = oxts_dir / "timestamps.txt"

    files = sorted(data_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No oxts data files in {data_dir}")

    rows = []
    for f in files:
        vals = np.loadtxt(f)
        rows.append(vals)
    data = np.array(rows, dtype=np.float64)

    # Parse timestamps
    if ts_path.exists():
        ts_lines = ts_path.read_text().strip().split("\n")
        timestamps = []
        for line in ts_lines:
            # KITTI Raw timestamps format: "2011-10-03 12:55:34.123456789"
            parts = line.strip().split()
            if len(parts) == 2:
                time_part = parts[1]
                h, m, s = time_part.split(":")
                t = float(h) * 3600 + float(m) * 60 + float(s)
                timestamps.append(t)
            else:
                timestamps.append(float(line.strip()))
        timestamps = np.array(timestamps, dtype=np.float64)
        timestamps -= timestamps[0]  # relative to first frame
    else:
        # Fallback: assume 10 Hz
        timestamps = np.arange(len(data), dtype=np.float64) * 0.1

    return data, timestamps
