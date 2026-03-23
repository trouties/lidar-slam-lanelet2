"""IMU data loader for KITTI Raw oxts sequences.

Loads accelerometer and gyroscope measurements from KITTI Raw oxts files
and maps them to the corresponding KITTI Odometry sequence.
"""

from __future__ import annotations

import os
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

# OxTS data column indices (from KITTI devkit readme)
# Columns 11-16 are acceleration (ax, ay, az) in body frame
# Columns 17-19 are angular velocity (wx, wy, wz) in body frame
_AX, _AY, _AZ = 11, 12, 13
_WX, _WY, _WZ = 17, 18, 19


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


def extract_imu(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract accelerometer and gyroscope from oxts data.

    Args:
        data: ``(N, 30)`` oxts data array.

    Returns:
        Tuple of ``(acc, gyro)`` where:
        - acc: ``(N, 3)`` accelerometer [ax, ay, az] in m/s²
        - gyro: ``(N, 3)`` gyroscope [wx, wy, wz] in rad/s
    """
    acc = data[:, [_AX, _AY, _AZ]]
    gyro = data[:, [_WX, _WY, _WZ]]
    return acc, gyro


def _load_imu_to_velo_rotation(raw_root: Path, date: str) -> np.ndarray | None:
    """Load the IMU-to-Velodyne rotation matrix from KITTI Raw calibration."""
    calib_path = raw_root / date / "calib_imu_to_velo.txt"
    if not calib_path.exists():
        return None
    with calib_path.open() as f:
        for line in f:
            if line.startswith("R:"):
                vals = list(map(float, line.split(":")[1].strip().split()))
                return np.array(vals).reshape(3, 3)
    return None


def load_imu_for_odometry_seq(
    seq: str,
    raw_root: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Load IMU measurements for a KITTI Odometry sequence.

    Applies IMU-to-Velodyne rotation if calibration is available,
    so that acc/gyro are in the Velodyne frame matching the pose graph.

    Args:
        seq: Odometry sequence id (e.g. ``"00"``).
        raw_root: Root of KITTI Raw data. Defaults to ``~/data/kitti_raw``.

    Returns:
        Tuple of ``(acc, gyro, timestamps)`` or ``None`` if data unavailable.
        - acc: ``(N, 3)`` accelerometer in Velodyne frame
        - gyro: ``(N, 3)`` gyroscope in Velodyne frame
        - timestamps: ``(N,)`` timestamps in seconds
    """
    if seq not in ODOM_TO_RAW:
        return None

    if raw_root is None:
        raw_root = Path(os.path.expanduser("~/data/kitti_raw"))
    else:
        raw_root = Path(raw_root)

    date, drive = ODOM_TO_RAW[seq]
    oxts_dir = raw_root / date / f"{date}_drive_{drive}_extract" / "oxts"

    if not oxts_dir.exists():
        # Try sync variant
        oxts_dir = raw_root / date / f"{date}_drive_{drive}_sync" / "oxts"
        if not oxts_dir.exists():
            return None

    try:
        data, timestamps = load_oxts_sequence(oxts_dir)
    except (FileNotFoundError, ValueError):
        return None

    acc, gyro = extract_imu(data)

    # Transform from IMU body frame to Velodyne frame
    R_imu_to_velo = _load_imu_to_velo_rotation(raw_root, date)
    if R_imu_to_velo is not None:
        acc = (R_imu_to_velo @ acc.T).T
        gyro = (R_imu_to_velo @ gyro.T).T

    # Ensure monotonic timestamps (drop non-monotonic samples)
    mono_mask = np.ones(len(timestamps), dtype=bool)
    for i in range(1, len(timestamps)):
        if timestamps[i] <= timestamps[i - 1]:
            mono_mask[i] = False
    if not mono_mask.all():
        acc = acc[mono_mask]
        gyro = gyro[mono_mask]
        timestamps = timestamps[mono_mask]

    return acc, gyro, timestamps
