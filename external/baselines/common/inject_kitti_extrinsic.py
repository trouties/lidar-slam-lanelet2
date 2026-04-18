#!/usr/bin/env python3
"""Inject KITTI Raw calib_imu_to_velo extrinsic into LIO-SAM params.yaml.

Background
----------
KITTI Raw ``calib_imu_to_velo.txt`` provides::

    R: 9 floats, row-major   R_velo_imu  (p_velo = R @ p_imu + T)
    T: 3 floats              IMU origin expressed in the Velodyne frame

LIO-SAM's ``imuPreintegration.cpp`` uses ``extRot`` as::

    acc_lidar = extRot @ acc_imu

so ``extrinsicRot == R_velo_imu`` directly (NOT its transpose).
``extrinsicTrans == T``.

``extrinsicRPY`` is set to the SAME matrix as ``extrinsicRot``. This matches
TixiaoShan's upstream ``params_kitti.yaml`` and was empirically validated:
overriding RPY to a pure identity (theoretically justified because
R_velo_imu is within ~1 deg of identity) regresses Seq 00 SE(3) APE from
30 m to 44 m. LIO-SAM treats ``extrinsicRPY`` as an exact rotation, not an
approximation, so the ~0.85 deg gap matters.

Usage
-----
    python3 inject_kitti_extrinsic.py \\
        --sequence 00 \\
        --kitti-raw-dir ~/data/kitti_raw \\
        --params external/baselines/lio_sam/config/params.yaml

Run by ``make build-liosam-<seq>`` before ``docker build`` so each sequence
gets its own image tag with the matching calibration baked in. Per-sequence
images are necessary because Seq 00/01/02 use a different calib date than
Seq 04-10.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

# KITTI Odometry sequence -> KITTI Raw date folder (calib is per-date)
ODOM_TO_RAW_DATE = {
    "00": "2011_10_03",
    "01": "2011_10_03",
    "02": "2011_10_03",
    "03": "2011_09_26",
    "04": "2011_09_30",
    "05": "2011_09_30",
    "06": "2011_09_30",
    "07": "2011_09_30",
    "08": "2011_09_30",
    "09": "2011_09_30",
    "10": "2011_09_30",
}


def load_calib_imu_to_velo(calib_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse ``calib_imu_to_velo.txt`` and validate the rotation.

    Returns
    -------
    (R_velo_imu, T_velo_imu)  -- ``p_velo = R @ p_imu + T``.
    """
    R: np.ndarray | None = None
    T: np.ndarray | None = None
    for line in calib_path.read_text().splitlines():
        line = line.strip()
        if line.startswith("R:"):
            R = np.array([float(x) for x in line[2:].split()]).reshape(3, 3)
        elif line.startswith("T:"):
            T = np.array([float(x) for x in line[2:].split()])
    if R is None or T is None:
        raise ValueError(f"Failed to parse R/T from {calib_path}")

    det = float(np.linalg.det(R))
    if not (0.99 < det < 1.01):
        raise ValueError(f"R has det={det:.6f}, not a proper rotation")
    if abs(np.trace(R) - 3.0) > 0.1:
        raise ValueError(
            f"R trace={np.trace(R):.4f}, expected ~3 for KITTI (IMU/Velo nearly aligned)"
        )
    return R, T


def inject(params_path: Path, R: np.ndarray, T: np.ndarray) -> None:
    """Rewrite ``extrinsicTrans`` / ``extrinsicRot`` / ``extrinsicRPY`` in-place."""
    text = params_path.read_text()
    rot_str = ", ".join(f"{v:.9e}" for v in R.flatten())
    trans_str = ", ".join(f"{v:.9e}" for v in T)

    text, n_rot = re.subn(
        r"extrinsicRot:\s*\[[^\]]*\]",
        f"extrinsicRot: [{rot_str}]",
        text,
        flags=re.DOTALL,
    )
    text, n_rpy = re.subn(
        r"extrinsicRPY:\s*\[[^\]]*\]",
        f"extrinsicRPY: [{rot_str}]",
        text,
        flags=re.DOTALL,
    )
    text, n_tr = re.subn(
        r"extrinsicTrans:\s*\[[^\]]*\]",
        f"extrinsicTrans: [{trans_str}]",
        text,
        flags=re.DOTALL,
    )
    if (n_rot, n_rpy, n_tr) != (1, 1, 1):
        raise ValueError(
            f"Substitution counts wrong: rot={n_rot} rpy={n_rpy} trans={n_tr}; "
            f"expected (1, 1, 1). Make sure the placeholder lines exist."
        )
    params_path.write_text(text)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--sequence", required=True, help="KITTI Odometry seq id, e.g. 00 or 05")
    ap.add_argument(
        "--kitti-raw-dir",
        type=Path,
        required=True,
        help="KITTI Raw root, e.g. ~/data/kitti_raw",
    )
    ap.add_argument(
        "--params",
        type=Path,
        required=True,
        help="Path to lio_sam/config/params.yaml to mutate in place",
    )
    args = ap.parse_args()

    date = ODOM_TO_RAW_DATE.get(args.sequence)
    if date is None:
        print(f"ERROR: no KITTI Raw date mapping for sequence {args.sequence}", file=sys.stderr)
        return 1

    calib = args.kitti_raw_dir.expanduser() / date / "calib_imu_to_velo.txt"
    if not calib.exists():
        print(f"ERROR: calib file missing: {calib}", file=sys.stderr)
        return 1

    R, T = load_calib_imu_to_velo(calib)
    inject(args.params, R, T)

    print(f"[inject_kitti_extrinsic] patched {args.params}")
    print(f"  seq {args.sequence} (date {date})")
    print(f"  R trace = {np.trace(R):.9f}  det = {np.linalg.det(R):.9f}")
    print(f"  T = [{T[0]:+.9e}, {T[1]:+.9e}, {T[2]:+.9e}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
