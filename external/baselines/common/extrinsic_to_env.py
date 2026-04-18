#!/usr/bin/env python3
"""Emit KITTI calib_imu_to_velo extrinsic as a comma-separated shell string.

SUP-01 Stage C: LIO-SAM now reads its LiDAR-IMU extrinsic from KITTI Raw
calibration at container start (runtime envsubst) instead of during
``docker build`` via ``inject_kitti_extrinsic.py``. This removes the
build-time layer-cache divergence tracked in ``refs/pipeline-notes.md``
§20.4.

Usage (inside the lio_sam container)::

    KITTI_EXT_ROT=$(python3 /scripts/extrinsic_to_env.py \\
        --sequence $SEQ --kitti-raw-dir /data/kitti_raw --field rot)
    KITTI_EXT_TRANS=$(python3 /scripts/extrinsic_to_env.py \\
        --sequence $SEQ --kitti-raw-dir /data/kitti_raw --field trans)
    envsubst < /config/params.yaml.tpl > /config/params.yaml

``--field rot`` prints 9 floats (row-major R_velo_imu) separated by ``, ``.
``--field trans`` prints 3 floats (T_velo_imu).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Reuses the validated calib parser from inject_kitti_extrinsic.py (rejects
# dets ≠ 1 and traces far from 3, guarding the Bug A reflection class).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from inject_kitti_extrinsic import ODOM_TO_RAW_DATE, load_calib_imu_to_velo  # noqa: E402


def format_field(values, precision: int = 9) -> str:
    return ", ".join(f"{v:+.{precision}e}" for v in values)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sequence", required=True, help="KITTI Odometry seq id (00-10)")
    ap.add_argument(
        "--kitti-raw-dir",
        type=Path,
        required=True,
        help="KITTI Raw root with <date>/calib_imu_to_velo.txt",
    )
    ap.add_argument(
        "--field",
        choices=("rot", "trans"),
        required=True,
        help="Which component to emit (rot = 9 floats, trans = 3 floats)",
    )
    args = ap.parse_args()

    date = ODOM_TO_RAW_DATE.get(args.sequence)
    if date is None:
        print(f"ERROR: no KITTI Raw date mapping for seq {args.sequence}", file=sys.stderr)
        return 1

    calib = args.kitti_raw_dir.expanduser() / date / "calib_imu_to_velo.txt"
    if not calib.exists():
        print(f"ERROR: calib file missing: {calib}", file=sys.stderr)
        return 1

    R, T = load_calib_imu_to_velo(calib)
    if args.field == "rot":
        print(format_field(R.flatten()))
    else:
        print(format_field(T))
    return 0


if __name__ == "__main__":
    sys.exit(main())
