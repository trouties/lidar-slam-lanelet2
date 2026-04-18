#!/usr/bin/env python3
"""Smoke-verify a cached KITTI rosbag (SUP-01 Stage B gate).

Runs inside the ``slam-baselines/bag-builder`` container (requires ROS Noetic
``rosbag`` module). Asserts hard invariants on:

- LiDAR / IMU message counts (per-sequence hardcoded expectations)
- IMU header stamp strict monotonicity (GTSAM preintegration precondition)
- PointCloud2 field layout (LIO-SAM requires ring/time)
- IMU z-accel gravity invariant (body-frame specific force ~ +9.8)

Exit 0 = all green; exit 1 = any hard check failed.

Usage (host)::

    make verify-bag-00

Usage (container)::

    python3 /scripts/verify_bag_impl.py --bag /cache/kitti_00_fixed.bag --sequence 00
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import rosbag  # noqa: F401  — provided by the bag-builder image

# KITTI Odometry frame counts (LiDAR scans per sequence).
EXPECTED_LIDAR_COUNT = {
    "00": 4541,
    "01": 1101,
    "02": 4661,
    "03": 801,
    "04": 271,
    "05": 2761,
    "06": 1101,
    "07": 1101,
    "08": 4071,
    "09": 1591,
    "10": 1201,
}

LIDAR_TOPIC = "/velodyne_points"
IMU_TOPIC = "/imu/data"


def _expect(label: str, ok: bool, msg: str) -> bool:
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {label}: {msg}")
    return ok


def verify_bag(bag_path: Path, sequence: str) -> bool:
    if not bag_path.exists():
        print(f"FATAL: bag not found: {bag_path}")
        return False

    print(f"Verifying {bag_path} (seq {sequence})")

    n_lidar = 0
    n_imu = 0
    prev_imu_ns: int | None = None
    imu_monotonic = True
    first_pc_fields: list[tuple[str, int, int]] | None = None
    first_pc_point_step = -1
    imu_z_accel_first200: list[float] = []

    with rosbag.Bag(str(bag_path), "r") as bag:
        for topic, msg, _t in bag.read_messages():
            if topic == LIDAR_TOPIC:
                n_lidar += 1
                if first_pc_fields is None:
                    first_pc_fields = [(f.name, f.offset, f.datatype) for f in msg.fields]
                    first_pc_point_step = msg.point_step
            elif topic == IMU_TOPIC:
                n_imu += 1
                ns = msg.header.stamp.to_nsec()
                if prev_imu_ns is not None and ns <= prev_imu_ns:
                    imu_monotonic = False
                prev_imu_ns = ns
                if n_imu <= 200:
                    imu_z_accel_first200.append(float(msg.linear_acceleration.z))

    all_ok = True

    # 1. LiDAR count
    exp_lidar = EXPECTED_LIDAR_COUNT.get(sequence)
    if exp_lidar is None:
        _expect("lidar count", True, f"seq {sequence} has no hardcoded target; got {n_lidar}")
    else:
        ok = abs(n_lidar - exp_lidar) <= 5
        all_ok &= _expect("lidar count", ok, f"{n_lidar} (expected {exp_lidar} ±5)")

    # 2. IMU count (≥ 8x LiDAR, ~10x typically)
    if n_imu == 0:
        all_ok &= _expect("imu count", False, "zero IMU messages — was --with-imu set?")
    elif n_lidar > 0:
        ratio = n_imu / n_lidar
        ok = 8.0 <= ratio <= 12.0
        all_ok &= _expect(
            "imu count",
            ok,
            f"{n_imu} samples over {n_lidar} frames (ratio {ratio:.2f}, expected 8-12)",
        )

    # 3. IMU monotonicity
    all_ok &= _expect(
        "imu monotonic",
        imu_monotonic,
        "strict forward timestamps" if imu_monotonic else "non-monotonic stamps detected!",
    )

    # 4. PC2 field layout (LIO-SAM requires ring + time)
    if first_pc_fields is None:
        all_ok &= _expect("pc2 fields", False, "no PointCloud2 message read")
    else:
        names = [n for n, _, _ in first_pc_fields]
        has_ring = "ring" in names
        has_time = "time" in names
        pc_ok = has_ring and has_time and first_pc_point_step >= 22
        all_ok &= _expect(
            "pc2 fields",
            pc_ok,
            f"names={names} point_step={first_pc_point_step} (need ring+time, step≥22)",
        )

    # 5. IMU body-frame z-accel gravity check (~ +9.8 when stationary level)
    if imu_z_accel_first200:
        mean_z = sum(imu_z_accel_first200) / len(imu_z_accel_first200)
        ok = 9.0 <= mean_z <= 10.5
        all_ok &= _expect(
            "imu z-accel gravity",
            ok,
            f"mean over first {len(imu_z_accel_first200)} samples = {mean_z:+.3f} "
            f"m/s² (expected 9.0-10.5 for body-frame specific force)",
        )
    else:
        all_ok &= _expect("imu z-accel gravity", False, "no IMU samples to measure")

    print()
    print("OVERALL: " + ("PASS" if all_ok else "FAIL"))
    return all_ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bag", type=Path, required=True, help="Path to kitti_SS_fixed.bag")
    ap.add_argument("--sequence", required=True, help="KITTI sequence id (00-10)")
    args = ap.parse_args()
    return 0 if verify_bag(args.bag, args.sequence) else 1


if __name__ == "__main__":
    sys.exit(main())
