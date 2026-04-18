#!/usr/bin/env python3
"""Repair non-monotonic IMU timestamps in a rosbag (SUP-01 P0-1).

GTSAM's ``PreintegratedImuMeasurements::integrateMeasurement`` throws
``dt <= 0`` when it receives an IMU sample whose header stamp is not
strictly greater than the previous one. KITTI Raw OxTS timestamps
occasionally contain duplicates or backward steps (hardware / parsing
artifact in ``oxts/timestamps.txt``), which propagates through
``kitti_to_rosbag.py`` and crashes LIO-SAM's imuPreintegration node on
Seq 00 well before any pose is published.

This script reads an input bag, forward-propagates IMU ``header.stamp``
values with the rule::

    t[i] := max(t[i-1] + eps_ns, t[i])

where ``eps_ns`` defaults to 1000 (1 microsecond). The bag record
timestamp for each IMU message is set to the repaired header stamp. All
non-IMU messages are passed through unchanged.

Usage
-----
    python3 fix_imu_timestamps.py --input in.bag --output out.bag

This runs inside the LIO-SAM / FAST-LIO2 docker containers (rosbag
module comes from ROS Noetic).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import rosbag
import rospy


def _enforce_monotonic_ns(prev_ns: int | None, orig_ns: int, eps_ns: int) -> int:
    """Forward-propagate a stamp so it stays >= prev_ns + eps_ns.

    Pure int arithmetic (extracted from repair() so it is unit-testable
    without mocking the rosbag IO). Passthrough when the original is
    already monotonic; otherwise bump to ``prev_ns + eps_ns``.
    """
    if prev_ns is None:
        return orig_ns
    return max(prev_ns + eps_ns, orig_ns)


def repair(
    input_bag: Path,
    output_bag: Path,
    imu_topic: str = "/imu/data",
    eps_ns: int = 1000,
) -> tuple[int, int, int]:
    """Copy input_bag to output_bag, forcing IMU header.stamp monotonicity.

    Returns
    -------
    (n_total_imu, n_repaired, n_other)
    """
    n_total_imu = 0
    n_repaired = 0
    n_other = 0
    prev_ns: int | None = None  # last repaired IMU stamp in nanoseconds

    with rosbag.Bag(str(input_bag), "r") as inbag, rosbag.Bag(str(output_bag), "w") as outbag:
        for topic, msg, t in inbag.read_messages():
            if topic == imu_topic:
                n_total_imu += 1
                orig_ns = msg.header.stamp.to_nsec()
                new_ns = _enforce_monotonic_ns(prev_ns, orig_ns, eps_ns)
                if new_ns != orig_ns:
                    n_repaired += 1
                prev_ns = new_ns
                new_stamp = rospy.Time(int(new_ns // 1_000_000_000), int(new_ns % 1_000_000_000))
                msg.header.stamp = new_stamp
                outbag.write(topic, msg, new_stamp)
            else:
                outbag.write(topic, msg, t)
                n_other += 1

    return n_total_imu, n_repaired, n_other


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--input", type=Path, required=True, help="Input bag with possibly bad IMU stamps"
    )
    ap.add_argument(
        "--output", type=Path, required=True, help="Output bag with repaired IMU stamps"
    )
    ap.add_argument("--imu-topic", default="/imu/data", help="IMU topic name (default: /imu/data)")
    ap.add_argument(
        "--eps-ns",
        type=int,
        default=1000,
        help="Minimum delta between consecutive IMU stamps, in nanoseconds (default: 1000 = 1us)",
    )
    args = ap.parse_args()

    if not args.input.exists():
        print(f"ERROR: input bag not found: {args.input}", file=sys.stderr)
        return 1

    n_imu, n_fixed, n_other = repair(args.input, args.output, args.imu_topic, args.eps_ns)
    print(
        f"fix_imu_timestamps: {args.input} -> {args.output}\n"
        f"  imu_total={n_imu}  imu_repaired={n_fixed}  other={n_other}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
