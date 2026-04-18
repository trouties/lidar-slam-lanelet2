#!/usr/bin/env python3
"""Extract poses from a recorded ROS odometry bag into KITTI format.

Reads a bag containing nav_msgs/Odometry messages, aligns them to the
original KITTI LiDAR timestamps, and writes a KITTI pose file
(12 floats per line, 3x4 row-major SE(3)).

Usage:
    python3 extract_poses.py \
        --odom-bag /tmp/odom.bag \
        --odom-topic /hdl_graph_slam/odom \
        --lidar-bag /tmp/kitti_00.bag \
        --output /output/poses_00.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rosbag


def load_frame_tf(path: str) -> np.ndarray:
    """Load a 4x4 SE(3) transform from a text file.

    Accepts either 16 floats (4x4) or 12 floats (3x4 row-major, last row assumed [0,0,0,1]).
    """
    data = np.loadtxt(path).flatten()
    if data.size == 16:
        T = data.reshape(4, 4)
    elif data.size == 12:
        T = np.eye(4)
        T[:3, :] = data.reshape(3, 4)
    else:
        raise ValueError(f"frame-tf file {path} has {data.size} floats; expected 12 or 16")
    return T


def write_audit(
    path: str,
    poses: list[np.ndarray],
    n_matched: int,
    max_dt: float,
    frame_tf_path: str | None,
) -> None:
    """Write a small JSON diagnostic summarizing the extracted trajectory."""
    sample_idx = [0, min(1, len(poses) - 1), min(10, len(poses) - 1), len(poses) - 1]
    sample_idx = sorted(set(i for i in sample_idx if 0 <= i < len(poses)))
    T0_inv = np.linalg.inv(poses[0]) if poses else np.eye(4)
    samples = []
    for i in sample_idx:
        T = poses[i]
        rel = T0_inv @ T
        samples.append({
            "idx": int(i),
            "abs_translation_m": [float(x) for x in T[:3, 3]],
            "abs_translation_norm_m": float(np.linalg.norm(T[:3, 3])),
            "rel_to_first_translation_m": [float(x) for x in rel[:3, 3]],
            "rel_to_first_translation_norm_m": float(np.linalg.norm(rel[:3, 3])),
        })
    first_t_norm = float(np.linalg.norm(poses[0][:3, 3])) if poses else 0.0
    audit = {
        "n_poses": len(poses),
        "n_matched": int(n_matched),
        "max_dt_s": float(max_dt),
        "frame_tf_path": frame_tf_path,
        "first_pose_abs_translation_norm_m": first_t_norm,
        "samples": samples,
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(audit, f, indent=2)
    print(f"Wrote audit to {path}")


def quat_to_rotation(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Convert quaternion (x, y, z, w) to 3x3 rotation matrix."""
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    q /= np.linalg.norm(q)
    x, y, z, w = q

    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def extract(
    odom_bag_path: str,
    odom_topic: str,
    lidar_bag_path: str,
    output_path: str,
    max_dt: float = 0.15,
    frame_tf_path: str | None = None,
    audit_out: str | None = None,
) -> None:
    """Extract poses aligned to LiDAR frame timestamps.

    Args:
        odom_bag_path: Path to bag with recorded odometry messages.
        odom_topic: ROS topic name for odometry.
        lidar_bag_path: Path to input KITTI bag (for LiDAR timestamps).
        output_path: Output KITTI pose file path.
        max_dt: Maximum time difference (s) for matching odom to LiDAR frame.
        frame_tf_path: Optional path to a 4x4 SE(3) TXT T. Each pose is
            post-processed as pose_new = T^{-1} @ pose_old @ T, which
            re-expresses the trajectory from the odometry topic's native frame
            into the target frame (e.g., /map -> /velodyne).
        audit_out: Optional path to write a small JSON diagnostic.
    """
    # Collect LiDAR timestamps
    print(f"Reading LiDAR timestamps from {lidar_bag_path}...")
    lidar_stamps = []
    with rosbag.Bag(lidar_bag_path, "r") as bag:
        for _, msg, t in bag.read_messages(topics=["/velodyne_points"]):
            lidar_stamps.append(t.to_sec())
    print(f"  {len(lidar_stamps)} LiDAR frames")

    # Collect odometry messages
    print(f"Reading odometry from {odom_bag_path} topic={odom_topic}...")
    odom_stamps = []
    odom_poses = []
    with rosbag.Bag(odom_bag_path, "r") as bag:
        for _, msg, t in bag.read_messages(topics=[odom_topic]):
            odom_stamps.append(t.to_sec())
            p = msg.pose.pose.position
            q = msg.pose.pose.orientation
            R = quat_to_rotation(q.x, q.y, q.z, q.w)
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = [p.x, p.y, p.z]
            odom_poses.append(T)
    print(f"  {len(odom_poses)} odometry messages")

    if not odom_poses:
        print("ERROR: No odometry messages found!", file=sys.stderr)
        sys.exit(1)

    odom_stamps = np.array(odom_stamps)

    # Match each LiDAR frame to nearest odometry message
    poses_out = []
    n_matched = 0
    last_pose = np.eye(4)  # hold-last-value for missing frames

    for i, lt in enumerate(lidar_stamps):
        idx = np.argmin(np.abs(odom_stamps - lt))
        dt = abs(odom_stamps[idx] - lt)

        if dt <= max_dt:
            last_pose = odom_poses[idx]
            n_matched += 1

        poses_out.append(last_pose)

    print(f"  Matched {n_matched}/{len(lidar_stamps)} frames (max_dt={max_dt}s)")

    # Apply frame transform (source frame -> target frame) if requested.
    if frame_tf_path:
        T_frame = load_frame_tf(frame_tf_path)
        T_frame_inv = np.linalg.inv(T_frame)
        poses_out = [T_frame_inv @ P @ T_frame for P in poses_out]
        print(f"  Applied frame-tf from {frame_tf_path}")

    # Validate: first pose should be approximately identity
    first_t = np.linalg.norm(poses_out[0][:3, 3])
    if first_t > 5.0:
        print(f"WARNING: First pose translation={first_t:.2f}m (expected ~0)")

    # Write KITTI format
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for T in poses_out:
            row = T[:3, :].flatten()
            f.write(" ".join(f"{v:.6e}" for v in row) + "\n")

    print(f"Wrote {len(poses_out)} poses to {output_path}")

    if audit_out:
        write_audit(audit_out, poses_out, n_matched, max_dt, frame_tf_path)


def main():
    parser = argparse.ArgumentParser(description="Extract KITTI poses from ROS odom bag")
    parser.add_argument("--odom-bag", required=True, help="Recorded odometry bag")
    parser.add_argument("--odom-topic", required=True, help="Odometry topic name")
    parser.add_argument("--lidar-bag", required=True, help="Input KITTI bag (for timestamps)")
    parser.add_argument("--output", required=True, help="Output KITTI pose file")
    parser.add_argument("--max-dt", type=float, default=0.15, help="Max time offset (s)")
    parser.add_argument(
        "--frame-tf", type=str, default=None,
        help="Optional 4x4 SE(3) TXT to re-express trajectory (pose_new = T^-1 @ pose @ T)",
    )
    parser.add_argument(
        "--audit-out", type=str, default=None,
        help="Optional JSON path for frame/match diagnostic output",
    )
    args = parser.parse_args()

    extract(
        odom_bag_path=args.odom_bag,
        odom_topic=args.odom_topic,
        lidar_bag_path=args.lidar_bag,
        output_path=args.output,
        max_dt=args.max_dt,
        frame_tf_path=args.frame_tf,
        audit_out=args.audit_out,
    )


if __name__ == "__main__":
    main()
