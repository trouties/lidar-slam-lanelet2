#!/usr/bin/env python3
"""Convert KITTI Odometry velodyne scans (+ optional IMU) to a ROS bag.

Usage:
    # LiDAR only (for hdl_graph_slam)
    python3 kitti_to_rosbag.py --kitti-dir /data/kitti --sequence 00 \
        --output /tmp/kitti_00.bag --lidar-only

    # LiDAR + IMU (for LIO-SAM, FAST-LIO2)
    python3 kitti_to_rosbag.py --kitti-dir /data/kitti --sequence 00 \
        --output /tmp/kitti_00.bag --with-imu --kitti-raw-dir /data/kitti_raw
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

# ROS imports
import rosbag
import rospy
from geometry_msgs.msg import Quaternion, Vector3
from sensor_msgs.msg import Imu, PointCloud2, PointField
from std_msgs.msg import Header

# KITTI Odometry → Raw drive mapping (official correspondence)
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

# Fixed epoch base for consistent ROS timestamps
EPOCH_BASE = 1317648000.0  # ~2011-10-03 12:00:00 UTC

# PointCloud2 field definitions — basic (x, y, z, intensity)
PC2_FIELDS_BASIC = [
    PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
]

# PointCloud2 field definitions — with ring + time (required by LIO-SAM)
PC2_FIELDS_RING = [
    PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
    PointField(name="ring", offset=16, datatype=PointField.UINT16, count=1),
    PointField(name="time", offset=18, datatype=PointField.FLOAT32, count=1),
]

# HDL-64E vertical angle range: approx -24.8° to +2.0°
_HDL64E_VFOV_MIN = np.deg2rad(-24.8)
_HDL64E_VFOV_MAX = np.deg2rad(2.0)
_HDL64E_N_SCAN = 64


def load_timestamps(times_path: Path) -> np.ndarray:
    """Load KITTI timestamps file (one float per line, seconds from start)."""
    return np.loadtxt(str(times_path), dtype=np.float64)


def load_velodyne_bin(bin_path: Path) -> np.ndarray:
    """Load a KITTI velodyne .bin file as (N, 4) float32 [x, y, z, reflectance]."""
    return np.fromfile(str(bin_path), dtype=np.float32).reshape(-1, 4)


def _compute_ring(points: np.ndarray) -> np.ndarray:
    """Compute ring (laser id) for HDL-64E from vertical angle."""
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    dist_xy = np.sqrt(x * x + y * y)
    vert_angle = np.arctan2(z, dist_xy)
    # Map vertical angle linearly to ring 0..63
    frac = (vert_angle - _HDL64E_VFOV_MIN) / (_HDL64E_VFOV_MAX - _HDL64E_VFOV_MIN)
    ring = np.clip(np.round(frac * (_HDL64E_N_SCAN - 1)), 0, _HDL64E_N_SCAN - 1)
    return ring.astype(np.uint16)


def _compute_point_time_offset(
    points: np.ndarray, scan_period: float = 0.1, zero_time: bool = False
) -> np.ndarray:
    """Compute per-point ``time`` field from azimuth for HDL-64E scan deskewing.

    LIO-SAM's imageProjection node uses this ``time`` field (seconds from the
    scan start) to compensate for vehicle motion within a single LiDAR
    revolution. KITTI ``.bin`` files do not carry per-point time, so we
    reconstruct it from the azimuth assuming a uniform 10 Hz rotation
    (``scan_period = 0.1 s``) starting at +X (forward). Points at azimuth 0
    get ``time = 0``; points at 2*pi get ``time = scan_period``.

    When ``zero_time=True``, return all zeros. KITTI Odometry scans are
    already deskewed by the dataset authors; feeding azimuth-based times to
    LIO-SAM triggers double-compensation that empirically worsens Seq 00
    APE by ~20x (α ablation, SUP-01 R2 follow-up). Historical stage0-v3
    baseline (SE(3) APE = 27m) was built with all-zero times.
    """
    if zero_time:
        return np.zeros(points.shape[0], dtype=np.float32)
    x = points[:, 0]
    y = points[:, 1]
    azimuth = np.arctan2(y, x)  # [-pi, pi]
    azimuth = np.where(azimuth < 0, azimuth + 2.0 * np.pi, azimuth)
    return (azimuth / (2.0 * np.pi) * scan_period).astype(np.float32)


def make_pointcloud2(
    points: np.ndarray,
    stamp: rospy.Time,
    frame_id: str,
    add_ring: bool = False,
    zero_time: bool = False,
) -> PointCloud2:
    """Create a PointCloud2 message from (N, 4) float32 array.

    Args:
        add_ring: If True, append ring (uint16) and time (float32) fields.
            Required by LIO-SAM's imageProjection node.
        zero_time: When add_ring=True, force per-point time field to 0
            (disables LIO-SAM internal deskew). KITTI is pre-deskewed so
            azimuth-based times double-compensate; use zero_time for LIO-SAM
            parity with the historical stage0-v3 baseline.
    """
    msg = PointCloud2()
    msg.header = Header(stamp=stamp, frame_id=frame_id)
    msg.height = 1
    msg.width = points.shape[0]
    msg.is_bigendian = False
    msg.is_dense = True

    if add_ring:
        msg.fields = PC2_FIELDS_RING
        msg.point_step = 22  # 4*4 + 2 + 4 = 22 bytes
        ring = _compute_ring(points)
        time_offset = _compute_point_time_offset(points, zero_time=zero_time)
        # Pack: x(f32) y(f32) z(f32) intensity(f32) ring(u16) time(f32)
        buf = bytearray(msg.point_step * msg.width)
        pts = points.astype(np.float32)
        for i in range(msg.width):
            offset = i * msg.point_step
            struct.pack_into(
                "<ffffHf",
                buf,
                offset,
                pts[i, 0],
                pts[i, 1],
                pts[i, 2],
                pts[i, 3],
                ring[i],
                time_offset[i],
            )
        msg.data = bytes(buf)
    else:
        msg.fields = PC2_FIELDS_BASIC
        msg.point_step = 16
        msg.data = points.astype(np.float32).tobytes()

    msg.row_step = msg.point_step * msg.width
    return msg


def load_oxts_data(oxts_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load OxTS data and timestamps from KITTI Raw.

    Returns:
        (data, timestamps) where data is (N, 30) and timestamps is (N,)
        in seconds relative to first entry.
    """
    data_dir = oxts_dir / "data"
    ts_path = oxts_dir / "timestamps.txt"

    files = sorted(data_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No oxts data in {data_dir}")

    rows = [np.loadtxt(str(f)) for f in files]
    data = np.array(rows, dtype=np.float64)

    # Parse timestamps
    from datetime import datetime

    raw_ts = []
    with open(ts_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Format: 2011-10-03 12:55:34.123456789
            dt = datetime.strptime(line[:26], "%Y-%m-%d %H:%M:%S.%f")
            raw_ts.append(dt.timestamp())
    timestamps = np.array(raw_ts, dtype=np.float64)
    timestamps -= timestamps[0]  # relative to first frame
    return data, timestamps


KITTI_TO_ROSBAG_VERSION = "v4-accel-mode-switch"


def _oxts_body_accel(oxts_row: np.ndarray) -> tuple[float, float, float]:
    """Return body-frame accelerometer specific-force (OxTS cols 14-16).

    OxTS ``af/al/au`` are body-frame SPECIFIC FORCE (gravity already
    included), not kinematic acceleration: a stationary, level sensor
    measures ``au ~ +9.8 m/s^2``, as directly verified on KITTI Raw
    2011_10_03 drive 0027 (mean au = 9.82 over the first 200 frames).

    The Stage 3 plan's gravity-reinjection formula assumed OxTS cols 14-16
    were kinematic (gravity removed); that premise is empirically wrong for
    KITTI, so we pass the body-frame specific-force through unchanged.
    Combined with the stage0-v3 baseline config (useImuHeadingInit=false,
    extrinsicRPY = extrinsicRot = R_velo_imu, datasheet IMU sigmas), this
    is what LIO-SAM's ``PreintegratedImuMeasurements`` expects IN THEORY.

    EMPIRICAL CAVEAT (SUP-01 R1): passing body-frame specific force to
    LIO-SAM's GTSAM preintegration regresses Seq 00 APE by ~40x (27m →
    1076m) vs the old nav-frame path. R1 inflated σ to compensate, made
    it worse (6406m). The current hypothesis is that LIO-SAM's internal
    ``n_gravity`` sign convention plus its extrinsic-rotated accel path
    double-applies gravity when fed body-frame specific force, producing
    ~+2g spurious kinematic accel per step. See ``--accel-mode nav`` for
    the empirically-stable (though theoretically-impure) fallback.
    """
    return float(oxts_row[14]), float(oxts_row[15]), float(oxts_row[16])


def _oxts_nav_accel(oxts_row: np.ndarray) -> tuple[float, float, float]:
    """Return nav-frame accelerometer kinematic-accel (OxTS cols 11-13).

    Cols ``ax/ay/az`` are OxTS's pre-processed east/north/up accel with
    gravity already subtracted (RT3003 internal filter output). Feeding
    this to LIO-SAM skips the body-frame double-gravity trap and was the
    IMU content that produced the historical stage0-v3 SE(3) APE = 27 m.
    It is physically wrong for body-frame preintegration on a moving
    vehicle but empirically the path GTSAM is happy with on KITTI.
    SUP-01 α fallback (see refs/pipeline-notes.md §20).
    """
    return float(oxts_row[11]), float(oxts_row[12]), float(oxts_row[13])


def _oxts_body_gyro(oxts_row: np.ndarray) -> tuple[float, float, float]:
    """Return body-frame angular velocity (OxTS cols 20-22: wf/wl/wu).

    Previously cols 17-19 (nav-frame wx/wy/wz) were used, which is wrong for
    body-frame preintegration.
    """
    return float(oxts_row[20]), float(oxts_row[21]), float(oxts_row[22])


def make_imu_msg(
    oxts_row: np.ndarray,
    stamp: rospy.Time,
    frame_id: str = "imu_link",
    accel_mode: str = "body",
) -> Imu:
    """Create an Imu message from a single OxTS data row.

    Parameters
    ----------
    accel_mode
        ``"body"`` (default, physically-correct): cols 14-16 body-frame
        specific force including gravity. Matches pytest assertions.
        ``"nav"``: cols 11-13 nav-frame kinematic accel. Empirically
        stable for LIO-SAM; required as an α fallback until the body-frame
        → GTSAM double-gravity issue is resolved.
    """
    msg = Imu()
    msg.header = Header(stamp=stamp, frame_id=frame_id)

    wx, wy, wz = _oxts_body_gyro(oxts_row)
    msg.angular_velocity = Vector3(x=wx, y=wy, z=wz)

    if accel_mode == "nav":
        ax, ay, az = _oxts_nav_accel(oxts_row)
    elif accel_mode == "body":
        ax, ay, az = _oxts_body_accel(oxts_row)
    else:
        raise ValueError(f"accel_mode must be 'body' or 'nav', got {accel_mode!r}")
    msg.linear_acceleration = Vector3(x=ax, y=ay, z=az)

    # Orientation from roll/pitch/yaw (euler → quaternion)
    roll, pitch, yaw = float(oxts_row[3]), float(oxts_row[4]), float(oxts_row[5])
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    msg.orientation = Quaternion(
        x=sr * cp * cy - cr * sp * sy,
        y=cr * sp * cy + sr * cp * sy,
        z=cr * cp * sy - sr * sp * cy,
        w=cr * cp * cy + sr * sp * sy,
    )

    # Covariance: -1 means unknown (diagonal)
    msg.orientation_covariance = [0.0] * 9
    msg.angular_velocity_covariance = [0.0] * 9
    msg.linear_acceleration_covariance = [0.0] * 9

    return msg


def convert(
    kitti_dir: str,
    sequence: str,
    output: str,
    with_imu: bool = False,
    kitti_raw_dir: str | None = None,
    add_ring: bool = False,
    accel_mode: str = "body",
    zero_time: bool = False,
) -> None:
    """Convert a KITTI Odometry sequence to a ROS bag."""
    kitti_path = Path(kitti_dir)
    seq_dir = kitti_path / "sequences" / sequence
    velo_dir = seq_dir / "velodyne"
    times_path = seq_dir / "times.txt"

    if not velo_dir.exists():
        print(f"ERROR: Velodyne dir not found: {velo_dir}", file=sys.stderr)
        sys.exit(1)

    timestamps = load_timestamps(times_path)
    bin_files = sorted(velo_dir.glob("*.bin"))
    n_frames = min(len(bin_files), len(timestamps))
    print(f"Converting seq {sequence}: {n_frames} frames")

    # Load IMU if requested
    imu_data = None
    imu_ts = None
    if with_imu:
        if sequence not in ODOM_TO_RAW:
            print(f"WARNING: No Raw mapping for seq {sequence}, skipping IMU")
        elif kitti_raw_dir is None:
            print("WARNING: --kitti-raw-dir not provided, skipping IMU")
        else:
            date, drive = ODOM_TO_RAW[sequence]
            oxts_dir = Path(kitti_raw_dir) / date / f"{date}_drive_{drive}_extract" / "oxts"
            if oxts_dir.exists():
                imu_data, imu_ts = load_oxts_data(oxts_dir)
                print(f"  Loaded {len(imu_data)} IMU samples from {oxts_dir}")
            else:
                print(f"WARNING: OxTS dir not found: {oxts_dir}, skipping IMU")

    bag = rosbag.Bag(output, "w")
    try:
        imu_cursor = 0  # tracks which IMU samples have been written

        for i in range(n_frames):
            t_sec = EPOCH_BASE + timestamps[i]
            ros_time = rospy.Time.from_sec(t_sec)

            # Write IMU messages between previous and current LiDAR frame
            if imu_data is not None and imu_ts is not None:
                t_lidar_rel = timestamps[i]
                t_prev_rel = timestamps[i - 1] if i > 0 else -0.001

                while imu_cursor < len(imu_ts):
                    t_imu_rel = imu_ts[imu_cursor]
                    if t_imu_rel > t_lidar_rel + 0.001:
                        break  # past current LiDAR frame
                    if t_imu_rel >= t_prev_rel - 0.001:
                        imu_stamp = rospy.Time.from_sec(EPOCH_BASE + t_imu_rel)
                        imu_msg = make_imu_msg(
                            imu_data[imu_cursor], imu_stamp, accel_mode=accel_mode
                        )
                        bag.write("/imu/data", imu_msg, imu_stamp)
                    imu_cursor += 1

            # Write LiDAR point cloud
            points = load_velodyne_bin(bin_files[i])
            pc2_msg = make_pointcloud2(
                points, ros_time, "velodyne", add_ring=add_ring, zero_time=zero_time
            )
            bag.write("/velodyne_points", pc2_msg, ros_time)

            if (i + 1) % 500 == 0 or i == n_frames - 1:
                print(f"  {i + 1}/{n_frames} frames written")

    finally:
        bag.close()

    print(f"Bag written: {output}")


def main():
    parser = argparse.ArgumentParser(description="Convert KITTI Odometry to ROS bag")
    parser.add_argument("--kitti-dir", required=True, help="KITTI Odometry dataset root")
    parser.add_argument("--sequence", required=True, help="Sequence id (e.g. 00)")
    parser.add_argument("--output", required=True, help="Output bag path")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--lidar-only", action="store_true", help="Only velodyne points")
    mode.add_argument("--with-imu", action="store_true", help="Include OxTS IMU data")
    parser.add_argument("--kitti-raw-dir", help="KITTI Raw root (for IMU)")
    parser.add_argument(
        "--add-ring", action="store_true", help="Add ring + time fields (required by LIO-SAM)"
    )
    parser.add_argument(
        "--accel-mode",
        choices=("body", "nav"),
        default="body",
        help="body (cols 14-16, specific force) | nav (cols 11-13, kinematic accel). "
        "nav is the α fallback for LIO-SAM pending body-frame GTSAM integration fix.",
    )
    parser.add_argument(
        "--zero-time",
        action="store_true",
        help="Force per-point time field to 0 (disables LIO-SAM internal deskew). "
        "KITTI is pre-deskewed, so azimuth-based times double-compensate.",
    )
    args = parser.parse_args()

    convert(
        kitti_dir=args.kitti_dir,
        sequence=args.sequence,
        output=args.output,
        with_imu=args.with_imu,
        kitti_raw_dir=args.kitti_raw_dir,
        add_ring=args.add_ring,
        accel_mode=args.accel_mode,
        zero_time=args.zero_time,
    )


if __name__ == "__main__":
    main()
