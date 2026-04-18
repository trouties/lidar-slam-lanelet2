"""Smoke tests for OxTS → body-frame IMU conversion.

Run: cd external/baselines/common && python3 -m pytest test_kitti_to_rosbag.py -v

"""

from __future__ import annotations

import struct
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

# Mock ROS modules so kitti_to_rosbag / fix_imu_timestamps can import without ROS installed
for mod in [
    "rosbag",
    "rospy",
    "sensor_msgs",
    "sensor_msgs.msg",
    "std_msgs",
    "std_msgs.msg",
    "geometry_msgs",
    "geometry_msgs.msg",
]:
    sys.modules.setdefault(mod, MagicMock())


# PointField side_effect captures kwargs so PC2_FIELDS_RING is introspectable
def _mk_pointfield(*, name, offset, datatype, count=1):
    return types.SimpleNamespace(name=name, offset=offset, datatype=datatype, count=count)


sys.modules["sensor_msgs.msg"].PointField = MagicMock(
    side_effect=_mk_pointfield, FLOAT32=7, UINT16=4
)

sys.path.insert(0, str(Path(__file__).parent))
from fix_imu_timestamps import _enforce_monotonic_ns  # noqa: E402
from inject_kitti_extrinsic import inject, load_calib_imu_to_velo  # noqa: E402
from kitti_to_rosbag import (  # noqa: E402
    PC2_FIELDS_RING,
    _compute_point_time_offset,
    _oxts_body_accel,
    _oxts_body_gyro,
    _oxts_nav_accel,
)


def _row(
    roll=0.0,
    pitch=0.0,
    yaw=0.0,
    ax=0.0,
    ay=0.0,
    az=0.0,
    af=0.0,
    al=0.0,
    au=0.0,
    wf=0.0,
    wl=0.0,
    wu=0.0,
):
    r = np.zeros(30, dtype=np.float64)
    r[3], r[4], r[5] = roll, pitch, yaw
    r[11], r[12], r[13] = ax, ay, az
    r[14], r[15], r[16] = af, al, au
    r[20], r[21], r[22] = wf, wl, wu
    return r


# ---------------------------------------------------------------------------
# IMU body-frame passthrough (cols 14-16, 20-22)
# ---------------------------------------------------------------------------


def test_accel_passthrough_from_cols_14_15_16():
    """af/al/au are body-frame specific force; we pass them through unchanged."""
    ax, ay, az = _oxts_body_accel(_row(af=0.6, al=0.1, au=9.82))
    assert ax == pytest.approx(0.6, abs=1e-9)
    assert ay == pytest.approx(0.1, abs=1e-9)
    assert az == pytest.approx(9.82, abs=1e-9)


def test_accel_ignores_roll_pitch_yaw():
    """No gravity-reinjection math: attitude angles must not leak into accel."""
    ax0, ay0, az0 = _oxts_body_accel(_row(af=1.0, al=2.0, au=9.8))
    ax1, ay1, az1 = _oxts_body_accel(
        _row(
            roll=np.deg2rad(30),
            pitch=np.deg2rad(-15),
            yaw=np.deg2rad(45),
            af=1.0,
            al=2.0,
            au=9.8,
        )
    )
    assert (ax0, ay0, az0) == (ax1, ay1, az1)


def test_accel_stationary_level_reports_positive_g():
    """Empirical KITTI OxTS: stationary level sensor reports au ~ +9.8."""
    ax, ay, az = _oxts_body_accel(_row(af=0.0, al=0.0, au=9.8205))
    assert abs(ax) < 1e-9
    assert abs(ay) < 1e-9
    assert az == pytest.approx(9.8205, abs=1e-9)


def test_gyro_reads_body_columns():
    """Body-frame gyro uses cols 20-22 (wf/wl/wu), not 17-19."""
    wx, wy, wz = _oxts_body_gyro(_row(wf=0.1, wl=0.2, wu=0.3))
    assert (wx, wy, wz) == (0.1, 0.2, 0.3)


def test_gyro_ignores_nav_frame_columns():
    """cols 17-19 (wx/wy/wz, nav frame) must not be read as body gyro."""
    row = _row(wf=0.1, wl=0.2, wu=0.3)
    row[17], row[18], row[19] = 99.0, 99.0, 99.0  # garbage nav-frame values
    wx, wy, wz = _oxts_body_gyro(row)
    assert (wx, wy, wz) == (0.1, 0.2, 0.3)


# ---------------------------------------------------------------------------
# IMU nav-frame fallback (SUP-01 α: LIO-SAM-only, cols 11-13)
# ---------------------------------------------------------------------------


def test_nav_accel_reads_cols_11_12_13():
    """ax/ay/az are nav-frame kinematic accel (gravity pre-subtracted)."""
    ax, ay, az = _oxts_nav_accel(_row(ax=0.1, ay=0.2, az=0.3))
    assert (ax, ay, az) == (0.1, 0.2, 0.3)


def test_nav_accel_differs_from_body_accel():
    """Cols 11-13 (nav) and 14-16 (body) must read DIFFERENT columns.

    If the two functions ever alias to the same columns, the α fallback
    silently reverts to body-frame and LIO-SAM regresses to ~1076 m.
    """
    # row with DISTINCT nav / body values
    row = _row(ax=1.0, ay=2.0, az=3.0, af=4.0, al=5.0, au=6.0)
    assert _oxts_nav_accel(row) == (1.0, 2.0, 3.0)
    assert _oxts_body_accel(row) == (4.0, 5.0, 6.0)


def test_nav_accel_stationary_is_near_zero():
    """Nav-frame cols 11-13 are kinematic (gravity removed): stationary ≈ 0.

    Complement to ``test_accel_stationary_level_reports_positive_g``; the two
    assertions together pin down the physical interpretation of each column.
    """
    ax, ay, az = _oxts_nav_accel(_row(ax=0.01, ay=-0.02, az=0.03))
    assert abs(ax) < 0.05
    assert abs(ay) < 0.05
    assert abs(az) < 0.05


# ---------------------------------------------------------------------------
# Extrinsic injection — Bug A regression guard (the Rot_y(π) reflection bug)
# ---------------------------------------------------------------------------


def _write_calib(tmp_path: Path, R_flat: list[float], T: list[float]) -> Path:
    """Write a synthetic calib_imu_to_velo.txt to tmp_path and return its path."""
    p = tmp_path / "calib_imu_to_velo.txt"
    p.write_text(
        f"calib_time: synthetic\n"
        f"R: {' '.join(f'{x:.9f}' for x in R_flat)}\n"
        f"T: {' '.join(f'{x:.9f}' for x in T)}\n"
    )
    return p


def test_inject_rejects_reflection_matrix(tmp_path):
    """det(R) = -1 (true reflection) must raise on the det check."""
    # diag(-1, 1, 1): det = -1, trace = 1 → det check fires first
    R_refl = [-1, 0, 0, 0, 1, 0, 0, 0, 1]
    calib = _write_calib(tmp_path, R_refl, [0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="proper rotation|det"):
        load_calib_imu_to_velo(calib)


def test_inject_rejects_bug_a_roty_pi_matrix(tmp_path):
    """The Bug A matrix Rot_y(π) = diag(-1, 1, -1) has det=+1 but trace=-1.

    It's technically a rotation (180° about y), but differs from the true
    KITTI extrinsic by 180°. KITTI IMU and Velo are within ~1° of each other,
    so trace ≈ 3; anything with trace far from 3 is a wrong extrinsic.
    This is the direct regression guard for SUP-01 Bug A.
    """
    R_bug_a = [-1, 0, 0, 0, 1, 0, 0, 0, -1]  # det = +1, trace = -1
    calib = _write_calib(tmp_path, R_bug_a, [0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="trace"):
        load_calib_imu_to_velo(calib)


def test_inject_rejects_low_trace_matrix(tmp_path):
    """trace(R) far from 3 must raise — KITTI IMU/Velo are nearly aligned."""
    # 90° rotation about z: det = +1 but trace = 1 (far from 3)
    R_rot90 = [0, -1, 0, 1, 0, 0, 0, 0, 1]
    calib = _write_calib(tmp_path, R_rot90, [0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="trace"):
        load_calib_imu_to_velo(calib)


def test_inject_yaml_roundtrip(tmp_path):
    """Inject into a params.yaml template, re-parse, assert Rot==RPY==R exactly."""
    params = tmp_path / "params.yaml"
    params.write_text(
        "# LIO-SAM params fixture\n"
        "extrinsicTrans: [0.0, 0.0, 0.0]\n"
        "extrinsicRot: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]\n"
        "extrinsicRPY: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]\n"
        "pointCloudTopic: /velodyne_points\n"
    )
    # Small-angle rotation (0.1 rad about z) — well within KITTI trace/det bounds
    theta = 0.05
    R = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    T = np.array([-0.808, 0.319, -0.799])

    inject(params, R, T)

    text = params.read_text()
    import re

    m_rot = re.search(r"extrinsicRot:\s*\[([^\]]+)\]", text)
    m_rpy = re.search(r"extrinsicRPY:\s*\[([^\]]+)\]", text)
    m_tr = re.search(r"extrinsicTrans:\s*\[([^\]]+)\]", text)
    assert m_rot and m_rpy and m_tr, "All three extrinsic fields must survive injection"

    rot_vals = [float(x) for x in m_rot.group(1).split(",")]
    rpy_vals = [float(x) for x in m_rpy.group(1).split(",")]
    tr_vals = [float(x) for x in m_tr.group(1).split(",")]

    assert rot_vals == rpy_vals, "extrinsicRPY must equal extrinsicRot byte-for-byte"
    assert np.allclose(np.array(rot_vals).reshape(3, 3), R, atol=1e-9)
    assert np.allclose(tr_vals, T, atol=1e-9)
    # Non-extrinsic lines preserved
    assert "pointCloudTopic: /velodyne_points" in text


def test_inject_kitti_real_seq00_calib():
    """Real KITTI Raw 2011_10_03 calib must load with det≈1, trace≈3, T in plausible range.

    Skipped when KITTI Raw is not mounted locally.
    """
    calib = Path.home() / "data" / "kitti_raw" / "2011_10_03" / "calib_imu_to_velo.txt"
    if not calib.exists():
        pytest.skip(f"KITTI Raw calib not found at {calib}")
    R, T = load_calib_imu_to_velo(calib)
    assert R.shape == (3, 3)
    assert T.shape == (3,)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-3)
    assert np.trace(R) == pytest.approx(3.0, abs=0.05)
    # KITTI IMU-to-Velo translation magnitude is ~1 m (bolted on the roof)
    assert 0.3 < np.linalg.norm(T) < 2.5


# ---------------------------------------------------------------------------
# Per-point time field (LIO-SAM deskew)
# ---------------------------------------------------------------------------


def test_per_point_time_monotonic_from_azimuth():
    """time ∈ [0, scan_period] and increases with azimuth (wraps -π→π correctly)."""
    # Points at 8 evenly-spaced azimuths around the sensor at radius 10m
    angles = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    # arctan2(y, x): azimuth = angle. Build points at radius 10, z=0.
    pts = np.zeros((len(angles), 4), dtype=np.float32)
    pts[:, 0] = 10.0 * np.cos(angles)
    pts[:, 1] = 10.0 * np.sin(angles)

    times = _compute_point_time_offset(pts, scan_period=0.1)
    assert times.dtype == np.float32
    assert times.min() >= 0.0
    assert times.max() <= 0.1 + 1e-6
    # Azimuth 0 → time 0; azimuth π/2 → 0.025; π → 0.05; 3π/2 → 0.075
    assert times[0] == pytest.approx(0.0, abs=1e-6)
    assert times[2] == pytest.approx(0.025, abs=1e-6)
    assert times[4] == pytest.approx(0.05, abs=1e-6)
    assert times[6] == pytest.approx(0.075, abs=1e-6)


def test_per_point_time_default_scan_period_is_10hz():
    """Default scan_period=0.1 s matches HDL-64E 10 Hz rotation."""
    pts = np.array([[10.0, 0.0, 0.0, 0.0], [-10.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    # arctan2(0, 10) = 0 → time 0; arctan2(0, -10) = π → time 0.05
    times = _compute_point_time_offset(pts)  # default scan_period
    assert times[0] == pytest.approx(0.0, abs=1e-6)
    assert times[1] == pytest.approx(0.05, abs=1e-6)


def test_per_point_time_zero_mode_forces_all_zeros():
    """zero_time=True disables the azimuth-based deskew compensation.

    KITTI scans are pre-deskewed so non-zero times cause LIO-SAM double-
    correction. The zero-time bag variant is the α+ LIO-SAM parity path
    for the stage0-v3 baseline.
    """
    angles = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    pts = np.zeros((len(angles), 4), dtype=np.float32)
    pts[:, 0] = 10.0 * np.cos(angles)
    pts[:, 1] = 10.0 * np.sin(angles)
    times = _compute_point_time_offset(pts, zero_time=True)
    assert times.dtype == np.float32
    assert np.all(times == 0.0)


# ---------------------------------------------------------------------------
# PointCloud2 field layout for LIO-SAM mode
# ---------------------------------------------------------------------------


def test_pointcloud2_liosam_mode_has_ring_and_time():
    """LIO-SAM's imageProjection requires ring (UINT16) + time (FLOAT32)."""
    names = [f.name for f in PC2_FIELDS_RING]
    assert names == ["x", "y", "z", "intensity", "ring", "time"]
    offsets = {f.name: f.offset for f in PC2_FIELDS_RING}
    assert offsets == {"x": 0, "y": 4, "z": 8, "intensity": 12, "ring": 16, "time": 18}
    dtypes = {f.name: f.datatype for f in PC2_FIELDS_RING}
    # FLOAT32 = 7, UINT16 = 4 per our mock (matching real PointField constants)
    assert dtypes["ring"] == 4  # UINT16
    assert dtypes["time"] == 7  # FLOAT32
    assert dtypes["x"] == dtypes["y"] == dtypes["z"] == dtypes["intensity"] == 7  # FLOAT32


def test_pointcloud2_point_step_matches_struct_pack():
    """point_step == 22 must equal actual struct.pack bytes (<ffffHf)."""
    # offset 18 (after ring at [16..18]) + 4 (time f32) = 22 bytes per point
    packed = struct.pack("<ffffHf", 1.0, 2.0, 3.0, 0.5, 42, 0.01)
    assert len(packed) == 22


# ---------------------------------------------------------------------------
# IMU timestamp monotonicity fix (GTSAM preintegration crash guard)
# ---------------------------------------------------------------------------


def test_fix_imu_timestamps_first_sample_passes_through():
    """prev_ns=None: the first sample must keep its original stamp unchanged."""
    assert _enforce_monotonic_ns(None, 1_000_000_000, eps_ns=1000) == 1_000_000_000


def test_fix_imu_timestamps_forward_sample_passthrough():
    """When orig is already ahead of prev+eps, return orig unchanged."""
    prev = 1_000_000_000
    orig = prev + 10_000_000  # 10 ms ahead
    assert _enforce_monotonic_ns(prev, orig, eps_ns=1000) == orig


def test_fix_imu_timestamps_duplicate_bumped_by_eps():
    """Duplicate stamp (equals prev) must be bumped to prev + eps_ns."""
    prev = 1_000_000_000
    orig = 1_000_000_000  # identical to prev
    out = _enforce_monotonic_ns(prev, orig, eps_ns=1000)
    assert out == prev + 1000


def test_fix_imu_timestamps_backstep_bumped_by_eps():
    """Backwards stamp (< prev) must be bumped to prev + eps_ns, not kept."""
    prev = 2_000_000_000
    orig = 1_500_000_000  # 500 ms backwards
    out = _enforce_monotonic_ns(prev, orig, eps_ns=1000)
    assert out == prev + 1000


def test_fix_imu_timestamps_sequence_is_strictly_monotonic():
    """Simulate a realistic OxTS sequence with duplicates; output is strictly monotonic."""
    raw = [
        1_000_000_000,  # first
        1_000_000_000,  # dup
        1_000_000_000,  # dup again
        1_010_000_000,  # jump forward
        1_009_999_999,  # backwards blip
        1_020_000_000,  # forward
    ]
    eps = 1000
    out: list[int] = []
    prev: int | None = None
    for ns in raw:
        rep = _enforce_monotonic_ns(prev, ns, eps)
        out.append(rep)
        prev = rep
    assert all(out[i + 1] - out[i] >= eps for i in range(len(out) - 1)), (
        f"non-monotonic output: {out}"
    )
    # First sample unchanged
    assert out[0] == raw[0]
    # Dups pushed by eps
    assert out[1] == raw[0] + eps
    assert out[2] == raw[0] + 2 * eps
    # Forward sample kept (already > prev + eps)
    assert out[3] == raw[3]
