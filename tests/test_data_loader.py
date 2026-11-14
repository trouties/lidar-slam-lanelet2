"""Tests for data loading module — all use synthetic data, no real KITTI dependency."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.kitti_loader import (
    KITTI_VERTICAL_ANGLE_OFFSET_DEG,
    KITTIDataset,
    correct_kitti_scan,
    load_calibration,
    load_poses,
    load_timestamps,
    load_velodyne_bin,
)
from src.data.transforms import apply_transform

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def kitti_root(tmp_path):
    """Create a minimal synthetic KITTI odometry directory structure."""
    seq_dir = tmp_path / "sequences" / "00"
    vel_dir = seq_dir / "velodyne"
    vel_dir.mkdir(parents=True)
    poses_dir = tmp_path / "poses"
    poses_dir.mkdir()

    # 3 small point clouds (5 points each)
    for i in range(3):
        pts = np.array(
            [[float(i), 1.0, 2.0, 0.5], [3.0, 4.0, 5.0, 0.9]],
            dtype=np.float32,
        )
        pts.tofile(vel_dir / f"{i:06d}.bin")

    # calib.txt
    with open(seq_dir / "calib.txt", "w") as f:
        # P0: identity-like 3x4
        f.write("P0: 1 0 0 0 0 1 0 0 0 0 1 0\n")
        f.write("P1: 1 0 0 -0.5 0 1 0 0 0 0 1 0\n")
        f.write("P2: 1 0 0 0 0 1 0 0 0 0 1 0\n")
        f.write("P3: 1 0 0 -0.5 0 1 0 0 0 0 1 0\n")
        f.write("Tr: 0 -1 0 0 0 0 -1 0 1 0 0 0\n")

    # times.txt
    with open(seq_dir / "times.txt", "w") as f:
        f.write("0.000000\n0.100000\n0.200000\n")

    # poses/00.txt (identity poses)
    with open(poses_dir / "00.txt", "w") as f:
        f.write("1 0 0 0 0 1 0 0 0 0 1 0\n")
        f.write("1 0 0 1 0 1 0 0 0 0 1 0\n")
        f.write("1 0 0 2 0 1 0 0 0 0 1 0\n")

    return tmp_path


# ---------------------------------------------------------------------------
# Tests: transforms (existing, kept)
# ---------------------------------------------------------------------------


def test_apply_transform_identity():
    points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    T = np.eye(4)
    result = apply_transform(points, T)
    np.testing.assert_array_almost_equal(result, points)


# ---------------------------------------------------------------------------
# Tests: load_velodyne_bin
# ---------------------------------------------------------------------------


def test_load_velodyne_bin(tmp_path):
    pts = np.array([[1.0, 2.0, 3.0, 0.5], [4.0, 5.0, 6.0, 0.9]], dtype=np.float32)
    path = tmp_path / "test.bin"
    pts.tofile(path)

    loaded = load_velodyne_bin(path)
    assert loaded.shape == (2, 4)
    np.testing.assert_array_almost_equal(loaded, pts)


def test_load_velodyne_bin_empty(tmp_path):
    path = tmp_path / "empty.bin"
    path.write_bytes(b"")

    loaded = load_velodyne_bin(path)
    assert loaded.shape == (0, 4)


# ---------------------------------------------------------------------------
# Tests: load_calibration
# ---------------------------------------------------------------------------


def test_load_calibration_keys_and_shape(kitti_root):
    calib = load_calibration(kitti_root / "sequences" / "00" / "calib.txt")
    assert set(calib.keys()) == {"P0", "P1", "P2", "P3", "Tr"}
    for mat in calib.values():
        assert mat.shape == (4, 4)
        np.testing.assert_array_equal(mat[3, :], [0, 0, 0, 1])


def test_load_calibration_values(kitti_root):
    calib = load_calibration(kitti_root / "sequences" / "00" / "calib.txt")
    # P0 should be identity
    np.testing.assert_array_almost_equal(calib["P0"], np.eye(4))
    # Tr: rotation that maps x->z, y->-x, z->-y
    expected_tr = np.array(
        [
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float64,
    )
    np.testing.assert_array_almost_equal(calib["Tr"], expected_tr)


# ---------------------------------------------------------------------------
# Tests: load_poses
# ---------------------------------------------------------------------------


def test_load_poses_shape(kitti_root):
    poses = load_poses(kitti_root / "poses" / "00.txt")
    assert poses.shape == (3, 4, 4)
    for i in range(3):
        np.testing.assert_array_equal(poses[i, 3, :], [0, 0, 0, 1])


def test_load_poses_values(kitti_root):
    poses = load_poses(kitti_root / "poses" / "00.txt")
    # First pose: identity
    np.testing.assert_array_almost_equal(poses[0], np.eye(4))
    # Second pose: translated by [1, 0, 0]
    assert poses[1, 0, 3] == 1.0
    # Third pose: translated by [2, 0, 0]
    assert poses[2, 0, 3] == 2.0


def test_load_poses_single_line(tmp_path):
    path = tmp_path / "single.txt"
    path.write_text("1 0 0 0 0 1 0 0 0 0 1 0\n")
    poses = load_poses(path)
    assert poses.shape == (1, 4, 4)
    np.testing.assert_array_almost_equal(poses[0], np.eye(4))


# ---------------------------------------------------------------------------
# Tests: load_timestamps
# ---------------------------------------------------------------------------


def test_load_timestamps(kitti_root):
    ts = load_timestamps(kitti_root / "sequences" / "00" / "times.txt")
    assert ts.shape == (3,)
    np.testing.assert_array_almost_equal(ts, [0.0, 0.1, 0.2])


def test_load_timestamps_single(tmp_path):
    path = tmp_path / "times.txt"
    path.write_text("0.5\n")
    ts = load_timestamps(path)
    assert ts.shape == (1,)
    assert ts[0] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Tests: KITTIDataset
# ---------------------------------------------------------------------------


def test_dataset_len(kitti_root):
    ds = KITTIDataset(kitti_root, sequence="00")
    assert len(ds) == 3


def test_dataset_getitem_with_poses(kitti_root):
    ds = KITTIDataset(kitti_root, sequence="00")
    pc, pose, ts = ds[0]
    assert pc.shape == (2, 4)
    assert pose is not None
    assert pose.shape == (4, 4)
    assert ts is not None
    assert isinstance(ts, float)


def test_dataset_getitem_no_poses(tmp_path):
    """Simulate sequence 11-21: no ground truth poses."""
    seq_dir = tmp_path / "sequences" / "11" / "velodyne"
    seq_dir.mkdir(parents=True)
    pts = np.array([[1.0, 2.0, 3.0, 0.5]], dtype=np.float32)
    pts.tofile(seq_dir / "000000.bin")

    ds = KITTIDataset(tmp_path, sequence="11")
    pc, pose, ts = ds[0]
    assert pc.shape == (1, 4)
    assert pose is None
    assert ts is None


def test_dataset_calibration_loaded(kitti_root):
    ds = KITTIDataset(kitti_root, sequence="00")
    assert ds.calibration is not None
    assert "Tr" in ds.calibration


def test_dataset_empty_velodyne(tmp_path):
    vel_dir = tmp_path / "sequences" / "00" / "velodyne"
    vel_dir.mkdir(parents=True)
    ds = KITTIDataset(tmp_path, sequence="00")
    assert len(ds) == 0


def test_dataset_missing_velodyne(tmp_path):
    (tmp_path / "sequences" / "00").mkdir(parents=True)
    ds = KITTIDataset(tmp_path, sequence="00")
    assert len(ds) == 0


# ---------------------------------------------------------------------------
# Tests: KITTI scan correction
# ---------------------------------------------------------------------------

_SCAN = np.array(
    [
        [10.0, 0.0, 0.0, 0.1],
        [10.0, 10.0, -1.7, 0.2],
        [30.0, 0.0, -1.73, 0.3],
        [5.0, -3.0, 2.0, 0.4],
        [-40.0, 25.0, -1.8, 0.5],
    ],
    dtype=np.float32,
)


def _spherical(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xyz = xyz.astype(np.float64)
    r_xy = np.hypot(xyz[:, 0], xyz[:, 1])
    return (
        np.linalg.norm(xyz, axis=1),
        np.arctan2(xyz[:, 1], xyz[:, 0]),
        np.degrees(np.arctan2(xyz[:, 2], r_xy)),
    )


def test_correct_kitti_scan_raises_elevation_only():
    out = correct_kitti_scan(_SCAN)
    assert out.shape == _SCAN.shape and out.dtype == _SCAN.dtype
    rng0, az0, el0 = _spherical(_SCAN[:, :3])
    rng1, az1, el1 = _spherical(out[:, :3])
    np.testing.assert_allclose(rng1, rng0, rtol=1e-6)
    np.testing.assert_allclose(az1, az0, atol=1e-6)
    np.testing.assert_allclose(el1 - el0, KITTI_VERTICAL_ANGLE_OFFSET_DEG, atol=1e-4)
    np.testing.assert_array_equal(out[:, 3], _SCAN[:, 3])


def test_correct_kitti_scan_matches_kiss_icp():
    pybind = pytest.importorskip("kiss_icp.pybind.kiss_icp_pybind")
    xyz = _SCAN[:, :3].astype(np.float64)
    reference = np.asarray(pybind._correct_kitti_scan(pybind._Vector3dVector(xyz)))
    np.testing.assert_allclose(correct_kitti_scan(xyz), reference, atol=1e-9)


def test_dataset_applies_correction_when_enabled(kitti_root):
    raw = KITTIDataset(kitti_root, sequence="00")[0][0]
    corrected = KITTIDataset(kitti_root, sequence="00", correct_scan=True)[0][0]
    np.testing.assert_allclose(corrected, correct_kitti_scan(raw))
    cfg = {"kitti_root": str(kitti_root), "correct_kitti_scan": True}
    from_cfg = KITTIDataset.from_config(cfg, "00")[0][0]
    np.testing.assert_array_equal(from_cfg, corrected)
    assert not KITTIDataset.from_config({"kitti_root": str(kitti_root)}, "00").correct_scan
