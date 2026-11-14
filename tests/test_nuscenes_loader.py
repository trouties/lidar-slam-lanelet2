"""Tests for the NuScenesDataset loader.

Uses lightweight mocks so the test suite does not require nuscenes-devkit
or actual nuScenes data to run.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest  # noqa: F401 (used via pytest.approx, pytest.fixture)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_pcd_bin(path: Path, points: np.ndarray) -> None:
    """Write a nuScenes-format .pcd.bin file (N×5 float32)."""
    path.write_bytes(points.astype(np.float32).tobytes())


def _make_se3(tx: float = 0.0, ty: float = 0.0, tz: float = 0.0) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = [tx, ty, tz]
    return T


def _quaternion_identity() -> list[float]:
    return [1.0, 0.0, 0.0, 0.0]  # w, x, y, z


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def temp_pcd_dir(tmp_path: Path) -> Path:
    """Create three synthetic .pcd.bin files and return the directory."""
    for i in range(3):
        pts = np.zeros((100, 5), dtype=np.float32)
        pts[:, 0] = np.linspace(1, 10, 100)  # x
        pts[:, 3] = 128.0  # intensity (mid-range)
        pts[:, 4] = float(i % 32)  # ring index
        _write_pcd_bin(tmp_path / f"frame_{i:06d}.pcd.bin", pts)
    return tmp_path


@pytest.fixture()
def mock_nusc(temp_pcd_dir: Path):
    """Build a minimal mock NuScenes instance for 3 keyframes in one scene."""
    nusc = MagicMock()
    nusc.dataroot = str(temp_pcd_dir.parent)

    files = sorted(temp_pcd_dir.iterdir())

    def make_sample_data(idx: int) -> dict:
        rel = str(files[idx].relative_to(temp_pcd_dir.parent))
        return {
            "filename": rel,
            "ego_pose_token": f"ep_{idx}",
            "calibrated_sensor_token": "cs_0",
            "timestamp": 1_000_000 * idx,  # microseconds
            "next": f"sd_{idx + 1}" if idx < 2 else "",  # sweep chain
        }

    sample_data_records = {f"sd_{i}": make_sample_data(i) for i in range(3)}

    def _get(table: str, token: str) -> dict:
        if table == "scene":
            return {
                "token": "scene_0",
                "first_sample_token": "samp_0",
                "name": "test_scene",
            }
        if table == "sample":
            idx = int(token.split("_")[1])
            return {
                "token": token,
                "data": {"LIDAR_TOP": f"sd_{idx}"},
                "next": f"samp_{idx + 1}" if idx < 2 else "",
            }
        if table == "sample_data":
            return sample_data_records[token]
        if table == "ego_pose":
            idx = int(token.split("_")[1])
            return {
                "rotation": _quaternion_identity(),
                "translation": [float(idx), 0.0, 0.0],  # moves 1 m/frame in x
            }
        if table == "calibrated_sensor":
            return {
                "rotation": _quaternion_identity(),
                "translation": [0.0, 0.0, 0.0],  # sensor at vehicle origin
            }
        raise KeyError(f"Unknown table: {table}/{token}")

    nusc.get = _get
    return nusc


# ---------------------------------------------------------------------------
# Tests: NuScenesDataset
# ---------------------------------------------------------------------------


def test_import():
    """nuscenes_loader imports without errors."""
    from src.data.nuscenes_loader import NuScenesDataset  # noqa: F401


def test_len(mock_nusc):
    from src.data.nuscenes_loader import NuScenesDataset

    ds = NuScenesDataset(mock_nusc, "scene_0")
    assert len(ds) == 3


def test_getitem_pointcloud_shape(mock_nusc):
    from src.data.nuscenes_loader import NuScenesDataset

    ds = NuScenesDataset(mock_nusc, "scene_0")
    pc, pose, ts = ds[0]
    assert pc.ndim == 2
    assert pc.shape[1] == 4, "Must drop ring column → 4 columns"
    assert pc.dtype == np.float32


def test_intensity_normalised(mock_nusc):
    """Intensity column must be normalised from [0,255] to [0,1]."""
    from src.data.nuscenes_loader import NuScenesDataset

    ds = NuScenesDataset(mock_nusc, "scene_0")
    pc, _, _ = ds[0]
    assert pc[:, 3].max() <= 1.0, "Intensity should be normalised to [0,1]"
    # fixture writes 128 → should be ~0.502
    np.testing.assert_allclose(pc[:, 3], 128.0 / 255.0, atol=1e-5)


def test_first_pose_is_identity(mock_nusc):
    """Pose 0 must be identity (all subsequent poses are relative to it)."""
    from src.data.nuscenes_loader import NuScenesDataset

    ds = NuScenesDataset(mock_nusc, "scene_0")
    _, pose0, _ = ds[0]
    pose0 = np.asarray(pose0)
    np.testing.assert_allclose(pose0, np.eye(4), atol=1e-9)


def test_poses_shape(mock_nusc):
    from src.data.nuscenes_loader import NuScenesDataset

    ds = NuScenesDataset(mock_nusc, "scene_0")
    assert ds.poses.shape == (3, 4, 4)


def test_pose_translation_increases(mock_nusc):
    """Each frame moves 1 m in x (from mock ego_pose); sensor at origin → same delta."""
    from src.data.nuscenes_loader import NuScenesDataset

    ds = NuScenesDataset(mock_nusc, "scene_0")
    # frame 1 should be ~1 m ahead of frame 0 in x
    tx = ds.poses[1, 0, 3]
    np.testing.assert_allclose(tx, 1.0, atol=1e-6)


def test_timestamps_seconds(mock_nusc):
    """Timestamps must be in seconds and start at 0."""
    from src.data.nuscenes_loader import NuScenesDataset

    ds = NuScenesDataset(mock_nusc, "scene_0")
    assert ds.timestamps[0] == pytest.approx(0.0)
    assert ds.timestamps[1] == pytest.approx(1.0)
    assert ds.timestamps[2] == pytest.approx(2.0)


def test_calibration_key(mock_nusc):
    """calibration dict must have 'Tr' key set to identity."""
    from src.data.nuscenes_loader import NuScenesDataset

    ds = NuScenesDataset(mock_nusc, "scene_0")
    assert "Tr" in ds.calibration
    np.testing.assert_array_equal(ds.calibration["Tr"], np.eye(4))


def test_quat_trans_to_se3_identity():
    """Identity quaternion + zero translation → identity matrix."""
    from src.data.nuscenes_loader import _quat_trans_to_se3

    T = _quat_trans_to_se3([1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(T, np.eye(4), atol=1e-9)


def test_quat_trans_to_se3_translation():
    """Pure translation is preserved."""
    from src.data.nuscenes_loader import _quat_trans_to_se3

    T = _quat_trans_to_se3([1.0, 0.0, 0.0, 0.0], [3.0, 4.0, 5.0])
    np.testing.assert_allclose(T[:3, 3], [3.0, 4.0, 5.0], atol=1e-9)
    np.testing.assert_allclose(T[:3, :3], np.eye(3), atol=1e-9)
