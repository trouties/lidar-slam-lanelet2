"""Tests for the benchmarks infrastructure."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from src.benchmarks.evaluator import ape_rotation_about_start, load_poses_kitti_format
from src.benchmarks.timing import StageTimer


class TestStageTimer:
    def test_single_shot(self):
        with StageTimer("test") as t:
            time.sleep(0.01)
        s = t.summary()
        assert s["n"] == 1
        assert s["p50"] > 0

    def test_multi_lap(self):
        t = StageTimer("test")
        for _ in range(5):
            with t:
                time.sleep(0.001)
        s = t.summary()
        assert s["n"] == 5
        assert s["total_ms"] > 0

    def test_empty(self):
        t = StageTimer("test")
        s = t.summary()
        assert s["n"] == 0


class TestLoadPoses:
    def test_roundtrip(self, tmp_path: Path):
        poses = [np.eye(4), np.eye(4)]
        poses[1][:3, 3] = [1.0, 2.0, 3.0]
        path = tmp_path / "poses.txt"
        with path.open("w") as f:
            for p in poses:
                row = p[:3, :].flatten()
                f.write(" ".join(f"{v:.6e}" for v in row) + "\n")
        loaded = load_poses_kitti_format(path)
        assert len(loaded) == 2
        np.testing.assert_allclose(loaded[1][:3, 3], [1.0, 2.0, 3.0])


class TestCacheTimingRoundtrip:
    """Test that timing survives cache save/load cycle."""

    def test_get_stage_metrics(self, tmp_path: Path):
        from src.cache.layered_cache import LayeredCache

        cache = LayeredCache(root=str(tmp_path / "cache"), sequence="test")
        cfg = {"data": {"kitti_root": "/tmp"}, "kiss_icp": {"voxel_size": 1.0}}

        # Save odometry with timing in metrics
        poses = np.eye(4).reshape(1, 4, 4)
        ts = np.array([0.0])
        timing = {
            "n": 1,
            "p50": 100.0,
            "p95": 100.0,
            "max": 100.0,
            "mean": 100.0,
            "total_ms": 100.0,
        }
        cache.save_odometry(poses, ts, cfg, metrics={"timing": timing})

        # Load metrics back
        m = cache.get_stage_metrics("odometry", cfg)
        assert m is not None
        assert m["timing"]["p50"] == 100.0

    def test_get_stage_metrics_stale(self, tmp_path: Path):
        from src.cache.layered_cache import LayeredCache

        cache = LayeredCache(root=str(tmp_path / "cache"), sequence="test")
        cfg1 = {"data": {"kitti_root": "/tmp"}, "kiss_icp": {"voxel_size": 1.0}}
        cfg2 = {"data": {"kitti_root": "/tmp"}, "kiss_icp": {"voxel_size": 2.0}}

        poses = np.eye(4).reshape(1, 4, 4)
        ts = np.array([0.0])
        cache.save_odometry(poses, ts, cfg1, metrics={"timing": {"p50": 50.0}})

        # Different config → stale → returns None
        assert cache.get_stage_metrics("odometry", cfg2) is None


class TestApeRotationAboutStart:
    @staticmethod
    def _curved_path(n: int = 200) -> np.ndarray:
        t = np.linspace(0.0, 1.0, n)
        return np.stack([300.0 * t, 80.0 * np.sin(3.0 * t), 2.0 * t**2], axis=1)

    def test_pure_tilt_about_start_is_removed(self):
        gt = self._curved_path()
        # 1 deg pitch about the start point: first-frame error grows to ~5 m.
        a = np.radians(1.0)
        R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
        est = gt @ R.T
        raw = float(np.sqrt(np.mean(np.sum((est - gt) ** 2, axis=1))))
        assert raw > 2.0
        assert ape_rotation_about_start(est, gt) < 1e-9

    def test_never_worse_than_first_frame(self):
        rng = np.random.default_rng(0)
        gt = self._curved_path()
        est = gt + np.cumsum(rng.normal(0.0, 0.05, gt.shape), axis=0)
        est -= est[0] - gt[0]
        raw = float(np.sqrt(np.mean(np.sum((est - gt) ** 2, axis=1))))
        assert ape_rotation_about_start(est, gt) <= raw + 1e-12
