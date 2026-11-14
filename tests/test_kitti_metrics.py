"""Tests for the KITTI devkit relative-error port."""

from __future__ import annotations

import numpy as np
import pytest

from src.benchmarks.kitti_metrics import SEGMENT_LENGTHS_M, kitti_segment_errors


def _straight_line(n: int, spacing: float) -> np.ndarray:
    """Camera-frame trajectory driving forward along +z."""
    poses = np.tile(np.eye(4), (n, 1, 1))
    poses[:, 2, 3] = np.arange(n) * spacing
    return poses


def _curved_trajectory(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """A smooth curve plus an estimate with accumulated random motion errors."""
    rng = np.random.default_rng(seed)
    gt = [np.eye(4)]
    est = [np.eye(4)]
    for i in range(1, n):
        yaw = 0.002 * np.sin(i / 150.0)
        step = np.eye(4)
        step[:3, :3] = np.array(
            [[np.cos(yaw), 0.0, np.sin(yaw)], [0.0, 1.0, 0.0], [-np.sin(yaw), 0.0, np.cos(yaw)]]
        )
        step[2, 3] = 1.0
        w = rng.normal(0.0, 3e-4, 3)
        K = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
        U, _, Vt = np.linalg.svd(np.eye(3) + K + 0.5 * K @ K)
        noise = np.eye(4)
        noise[:3, :3] = U @ Vt
        noise[:3, 3] = rng.normal(0.0, 5e-3, 3)
        gt.append(gt[-1] @ step)
        est.append(est[-1] @ step @ noise)
    return np.asarray(gt), np.asarray(est)


def test_perfect_estimate_has_zero_error():
    gt, _ = _curved_trajectory(1200, seed=0)
    e = kitti_segment_errors(gt, gt.copy())
    assert e.count > 0
    assert e.t_rel_percent() == pytest.approx(0.0, abs=1e-9)
    assert e.r_rel_deg_per_100m() == pytest.approx(0.0, abs=1e-5)


def test_one_percent_scale_error_gives_one_percent():
    gt = _straight_line(10000, spacing=0.1)  # 1000 m
    est = gt.copy()
    est[:, :3, 3] *= 1.01
    e = kitti_segment_errors(gt, est)
    # The devkit closes a segment at the first frame *beyond* the nominal
    # length, so each segment is one spacing (0.1 m) longer than nominal.
    assert e.t_rel_percent() == pytest.approx(1.0, abs=2e-3)
    assert e.r_rel_deg_per_100m() == pytest.approx(0.0, abs=1e-9)
    axes = e.axis_breakdown_percent()
    assert axes["forward"] == pytest.approx(1.0, abs=2e-3)
    assert axes["lateral"] == pytest.approx(0.0, abs=1e-9)
    assert axes["vertical"] == pytest.approx(0.0, abs=1e-9)


def test_segments_shorter_than_path_are_skipped():
    gt = _straight_line(3001, spacing=0.1)  # 300 m: only 100/200 m segments fit
    e = kitti_segment_errors(gt, gt.copy())
    assert set(e.by_length()) == {100, 200}
    assert set(e.by_length()).issubset(SEGMENT_LENGTHS_M)


def test_matches_kiss_icp_reference_implementation():
    sequence_error = pytest.importorskip("kiss_icp.metrics").sequence_error
    gt, est = _curved_trajectory(1500, seed=1)
    e = kitti_segment_errors(gt, est)
    t_ref, r_ref_deg_per_m = sequence_error(gt, est)
    assert e.t_rel_percent() == pytest.approx(t_ref, rel=1e-6)
    # KISS-ICP returns deg/m; rotation agrees to ~0.1 % (its C++ path
    # accumulates in lower precision near acos(1)).
    assert e.r_rel_deg_per_100m() / 100.0 == pytest.approx(r_ref_deg_per_m, rel=2e-3)


def test_shape_mismatch_raises():
    gt = _straight_line(100, 1.0)
    with pytest.raises(ValueError):
        kitti_segment_errors(gt, gt[:-1])
