"""Tests of the degeneracy analyzer and the edge relaxation built on it."""

from __future__ import annotations

import numpy as np
import pytest

from src.odometry.degeneracy import (
    DegeneracyAnalyzer,
    DegeneracyScore,
    _pca_normals,
    _voxel_downsample,
    directional_covariance,
    downgraded_edge_covariances,
    sustained_mask,
)

# ---------------------------------------------------------------------------
# Synthetic point cloud generators
# ---------------------------------------------------------------------------


def _corridor(length: float = 30.0, width: float = 4.0, n_per_side: int = 800) -> np.ndarray:
    """Two parallel walls + a ground plane, no forward features.

    The x-axis is the unobserved (forward) direction: sliding the source
    along +x produces zero residual for all correspondences, so the
    translation Hessian must have a near-zero eigenvalue along x.
    """
    rng = np.random.default_rng(0)
    ys_left = np.full(n_per_side, -width / 2)
    ys_right = np.full(n_per_side, width / 2)
    zs = rng.uniform(-1.0, 2.0, size=n_per_side)
    xs = rng.uniform(0.0, length, size=n_per_side)
    left_wall = np.stack([xs, ys_left, zs], axis=1)
    right_wall = np.stack([xs, ys_right, zs], axis=1)

    n_floor = n_per_side
    xf = rng.uniform(0.0, length, size=n_floor)
    yf = rng.uniform(-width / 2, width / 2, size=n_floor)
    zf = np.full(n_floor, -1.5)
    floor = np.stack([xf, yf, zf], axis=1)
    return np.concatenate([left_wall, right_wall, floor], axis=0)


def _intersection(size: float = 20.0, n_per_plane: int = 800) -> np.ndarray:
    """Three mutually orthogonal planes (floor + 2 walls) — fully constrained."""
    rng = np.random.default_rng(1)
    xs1 = rng.uniform(-size / 2, size / 2, n_per_plane)
    ys1 = rng.uniform(-size / 2, size / 2, n_per_plane)
    zs1 = np.full(n_per_plane, -1.5)
    floor = np.stack([xs1, ys1, zs1], axis=1)

    xs2 = np.full(n_per_plane, -size / 2)
    ys2 = rng.uniform(-size / 2, size / 2, n_per_plane)
    zs2 = rng.uniform(-1.0, 2.0, n_per_plane)
    wall_x = np.stack([xs2, ys2, zs2], axis=1)

    xs3 = rng.uniform(-size / 2, size / 2, n_per_plane)
    ys3 = np.full(n_per_plane, -size / 2)
    zs3 = rng.uniform(-1.0, 2.0, n_per_plane)
    wall_y = np.stack([xs3, ys3, zs3], axis=1)
    return np.concatenate([floor, wall_x, wall_y], axis=0)


# ---------------------------------------------------------------------------
# Unit tests: helpers
# ---------------------------------------------------------------------------


def test_voxel_downsample_reduces_points():
    pts = np.random.default_rng(0).uniform(-1, 1, size=(1000, 3))
    ds = _voxel_downsample(pts, voxel_size=0.5)
    assert ds.shape[0] < pts.shape[0]
    assert ds.shape[1] == 3


def test_voxel_downsample_empty_input():
    assert _voxel_downsample(np.zeros((0, 3)), 0.5).shape == (0, 3)


def test_pca_normals_shapes_and_unit_length():
    pts = _intersection()
    normals, quality = _pca_normals(pts, k=10)
    assert normals.shape == pts.shape
    assert quality.shape == (pts.shape[0],)
    norm_mag = np.linalg.norm(normals, axis=1)
    np.testing.assert_allclose(norm_mag, 1.0, rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# Unit tests: DegeneracyAnalyzer core behavior
# ---------------------------------------------------------------------------


def test_corridor_is_degenerate():
    """Long corridor should produce a very small lambda_min along x."""
    cloud = _corridor()
    analyzer = DegeneracyAnalyzer(voxel_size=0.3, normal_k=10)
    score = analyzer.analyze(cloud, cloud)
    assert not score.is_null
    # x direction should dominate the smallest eigenvector
    assert abs(score.eig_direction[0]) > 0.9, (
        f"expected eig_direction aligned with x, got {score.eig_direction}"
    )
    # lambda_min << lambda_max for a corridor
    assert score.cond_number > 50.0, f"expected corridor cond > 50, got {score.cond_number}"


def test_intersection_is_well_constrained():
    """Three orthogonal planes should yield a cond number close to 1."""
    cloud = _intersection()
    analyzer = DegeneracyAnalyzer(voxel_size=0.3, normal_k=10)
    score = analyzer.analyze(cloud, cloud)
    assert not score.is_null
    assert score.cond_number < 5.0, f"expected intersection cond < 5, got {score.cond_number}"
    assert score.lambda_min > 0.0


def test_corridor_cond_vastly_exceeds_intersection_cond():
    """Monotonicity: corridor degeneracy must be strictly larger."""
    a = DegeneracyAnalyzer(voxel_size=0.3, normal_k=10)
    corr = a.analyze(_corridor(), _corridor())
    inter = a.analyze(_intersection(), _intersection())
    assert corr.cond_number > inter.cond_number * 10.0


def test_max_nn_dist_rejects_far_points():
    """All correspondences beyond max_nn_dist should be filtered out."""
    source = np.random.default_rng(2).uniform(-1, 1, size=(500, 3))
    target = source + np.array([100.0, 0.0, 0.0])  # translated out of range
    analyzer = DegeneracyAnalyzer(max_nn_dist=0.5, voxel_size=0.1, normal_k=10)
    score = analyzer.analyze(source, target)
    assert score.is_null


def test_empty_inputs_return_null():
    analyzer = DegeneracyAnalyzer()
    score = analyzer.analyze(np.zeros((0, 3)), np.zeros((0, 3)))
    assert score.is_null


def test_score_null_constructor():
    s = DegeneracyScore.null()
    assert s.is_null
    assert np.isnan(s.cond_number)
    assert s.n_corr == 0


# ---------------------------------------------------------------------------
# Sustained runs (EMA + minimum run length)
# ---------------------------------------------------------------------------


def _max_run(mask: np.ndarray) -> int:
    best = current = 0
    for v in mask:
        current = current + 1 if v else 0
        best = max(best, current)
    return best


def test_sustained_mask_rejects_single_spike():
    cond = np.array([1.0, 1.0, 1.0, 10.0, 1.0, 1.0, 1.0, 1.0])
    assert not sustained_mask(cond, threshold=5.0, ema_alpha=1.0, min_consecutive=5).any()


def test_sustained_mask_keeps_a_long_run():
    cond = np.array([1.0, 1.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 1.0, 1.0])
    sustained = sustained_mask(cond, threshold=5.0, ema_alpha=1.0, min_consecutive=5)
    # The opening with a 5-wide element keeps the whole 6-frame run.
    assert sustained[2:8].all()
    assert not sustained[0:2].any()
    assert not sustained[8:].any()


def test_sustained_mask_ema_breaks_alternating_noise():
    # Half the raw frames exceed the threshold, but the EMA stays below it on the dips.
    cond = np.tile([2.0, 8.0], 10)
    sustained = sustained_mask(cond, threshold=5.0, ema_alpha=0.3, min_consecutive=5)
    assert _max_run(sustained) < 5


# ---------------------------------------------------------------------------
# Directional inflation of the odometry-edge covariance
# ---------------------------------------------------------------------------


def test_directional_covariance_symmetric_positive_definite():
    base = [0.1, 0.1, 0.1, 0.01, 0.01, 0.01]
    cov = directional_covariance(base, np.array([0.6, -0.8, 0.0]), 10.0)
    assert cov.shape == (6, 6)
    np.testing.assert_allclose(cov, cov.T, atol=1e-14)
    assert np.linalg.eigvalsh(cov).min() > 0


def test_directional_covariance_inflates_only_along_the_direction():
    sigma, alpha = 0.1, 10.0
    cov = directional_covariance([sigma] * 3 + [0.01] * 3, np.array([1.0, 0.0, 0.0]), alpha)
    # GTSAM tangent order: rotation block [0:3, 0:3], translation block [3:6, 3:6].
    trans_eigs = np.sort(np.linalg.eigvalsh(cov[3:6, 3:6]))
    np.testing.assert_allclose(trans_eigs, np.sort([sigma**2, sigma**2, (sigma * alpha) ** 2]))
    np.testing.assert_allclose(cov[0:3, 0:3], np.diag([0.01**2] * 3), atol=1e-14)
    assert cov[3, 3] == pytest.approx((sigma * alpha) ** 2)


def test_directional_covariance_falls_back_on_zero_direction():
    sigma, alpha = 0.1, 10.0
    cov = directional_covariance([sigma] * 3 + [0.01] * 3, np.zeros(3), alpha)
    np.testing.assert_allclose(cov[3:6, 3:6], np.diag([(sigma * alpha) ** 2] * 3), atol=1e-14)


def _scores(n: int, runs: dict[tuple[int, int], float], direction=(1.0, 0.0, 0.0)) -> np.ndarray:
    scores = np.zeros((n, 7), dtype=np.float64)
    scores[:, 0] = 1.0
    for (a, b), value in runs.items():
        scores[a:b, 0] = value
    scores[:, 4:7] = direction
    return scores


_DEG_CFG = {
    "cond_threshold": 5.0,
    "sigma_inflation_factor": 10.0,
    "ema_alpha": 1.0,
    "min_consecutive": 5,
}


def test_downgraded_edges_only_on_sustained_runs():
    scores = _scores(20, {(5, 6): 100.0, (10, 18): 50.0})
    poses = np.tile(np.eye(4), (20, 1, 1))
    edges = downgraded_edge_covariances(scores, poses, [0.1] * 3 + [0.01] * 3, _DEG_CFG)
    assert edges[5] is None
    assert all(isinstance(edges[i], np.ndarray) and edges[i].shape == (6, 6) for i in range(10, 18))
    assert edges[0] is None and edges[9] is None and edges[19] is None


def test_downgraded_edges_express_the_direction_in_the_body_frame():
    """A map-frame x direction is the body y axis of a pose turned by -90 deg about z."""
    scores = _scores(10, {(0, 10): 50.0}, direction=(1.0, 0.0, 0.0))
    yaw = -np.pi / 2
    R = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    poses = np.tile(np.eye(4), (10, 1, 1))
    poses[:, :3, :3] = R
    sigma, alpha = 0.1, 10.0
    edges = downgraded_edge_covariances(scores, poses, [sigma] * 3 + [0.01] * 3, _DEG_CFG)
    trans = edges[5][3:6, 3:6]
    assert trans[1, 1] == pytest.approx((sigma * alpha) ** 2)
    assert trans[0, 0] == pytest.approx(sigma**2)


def test_downgraded_edges_disabled_without_inflation():
    scores = _scores(10, {(0, 10): 50.0})
    cfg = {**_DEG_CFG, "sigma_inflation_factor": 1.0}
    edges = downgraded_edge_covariances(scores, np.tile(np.eye(4), (10, 1, 1)), [0.1] * 6, cfg)
    assert all(e is None for e in edges)
