"""Tests for the covariances PoseGraphOptimizer reports."""

from __future__ import annotations

import gtsam
import numpy as np
import pytest

from src.optimization.pose_graph import PoseGraphOptimizer


def _straight(n: int, step: float = 1.0) -> list[np.ndarray]:
    poses = []
    for i in range(n):
        T = np.eye(4)
        T[0, 3] = i * step
        poses.append(T)
    return poses


def _curved(n: int) -> list[np.ndarray]:
    """A drive along a turning, slightly climbing path (2 m and 3 deg per step)."""
    poses = [np.eye(4)]
    step = gtsam.Pose3(
        gtsam.Rot3.Ypr(np.radians(3.0), np.radians(0.5), 0.0), gtsam.Point3(2.0, 0, 0)
    )
    for _ in range(n - 1):
        poses.append(poses[-1] @ step.matrix())
    return poses


def _adjoint(T: np.ndarray) -> np.ndarray:
    return gtsam.Pose3(T).AdjointMap()


def test_marginals_require_optimize() -> None:
    opt = PoseGraphOptimizer()
    opt.build_graph(_straight(3))
    with pytest.raises(RuntimeError, match="optimize"):
        opt.get_position_marginals([0])


def test_marginals_shapes_and_psd() -> None:
    """All returned blocks are 3x3, symmetric, and positive semidefinite."""
    opt = PoseGraphOptimizer()
    opt.build_graph(_curved(20))
    opt.optimize()
    cov = opt.get_position_marginals(keys=[0, 5, 10, 15, 19])
    assert set(cov.keys()) == {0, 5, 10, 15, 19}
    for k, c in cov.items():
        assert c.shape == (3, 3), f"key {k} shape {c.shape}"
        np.testing.assert_allclose(c, c.T, atol=1e-9)
        assert np.linalg.eigvalsh(c).min() > -1e-9, f"key {k} not PSD"


def test_marginals_grow_with_chain_distance() -> None:
    """With only the frame-0 prior, position uncertainty accumulates along the chain."""
    opt = PoseGraphOptimizer()
    opt.build_graph(_straight(30))
    opt.optimize()
    cov = opt.get_position_marginals(keys=[0, 10, 20, 29])
    traces = [float(np.trace(cov[k])) for k in (0, 10, 20, 29)]
    assert traces[0] < traces[1] < traces[2] < traces[3], f"non-monotone: {traces}"


def test_marginals_default_all_keys() -> None:
    opt = PoseGraphOptimizer()
    opt.build_graph(_straight(8))
    opt.optimize()
    assert set(opt.get_position_marginals().keys()) == set(range(8))


def test_position_marginal_is_in_map_axes() -> None:
    """A pose facing +y with an uncertain forward step is uncertain along map y.

    GTSAM's marginal is in the pose's body axes, where the same uncertainty
    sits on x; the reported covariance must be rotated into the map.
    """
    T1 = gtsam.Pose3(gtsam.Rot3.Yaw(np.pi / 2), gtsam.Point3(0, 1, 0)).matrix()
    opt = PoseGraphOptimizer(
        odom_sigmas=[1.0, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6],
        prior_sigmas=[1e-6] * 6,
    )
    opt.build_graph([np.eye(4), T1])
    opt.optimize()
    cov = opt.get_position_marginals([1])[1]
    np.testing.assert_allclose(cov, np.diag([0.0, 1.0, 0.0]), atol=1e-6)


def test_relative_covariance_matches_chain_propagation() -> None:
    """On a chain, the relative-pose covariance is the edges' covariances propagated.

    ``Sigma_k = Ad(D_k^-1) Sigma_{k-1} Ad(D_k^-1)^T + Q`` along the relative
    motions ``D_k``, the open-traverse error propagation.
    """
    poses = _curved(25)
    odom = [0.02, 0.01, 0.005, 0.001, 0.002, 0.003]
    opt = PoseGraphOptimizer(odom_sigmas=odom)
    opt.build_graph(poses)
    opt.optimize()
    first, last = 4, 24
    got = opt.relative_pose_covariances(first, [first, last])

    Q = np.diag(np.array([*odom[3:], *odom[:3]]) ** 2)
    S = np.zeros((6, 6))
    for k in range(first + 1, last + 1):
        A = _adjoint(np.linalg.inv(np.linalg.inv(poses[k - 1]) @ poses[k]))
        S = A @ S @ A.T + Q
    np.testing.assert_allclose(got[last], S, rtol=1e-6, atol=1e-12)
    np.testing.assert_array_equal(got[first], np.zeros((6, 6)))


def test_relative_covariance_does_not_depend_on_the_datum() -> None:
    """A loosely or tightly fixed frame 0 changes absolute, not relative, covariances."""
    poses = _curved(30)
    T_loop = np.linalg.inv(poses[3]) @ poses[27]
    rel, absolute = [], []
    for prior in ([1e-6] * 6, [1.0, 1.0, 1.0, 0.1, 0.1, 0.1]):
        opt = PoseGraphOptimizer(prior_sigmas=prior)
        opt.build_graph(poses)
        opt.add_loop_closure(3, 27, T_loop, sigmas=[0.05, 0.05, 0.05, 0.005, 0.005, 0.005])
        opt.optimize()
        rel.append(opt.relative_pose_covariances(10, [20])[20])
        absolute.append(opt.get_position_marginals([20])[20])
    np.testing.assert_allclose(rel[0], rel[1], rtol=1e-5, atol=1e-12)
    assert np.trace(absolute[1]) > 10 * np.trace(absolute[0])


def test_position_priors_bound_the_gap() -> None:
    """Position fixes at both ends of a gap: uncertainty peaks inside, collapses at the fixes."""
    poses = _straight(41)
    opt = PoseGraphOptimizer()
    opt.build_graph(poses)
    for k in (0, 20, 40):
        opt.add_position_prior(k, poses[k][:3, 3], [0.01, 0.01, 0.01])
    opt.optimize()
    tr = {k: float(np.trace(c)) for k, c in opt.get_position_marginals([0, 10, 20, 30, 40]).items()}
    assert tr[10] > 10 * tr[20] and tr[30] > 10 * tr[20] and tr[30] > 10 * tr[40]
    assert tr[20] < 3 * 0.01**2 + 1e-9
