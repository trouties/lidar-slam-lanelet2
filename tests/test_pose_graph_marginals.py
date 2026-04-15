"""Tests for PoseGraphOptimizer.get_position_marginals (SUP-06)."""

from __future__ import annotations

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


def test_marginals_require_optimize() -> None:
    opt = PoseGraphOptimizer()
    opt.build_graph(_straight(3))
    with pytest.raises(RuntimeError, match="optimize"):
        opt.get_position_marginals([0])


def test_marginals_shapes_and_psd() -> None:
    """All returned blocks are 3x3, symmetric, and positive semidefinite."""
    poses = _straight(20)
    opt = PoseGraphOptimizer()
    opt.build_graph(poses)
    opt.optimize()
    cov = opt.get_position_marginals(keys=[0, 5, 10, 15, 19])
    assert set(cov.keys()) == {0, 5, 10, 15, 19}
    for k, c in cov.items():
        assert c.shape == (3, 3), f"key {k} shape {c.shape}"
        np.testing.assert_allclose(c, c.T, atol=1e-9)
        eigvals = np.linalg.eigvalsh(0.5 * (c + c.T))
        assert eigvals.min() > -1e-9, f"key {k} not PSD: eigvals={eigvals}"


def test_marginals_grow_with_chain_distance() -> None:
    """With only frame-0 prior, translation marginals accumulate along the chain."""
    poses = _straight(30)
    opt = PoseGraphOptimizer()
    opt.build_graph(poses)  # default: only frame 0 prior
    opt.optimize()
    cov = opt.get_position_marginals(keys=[0, 10, 20, 29])
    traces = [float(np.trace(cov[k])) for k in (0, 10, 20, 29)]
    assert traces[0] < traces[1] < traces[2] < traces[3], f"non-monotone: {traces}"


def test_denial_window_inflation() -> None:
    """Replicates SUP-06 acceptance on a synthetic chain.

    priors every 5 frames outside denial window [20, 40]; verify inflation
    in the middle of the window (>= 2x pre-denial baseline) and recovery
    afterwards (<= 1.5x baseline).
    """
    poses = _straight(60)
    opt = PoseGraphOptimizer()
    denial_start, denial_end = 20, 40
    prior_idx = [i for i in range(0, 60, 5) if i < denial_start or i > denial_end]
    opt.build_graph(poses, prior_indices=prior_idx)
    opt.optimize()

    cov = opt.get_position_marginals(keys=[18, 19, 30, 41, 42])
    pre = float(np.mean([np.trace(cov[18]), np.trace(cov[19])]))
    mid = float(np.trace(cov[30]))
    post = float(np.mean([np.trace(cov[41]), np.trace(cov[42])]))

    assert mid / pre >= 2.0, (
        f"denial inflation only {mid / pre:.2f}x (pre={pre:.3e} mid={mid:.3e})"
    )
    assert post / pre <= 1.5, (
        f"recovery ratio {post / pre:.2f}x > 1.5 (pre={pre:.3e} post={post:.3e})"
    )


def test_marginals_default_all_keys() -> None:
    """Omitting `keys` returns one entry per pose."""
    opt = PoseGraphOptimizer()
    opt.build_graph(_straight(8))
    opt.optimize()
    cov = opt.get_position_marginals()
    assert set(cov.keys()) == set(range(8))
