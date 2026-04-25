"""Tests for pose graph optimization and loop closure detection."""

from __future__ import annotations

import numpy as np

from src.optimization.loop_closure import LoopClosureDetector
from src.optimization.pose_graph import PoseGraphOptimizer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_straight_trajectory(n: int, step: float = 1.0) -> list[np.ndarray]:
    """Create a straight-line trajectory along x-axis."""
    poses = []
    for i in range(n):
        T = np.eye(4)
        T[0, 3] = i * step
        poses.append(T)
    return poses


def _make_loop_trajectory(n: int = 40, radius: float = 20.0) -> list[np.ndarray]:
    """Create a circular trajectory that revisits the start."""
    poses = []
    for i in range(n):
        angle = 2 * np.pi * i / n
        T = np.eye(4)
        T[0, 3] = radius * np.cos(angle)
        T[1, 3] = radius * np.sin(angle)
        # Rotation around z-axis to face forward
        T[0, 0] = np.cos(angle + np.pi / 2)
        T[0, 1] = -np.sin(angle + np.pi / 2)
        T[1, 0] = np.sin(angle + np.pi / 2)
        T[1, 1] = np.cos(angle + np.pi / 2)
        poses.append(T)
    return poses


def _add_drift(poses: list[np.ndarray], drift_per_frame: float = 0.01) -> list[np.ndarray]:
    """Add cumulative drift to a trajectory."""
    drifted = []
    for i, pose in enumerate(poses):
        p = pose.copy()
        p[0, 3] += i * drift_per_frame
        p[1, 3] += i * drift_per_frame * 0.5
        drifted.append(p)
    return drifted


# ---------------------------------------------------------------------------
# Tests: PoseGraphOptimizer
# ---------------------------------------------------------------------------


def test_build_graph_size():
    """3 poses → 1 prior + 2 between = 3 factors."""
    poses = _make_straight_trajectory(3)
    opt = PoseGraphOptimizer()
    opt.build_graph(poses)
    assert opt.graph_size == 3
    assert opt.n_poses == 3


def test_optimize_straight_line():
    """Straight trajectory should remain roughly straight after optimization."""
    poses = _make_straight_trajectory(5)
    opt = PoseGraphOptimizer()
    opt.build_graph(poses)
    result = opt.optimize()

    assert len(result) == 5
    for i, pose in enumerate(result):
        assert pose.shape == (4, 4)
        np.testing.assert_array_almost_equal(pose[:3, 3], [float(i), 0, 0], decimal=2)


def test_optimize_reduces_error():
    """Optimization should bring noisy poses closer to the prior."""
    poses = _make_straight_trajectory(5)
    # Add noise to initial poses (but keep the factors from the clean poses)
    noisy = [p.copy() for p in poses]
    noisy[2][0, 3] += 0.5  # perturb middle pose
    noisy[3][1, 3] += 0.3

    opt = PoseGraphOptimizer()
    opt.build_graph(poses)  # build graph from clean relative transforms

    # Replace initial values with noisy ones
    import gtsam

    opt.initial_values = gtsam.Values()
    for i, p in enumerate(noisy):
        opt.initial_values.insert(i, gtsam.Pose3(p))

    result = opt.optimize()
    # Optimized poses should be closer to clean trajectory
    err_before = sum(np.linalg.norm(noisy[i][:3, 3] - poses[i][:3, 3]) for i in range(5))
    err_after = sum(np.linalg.norm(result[i][:3, 3] - poses[i][:3, 3]) for i in range(5))
    assert err_after < err_before


def test_add_loop_closure_increases_graph():
    """Adding a loop closure should increase graph size by 1."""
    poses = _make_straight_trajectory(5)
    opt = PoseGraphOptimizer()
    opt.build_graph(poses)
    size_before = opt.graph_size

    opt.add_loop_closure(0, 4, np.eye(4))
    assert opt.graph_size == size_before + 1


def test_edge_sigmas_none_matches_default():
    """Passing edge_sigmas=None must produce exactly the same optimization result."""
    poses = _make_straight_trajectory(6)
    opt_a = PoseGraphOptimizer()
    opt_a.build_graph(poses, edge_sigmas=None)
    result_a = opt_a.optimize()

    opt_b = PoseGraphOptimizer()
    opt_b.build_graph(poses)
    result_b = opt_b.optimize()

    for pa, pb in zip(result_a, result_b):
        np.testing.assert_array_almost_equal(pa, pb, decimal=10)


def test_edge_sigmas_accepts_full_covariance_matrix():
    """PoseGraphOptimizer.build_graph accepts (6,6) ndarray per-edge override.

    Dispatches to Gaussian.Covariance (full) noise model instead of
    Diagonal.Sigmas (list). Used by SUP-07 directional sigma inflation.

    Compare to baseline (no override) with the same noisy initial: pose 2
    should drift further from clean x when its inflow edge is relaxed
    along x. We use a corner-at-pose-4 prior (via an explicit loop closure
    to the far-end clean position) so the drift is observable — otherwise
    GTSAM's LM snaps to the unique MAP from the clean between factors
    despite relaxed sigmas.
    """
    import gtsam

    poses = _make_straight_trajectory(5)
    noisy_init = [p.copy() for p in poses]
    # Large x-perturbation on both endpoints so the relaxed middle edge
    # can actually absorb it instead of being pulled back by the tight chain.
    noisy_init[2][0, 3] += 1.0
    noisy_init[3][0, 3] += 1.0
    noisy_init[4][0, 3] += 1.0

    sigma = 0.1
    alpha = 100.0
    cov = np.zeros((6, 6), dtype=np.float64)
    cov[0:3, 0:3] = np.diag([0.01**2] * 3)  # rotation untouched
    cov[3:6, 3:6] = np.diag([sigma**2, sigma**2, sigma**2])
    v = np.array([1.0, 0.0, 0.0])  # inflate along x
    cov[3:6, 3:6] += sigma**2 * (alpha**2 - 1) * np.outer(v, v)

    # Relax the last edges (2,3,4) along x so pose 4 can float
    inflated: list[list[float] | np.ndarray | None] = [None] * 5
    inflated[3] = cov
    inflated[4] = cov

    # Baseline (tight sigmas everywhere)
    opt_a = PoseGraphOptimizer()
    opt_a.build_graph(poses, edge_sigmas=None)
    opt_a.initial_values = gtsam.Values()
    for i, p in enumerate(noisy_init):
        opt_a.initial_values.insert(i, gtsam.Pose3(p))
    result_a = opt_a.optimize()

    # Directional: relaxed only along x on the last two edges
    opt_b = PoseGraphOptimizer()
    opt_b.build_graph(poses, edge_sigmas=inflated)
    opt_b.initial_values = gtsam.Values()
    for i, p in enumerate(noisy_init):
        opt_b.initial_values.insert(i, gtsam.Pose3(p))
    result_b = opt_b.optimize()

    # With directional relaxation along x, pose 4's x should stay closer to
    # noisy init (5.0) and farther from clean (4.0) than the baseline.
    err_x_a = abs(result_a[4][0, 3] - 4.0)
    err_x_b = abs(result_b[4][0, 3] - 4.0)
    assert err_x_b > err_x_a, (
        f"Directional inflation along x should let pose 4 drift more on x: "
        f"baseline x_err={err_x_a:.4f} directional x_err={err_x_b:.4f}"
    )


def test_edge_sigmas_directional_preserves_perpendicular_direction():
    """Directional covariance inflated along x should NOT relax y-axis perturbation."""
    import gtsam

    poses = _make_straight_trajectory(5)
    noisy_init = [p.copy() for p in poses]
    noisy_init[2][1, 3] += 0.5  # perturb pose 2 along y (perpendicular to inflated x)

    sigma = 0.1
    alpha = 100.0
    cov = np.zeros((6, 6), dtype=np.float64)
    cov[0:3, 0:3] = np.diag([0.01**2] * 3)
    cov[3:6, 3:6] = np.diag([sigma**2, sigma**2, sigma**2])
    v = np.array([1.0, 0.0, 0.0])  # inflate only along x
    cov[3:6, 3:6] += sigma**2 * (alpha**2 - 1) * np.outer(v, v)

    # Directional: inflate edges (1,2) and (2,3) along x only
    inflated: list[list[float] | np.ndarray | None] = [None] * 5
    inflated[2] = cov
    inflated[3] = cov

    opt_dir = PoseGraphOptimizer()
    opt_dir.build_graph(poses, edge_sigmas=inflated)
    opt_dir.initial_values = gtsam.Values()
    for i, p in enumerate(noisy_init):
        opt_dir.initial_values.insert(i, gtsam.Pose3(p))
    result_dir = opt_dir.optimize()

    # Uniform comparator: inflate tx/ty/tz equally
    inflated_uniform: list[list[float] | np.ndarray | None] = [None] * 5
    inflated_uniform[2] = [sigma * alpha, sigma * alpha, sigma * alpha, 0.01, 0.01, 0.01]
    inflated_uniform[3] = [sigma * alpha, sigma * alpha, sigma * alpha, 0.01, 0.01, 0.01]
    opt_uni = PoseGraphOptimizer()
    opt_uni.build_graph(poses, edge_sigmas=inflated_uniform)
    opt_uni.initial_values = gtsam.Values()
    for i, p in enumerate(noisy_init):
        opt_uni.initial_values.insert(i, gtsam.Pose3(p))
    result_uni = opt_uni.optimize()

    # Under uniform inflation, both x and y are relaxed → pose 2 stays close to noisy init on y
    err_y_uniform = abs(result_uni[2][1, 3] - 0.0)  # clean y=0
    # Under directional inflation along x, y direction stays tight → pose 2 snaps back to y=0
    err_y_dir = abs(result_dir[2][1, 3] - 0.0)

    assert err_y_dir < err_y_uniform, (
        f"Directional (x-only) should keep y tight better than uniform inflation: "
        f"dir_err_y={err_y_dir:.4f} uniform_err_y={err_y_uniform:.4f}"
    )


def test_edge_sigmas_downgrade_relaxes_constraint():
    """Inflating sigma on one edge should let the noisy initial value drift back.

    Strategy: build a 5-pose straight trajectory, construct the graph from
    CLEAN relative transforms so factors push toward clean positions, then
    replace initial value for pose 2 with a noisy one. With edge (1, 2) at
    normal sigma, optimization snaps pose 2 close to clean. With the same
    edge inflated 100× (and edge (2, 3) also inflated so pose 2 isn't pulled
    from the far side), pose 2 should stay closer to its noisy initial.
    """
    import gtsam

    poses = _make_straight_trajectory(5)
    noisy_init = [p.copy() for p in poses]
    noisy_init[2][0, 3] += 0.5  # perturb middle pose forward

    # Baseline: uniform sigmas
    opt_a = PoseGraphOptimizer()
    opt_a.build_graph(poses)
    opt_a.initial_values = gtsam.Values()
    for i, p in enumerate(noisy_init):
        opt_a.initial_values.insert(i, gtsam.Pose3(p))
    result_a = opt_a.optimize()

    # Inflated: edges (1,2) and (2,3) translation sigma 100×
    inflated: list[list[float] | None] = [None] * 5
    inflated[2] = [10.0, 10.0, 10.0, 0.01, 0.01, 0.01]
    inflated[3] = [10.0, 10.0, 10.0, 0.01, 0.01, 0.01]
    opt_b = PoseGraphOptimizer()
    opt_b.build_graph(poses, edge_sigmas=inflated)
    opt_b.initial_values = gtsam.Values()
    for i, p in enumerate(noisy_init):
        opt_b.initial_values.insert(i, gtsam.Pose3(p))
    result_b = opt_b.optimize()

    err_a = abs(result_a[2][0, 3] - 2.0)  # clean middle pose is at x=2
    err_b = abs(result_b[2][0, 3] - 2.0)
    assert err_b > err_a, (
        f"Inflated sigmas should relax the constraint on pose 2: "
        f"baseline err={err_a:.4f}, inflated err={err_b:.4f}"
    )


def test_loop_closure_reduces_drift():
    """Loop closure should pull drifted initial values toward the constraint."""
    import gtsam

    n = 20
    poses_clean = _make_loop_trajectory(n, radius=20.0)
    poses_drifted = _add_drift(poses_clean, drift_per_frame=0.1)

    # Build graph with CLEAN between factors (correct local motion)
    opt = PoseGraphOptimizer()
    opt.build_graph(poses_clean)

    # Replace initial values with drifted estimates
    opt.initial_values = gtsam.Values()
    for i, p in enumerate(poses_drifted):
        opt.initial_values.insert(i, gtsam.Pose3(p))

    # Add loop closure: last pose should be near first (near-identity)
    relative = np.linalg.inv(poses_clean[0]) @ poses_clean[-1]
    opt.add_loop_closure(0, n - 1, relative)

    result = opt.optimize()

    # After optimization, poses should be closer to clean trajectory
    err_before = sum(
        np.linalg.norm(poses_drifted[k][:3, 3] - poses_clean[k][:3, 3]) for k in range(n)
    )
    err_after = sum(np.linalg.norm(result[k][:3, 3] - poses_clean[k][:3, 3]) for k in range(n))
    assert err_after < err_before, (
        f"Loop closure should reduce total error: {err_before:.3f} → {err_after:.3f}"
    )


# ---------------------------------------------------------------------------
# Tests: Switchable Constraints (robust kernel on loop closures)
# ---------------------------------------------------------------------------


def _optimize_with_outlier(
    robust_kernel: str | None, robust_scale: float = 1.0, outlier_offset: float = 50.0
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Build a straight-line graph with one bogus loop closure, optimize.

    Returns ``(optimized, clean)`` so callers can score APE against the
    ground-truth clean trajectory. The bogus closure asserts that frame
    ``n-1`` coincides with frame ``0`` offset by ``outlier_offset`` in y
    — a ~50 m lateral jump that is geometrically inconsistent with the
    correct between-factors.
    """
    import gtsam

    n = 12
    clean = _make_straight_trajectory(n, step=1.0)
    opt = PoseGraphOptimizer(robust_kernel=robust_kernel, robust_scale=robust_scale)
    opt.build_graph(clean)

    relative_bogus = np.eye(4)
    relative_bogus[1, 3] = outlier_offset
    opt.add_loop_closure(0, n - 1, relative_bogus)

    # Re-insert clean initials (build_graph already did this; keep explicit).
    opt.initial_values = gtsam.Values()
    for i, p in enumerate(clean):
        opt.initial_values.insert(i, gtsam.Pose3(p))

    return opt.optimize(), clean


def test_robust_kernel_none_is_bitexact():
    """robust_kernel=None must be bit-exact equivalent to legacy default ctor."""
    import gtsam

    clean = _make_straight_trajectory(10, step=1.0)

    legacy = PoseGraphOptimizer()
    legacy.build_graph(clean)
    rel = np.eye(4)
    rel[0, 3] = 9.0  # consistent loop
    legacy.add_loop_closure(0, 9, rel)
    poses_legacy = legacy.optimize()

    new = PoseGraphOptimizer(robust_kernel=None, robust_scale=1.0)
    new.build_graph(clean)
    new.initial_values = gtsam.Values()
    for i, p in enumerate(clean):
        new.initial_values.insert(i, gtsam.Pose3(p))
    new.add_loop_closure(0, 9, rel)
    poses_new = new.optimize()

    for a, b in zip(poses_legacy, poses_new):
        np.testing.assert_allclose(a, b, atol=1e-10)


def test_robust_kernel_suppresses_outlier_closure():
    """A bogus 50 m closure: Gaussian bends trajectory, robust kernels down-weight it.

    Measure APE at the inlier poses (skip the two closure endpoints
    whose residual is the outlier). Robust optimizers should keep the
    straight-line interior much closer to the clean ground truth than
    the pure-Gaussian optimizer.
    """
    gaussian_poses, clean = _optimize_with_outlier(None)
    inlier_idx = list(range(1, len(clean) - 1))

    gaussian_err = np.mean(
        [np.linalg.norm(gaussian_poses[k][:3, 3] - clean[k][:3, 3]) for k in inlier_idx]
    )

    kernels = [("huber", 1.0), ("cauchy", 1.0), ("gm", 1.0), ("dcs", 1.0)]
    robust_errs = {}
    for name, scale in kernels:
        poses, _ = _optimize_with_outlier(name, scale)
        err = np.mean(
            [np.linalg.norm(poses[k][:3, 3] - clean[k][:3, 3]) for k in inlier_idx]
        )
        robust_errs[name] = err

    # At least three of the four robust kernels must slash interior APE to
    # well below half of the Gaussian baseline.
    improvements = [name for name, err in robust_errs.items() if err < 0.5 * gaussian_err]
    assert len(improvements) >= 3, (
        f"Gaussian inlier err={gaussian_err:.3f}, robust errs={robust_errs}. "
        f"Only {improvements} beat the 0.5× threshold."
    )


def test_unknown_robust_kernel_raises():
    """Nonsense kernel name must fail loudly rather than silently no-op."""
    import pytest

    clean = _make_straight_trajectory(5, step=1.0)
    opt = PoseGraphOptimizer(robust_kernel="magicspell", robust_scale=1.0)
    opt.build_graph(clean)
    rel = np.eye(4)
    rel[0, 3] = 4.0
    with pytest.raises(ValueError, match="unknown robust kernel"):
        opt.add_loop_closure(0, 4, rel)


def test_cluster_gt_pairs_event_count_monotone():
    """Larger cluster_gap must not increase the event count (union-find monotonicity)."""
    from scripts.eval_loop_closure_pr import _cluster_gt_pairs

    # Synthetic GT: 3 clusters of 5 nearby pairs each, 200 frames apart.
    pairs = set()
    for c in range(3):
        base_i = c * 200
        base_j = base_i + 1000
        for d in range(5):
            pairs.add((base_i + d, base_j + d))

    counts = {g: len(_cluster_gt_pairs(pairs, cluster_gap=g)) for g in (5, 50, 100, 500)}
    # At gap ≥ 5 neighbors merge within a cluster → 3 events; gap=500 also
    # cannot merge between clusters because they are 200 frames apart but
    # only on the i axis; however inter-cluster Chebyshev distance is 200
    # which exceeds gap=100 but is dominated by gap=500. Either way:
    # monotone non-increase holds.
    assert counts[5] >= counts[50] >= counts[100] >= counts[500], counts
    assert counts[5] == 3


# ---------------------------------------------------------------------------
# Tests: LoopClosureDetector
# ---------------------------------------------------------------------------


def test_detect_candidates_finds_revisit():
    """Circular trajectory should have candidates near the closure point."""
    poses = _make_loop_trajectory(40, radius=10.0)
    detector = LoopClosureDetector(distance_threshold=5.0, min_frame_gap=10)
    candidates = detector.detect_candidates(poses)
    assert len(candidates) > 0
    # At least one candidate should involve early and late frames
    has_closure = any(i < 10 and j > 30 for i, j in candidates)
    assert has_closure, f"Expected early-late closure, got: {candidates}"


def test_detect_candidates_no_loops():
    """Straight trajectory should have no candidates."""
    poses = _make_straight_trajectory(50, step=5.0)
    detector = LoopClosureDetector(distance_threshold=15.0, min_frame_gap=10)
    candidates = detector.detect_candidates(poses)
    assert len(candidates) == 0


def test_detect_candidates_respects_min_gap():
    """Candidates must have at least min_frame_gap separation."""
    poses = _make_loop_trajectory(40, radius=10.0)
    detector = LoopClosureDetector(distance_threshold=50.0, min_frame_gap=20)
    candidates = detector.detect_candidates(poses)
    for i, j in candidates:
        assert j - i >= 20


def test_detect_without_dataset():
    """detect() without dataset should return pose-derived closures."""
    poses = _make_loop_trajectory(40, radius=10.0)
    detector = LoopClosureDetector(distance_threshold=5.0, min_frame_gap=10)
    closures = detector.detect(poses, dataset=None)
    assert len(closures) > 0
    for i, j, rel_pose in closures:
        assert rel_pose.shape == (4, 4)
        assert j > i


def test_icp_downsample_voxel_configurable():
    """icp_downsample_voxel from __init__ reaches the downsample call."""
    # Dense 2m × 2m × 2m grid with 0.2 m steps ≈ 1000 points.
    # Coarser voxel → fewer retained points; finer voxel → more.
    axis = np.arange(-1.0, 1.0, 0.2)
    grid = np.array([[x, y, z] for x in axis for y in axis for z in axis])

    det_coarse = LoopClosureDetector(icp_downsample_voxel=1.0)
    assert det_coarse.icp_downsample_voxel == 1.0
    pcd_coarse = det_coarse._build_downsampled_pcd(grid)

    det_fine = LoopClosureDetector(icp_downsample_voxel=0.25)
    assert det_fine.icp_downsample_voxel == 0.25
    pcd_fine = det_fine._build_downsampled_pcd(grid)

    # Finer voxel must retain strictly more points on a dense input —
    # proves the parameter actually flows through to voxel_down_sample.
    assert len(pcd_fine.points) > len(pcd_coarse.points), (
        f"Finer voxel should retain more points: "
        f"coarse(voxel=1.0)={len(pcd_coarse.points)} "
        f"fine(voxel=0.25)={len(pcd_fine.points)}"
    )

    # Default value sanity check
    det_default = LoopClosureDetector()
    assert det_default.icp_downsample_voxel == 1.0


# ---------------------------------------------------------------------------
# Tests: Integration (pose graph + loop closure)
# ---------------------------------------------------------------------------


def test_full_pipeline_synthetic():
    """Full pipeline: detect closures on clean loop, optimize drifted initial values."""
    import gtsam

    poses_clean = _make_loop_trajectory(40, radius=20.0)
    poses_drifted = _add_drift(poses_clean, drift_per_frame=0.1)

    # Build graph from clean between factors, initialize with drifted values
    opt = PoseGraphOptimizer()
    opt.build_graph(poses_clean)
    opt.initial_values = gtsam.Values()
    for i, p in enumerate(poses_drifted):
        opt.initial_values.insert(i, gtsam.Pose3(p))

    # Detect loop closures on clean trajectory and add them
    detector = LoopClosureDetector(distance_threshold=10.0, min_frame_gap=15)
    closures = detector.detect(poses_clean, dataset=None)
    for i, j, rel_pose in closures:
        opt.add_loop_closure(i, j, rel_pose)

    result = opt.optimize()
    assert len(result) == 40

    # Drifted initial values should be corrected toward clean trajectory
    err_before = sum(
        np.linalg.norm(poses_drifted[k][:3, 3] - poses_clean[k][:3, 3]) for k in range(40)
    )
    err_after = sum(np.linalg.norm(result[k][:3, 3] - poses_clean[k][:3, 3]) for k in range(40))
    assert err_after < err_before
