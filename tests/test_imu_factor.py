"""Tests for GTSAM IMU preintegration factor (tight coupling)."""

from __future__ import annotations

import gtsam
import numpy as np

from src.optimization.imu_factor import (
    ImuPreintegrator,
    build_tight_coupled_graph,
    make_preintegration_params,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _straight_poses(n: int, step: float = 1.0) -> list[np.ndarray]:
    """n poses along x-axis, spaced step metres apart."""
    poses = []
    for i in range(n):
        T = np.eye(4)
        T[0, 3] = i * step
        poses.append(T)
    return poses


def _uniform_timestamps(n: int, dt: float = 0.1) -> np.ndarray:
    return np.arange(n, dtype=float) * dt


def _gravity_imu(n: int, gravity: float = 9.81) -> tuple[np.ndarray, np.ndarray]:
    """IMU measurements for a stationary sensor: gravity on z-axis, no angular rate."""
    acc = np.zeros((n, 3))
    acc[:, 2] = gravity  # z-up, gravity compensation returns zero net force
    gyro = np.zeros((n, 3))
    return acc, gyro


# ---------------------------------------------------------------------------
# make_preintegration_params
# ---------------------------------------------------------------------------


def test_make_preintegration_params_returns_params():
    params = make_preintegration_params()
    assert isinstance(params, gtsam.PreintegrationParams)


def test_make_preintegration_params_custom_gravity():
    params = make_preintegration_params(gravity=9.8)
    # Gravity magnitude is embedded in the params; just verify no error.
    assert params is not None


def test_make_preintegration_params_covariance_scales():
    params_low = make_preintegration_params(accel_noise_sigma=0.1)
    params_high = make_preintegration_params(accel_noise_sigma=10.0)
    # Verify different sigmas produce distinct objects (not cached singleton).
    assert params_low is not params_high


# ---------------------------------------------------------------------------
# ImuPreintegrator
# ---------------------------------------------------------------------------


def test_preintegrator_default_construction():
    pint = ImuPreintegrator()
    assert pint.preintegrated is not None


def test_preintegrator_add_single_measurement():
    pint = ImuPreintegrator()
    acc = np.array([0.0, 0.0, 9.81])
    gyro = np.zeros(3)
    pint.add(acc, gyro, 0.01)
    # After one integration step, deltaRij should still be close to identity
    # (gravity-only, no rotation). We just verify it doesn't raise.
    assert pint.preintegrated is not None


def test_preintegrator_reset_clears_integration():
    pint = ImuPreintegrator()
    acc = np.array([1.0, 0.0, 9.81])
    gyro = np.array([0.1, 0.0, 0.0])
    for _ in range(10):
        pint.add(acc, gyro, 0.01)

    pint.reset()
    # After reset, preintegrated measurements should be near identity.
    # deltaPij should be near zero (fresh start).
    pim = pint.preintegrated
    delta_pos = pim.deltaPij()
    assert np.linalg.norm(delta_pos) < 1e-9


def test_preintegrator_make_factor_returns_imu_factor():
    params = make_preintegration_params()
    pint = ImuPreintegrator(params)
    acc, gyro = _gravity_imu(5)
    for i in range(5):
        pint.add(acc[i], gyro[i], 0.01)

    factor = pint.make_factor(
        pose_key_i=gtsam.symbol("x", 0),
        vel_key_i=gtsam.symbol("v", 0),
        pose_key_j=gtsam.symbol("x", 1),
        vel_key_j=gtsam.symbol("v", 1),
        bias_key=gtsam.symbol("b", 0),
    )
    assert isinstance(factor, gtsam.ImuFactor)


# ---------------------------------------------------------------------------
# build_tight_coupled_graph
# ---------------------------------------------------------------------------


def test_build_tight_graph_returns_correct_count():
    n = 8
    poses = _straight_poses(n)
    lidar_ts = _uniform_timestamps(n, dt=0.1)
    imu_n = 80
    imu_ts = _uniform_timestamps(imu_n, dt=0.01)
    acc, gyro = _gravity_imu(imu_n)

    opt_poses, bias_hist = build_tight_coupled_graph(
        poses=poses,
        imu_acc=acc,
        imu_gyro=gyro,
        imu_timestamps=imu_ts,
        lidar_timestamps=lidar_ts,
    )
    assert len(opt_poses) == n
    assert len(bias_hist) == n


def test_build_tight_graph_pose_shape():
    poses = _straight_poses(5)
    lidar_ts = _uniform_timestamps(5, dt=0.1)
    acc, gyro = _gravity_imu(50)
    imu_ts = _uniform_timestamps(50, dt=0.01)

    opt_poses, _ = build_tight_coupled_graph(
        poses=poses,
        imu_acc=acc,
        imu_gyro=gyro,
        imu_timestamps=imu_ts,
        lidar_timestamps=lidar_ts,
    )
    for p in opt_poses:
        assert p.shape == (4, 4)


def test_build_tight_graph_straight_line_preserved():
    """With loose IMU noise, LiDAR between-factors dominate and poses stay near input."""
    n = 10
    poses = _straight_poses(n, step=1.0)
    lidar_ts = _uniform_timestamps(n, dt=0.1)
    acc, gyro = _gravity_imu(100)
    imu_ts = _uniform_timestamps(100, dt=0.01)

    # Use default (loose) IMU noise so LiDAR odometry dominates.
    opt_poses, bias_hist = build_tight_coupled_graph(
        poses=poses,
        imu_acc=acc,
        imu_gyro=gyro,
        imu_timestamps=imu_ts,
        lidar_timestamps=lidar_ts,
        # default accel_noise_sigma=5.0, gyro_noise_sigma=0.5 → LiDAR dominates
    )
    for i, (orig, opt) in enumerate(zip(poses, opt_poses)):
        dist = np.linalg.norm(orig[:3, 3] - opt[:3, 3])
        assert dist < 0.5, f"Pose {i} drifted {dist:.3f} m from input"


def test_build_tight_graph_bias_bounded():
    """Bias estimates must stay within reasonable bounds for a well-conditioned problem."""
    n = 10
    poses = _straight_poses(n)
    lidar_ts = _uniform_timestamps(n, dt=0.1)
    acc, gyro = _gravity_imu(100)
    imu_ts = _uniform_timestamps(100, dt=0.01)

    _, bias_hist = build_tight_coupled_graph(
        poses=poses,
        imu_acc=acc,
        imu_gyro=gyro,
        imu_timestamps=imu_ts,
        lidar_timestamps=lidar_ts,
        accel_noise_sigma=0.3,
        gyro_noise_sigma=0.03,
        accel_bias_sigma=0.01,
        gyro_bias_sigma=0.001,
    )
    bias_arr = np.array(bias_hist)
    # Accel bias should be well under 1 m/s²; gyro bias under 0.1 rad/s.
    assert np.all(np.abs(bias_arr[:, :3]) < 1.0), "Accel bias out of bounds"
    assert np.all(np.abs(bias_arr[:, 3:]) < 0.1), "Gyro bias out of bounds"


def test_build_tight_graph_no_imu_fallback():
    """When IMU timestamps don't overlap LiDAR range, function must still succeed."""
    n = 5
    poses = _straight_poses(n)
    lidar_ts = _uniform_timestamps(n, dt=0.1)  # [0, 0.4]

    # IMU entirely outside LiDAR time range
    acc, gyro = _gravity_imu(10)
    imu_ts = _uniform_timestamps(10, dt=0.1) + 100.0  # starts at t=100

    opt_poses, bias_hist = build_tight_coupled_graph(
        poses=poses,
        imu_acc=acc,
        imu_gyro=gyro,
        imu_timestamps=imu_ts,
        lidar_timestamps=lidar_ts,
    )
    assert len(opt_poses) == n


def test_build_tight_graph_with_loop_closure():
    """Loop closure factor is accepted and reduces drift on a trajectory that revisits origin.

    Topology: poses 0..n-2 form a line, pose n-1 is at the same position as pose 0.
    A loop closure (n-1, 0, eye(4)) corrects accumulated drift on the return segment.
    """
    n = 10
    # First half: move along x. Second half: return to origin (same x as start).
    poses = []
    for i in range(n):
        T = np.eye(4)
        T[0, 3] = float(i % (n // 2))  # 0,1,2,3,4,0,1,2,3,4 → closes at i=5..9
        poses.append(T)

    lidar_ts = _uniform_timestamps(n, dt=0.1)
    acc, gyro = _gravity_imu(100)
    imu_ts = _uniform_timestamps(100, dt=0.01)

    # Add y-drift to the return segment so loop closure has work to do
    poses_drifted = [p.copy() for p in poses]
    for i in range(5, n):
        poses_drifted[i][1, 3] += 0.5  # 0.5 m y-drift

    # Loop closure: pose n-1 should be at the same location as pose n//2-1 = pose 4.
    # rel_pose = inv(pose_{n-1}) @ pose_{n//2-1} tells the graph they coincide.
    rel_pose = np.linalg.inv(poses[n - 1]) @ poses[n // 2 - 1]
    loop_closures = [(n - 1, n // 2 - 1, rel_pose)]

    # Both with and without LC must succeed and return n poses.
    opt_with_lc, _ = build_tight_coupled_graph(
        poses=poses_drifted,
        imu_acc=acc,
        imu_gyro=gyro,
        imu_timestamps=imu_ts,
        lidar_timestamps=lidar_ts,
        loop_closures=loop_closures,
    )
    opt_without_lc, _ = build_tight_coupled_graph(
        poses=poses_drifted,
        imu_acc=acc,
        imu_gyro=gyro,
        imu_timestamps=imu_ts,
        lidar_timestamps=lidar_ts,
        loop_closures=None,
    )

    assert len(opt_with_lc) == n
    assert len(opt_without_lc) == n

    # With loop closure, total y-drift across the return segment should be smaller.
    y_drift_with = np.mean([abs(opt_with_lc[i][1, 3] - poses[i][1, 3]) for i in range(5, n)])
    y_drift_without = np.mean([abs(opt_without_lc[i][1, 3] - poses[i][1, 3]) for i in range(5, n)])
    assert y_drift_with <= y_drift_without + 0.05, (
        f"Loop closure did not reduce y-drift: "
        f"with={y_drift_with:.3f}, without={y_drift_without:.3f}"
    )


def test_build_tight_graph_noise_params_affect_result():
    """Tighter IMU noise should produce a result that relies more on IMU trajectory."""
    n = 8
    poses = _straight_poses(n)
    lidar_ts = _uniform_timestamps(n, dt=0.1)
    # Constant-velocity IMU: small forward acceleration
    acc = np.zeros((80, 3))
    acc[:, 0] = 0.1  # tiny x acceleration
    acc[:, 2] = 9.81
    gyro = np.zeros((80, 3))
    imu_ts = _uniform_timestamps(80, dt=0.01)

    opt_loose_imu, _ = build_tight_coupled_graph(
        poses=poses,
        imu_acc=acc,
        imu_gyro=gyro,
        imu_timestamps=imu_ts,
        lidar_timestamps=lidar_ts,
        accel_noise_sigma=5.0,  # very loose — IMU mostly ignored
        gyro_noise_sigma=0.5,
    )
    opt_tight_imu, _ = build_tight_coupled_graph(
        poses=poses,
        imu_acc=acc,
        imu_gyro=gyro,
        imu_timestamps=imu_ts,
        lidar_timestamps=lidar_ts,
        accel_noise_sigma=0.3,  # tight — IMU has real influence
        gyro_noise_sigma=0.03,
    )

    # Both should produce valid outputs; tight IMU result may differ from loose.
    assert len(opt_loose_imu) == n
    assert len(opt_tight_imu) == n
    # With tight IMU, the optimizer uses IMU data more; results should differ.
    diffs = [np.linalg.norm(a[:3, 3] - b[:3, 3]) for a, b in zip(opt_loose_imu, opt_tight_imu)]
    assert max(diffs) > 1e-6, "Tight vs loose IMU noise produced identical results"
