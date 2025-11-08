"""Tests for Error-State Kalman Filter (ESKF)."""

from __future__ import annotations

import numpy as np

from src.fusion.eskf import (
    ESKF,
    matrix_from_quaternion,
    quaternion_from_matrix,
    quaternion_multiply,
    small_angle_quaternion,
)

# ---------------------------------------------------------------------------
# Quaternion utility tests
# ---------------------------------------------------------------------------


def test_quaternion_roundtrip():
    """matrix → quaternion → matrix should preserve rotation."""
    # 45-degree rotation around z-axis
    angle = np.pi / 4
    R = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    q = quaternion_from_matrix(R)
    R_back = matrix_from_quaternion(q)
    np.testing.assert_array_almost_equal(R_back, R, decimal=10)


def test_quaternion_identity():
    """Identity rotation → quaternion [1,0,0,0]."""
    q = quaternion_from_matrix(np.eye(3))
    np.testing.assert_array_almost_equal(q, [1, 0, 0, 0], decimal=10)


def test_quaternion_multiply_identity():
    """q * identity = q."""
    q = quaternion_from_matrix(np.eye(3))
    q2 = np.array([0.707, 0.707, 0, 0])
    q2 /= np.linalg.norm(q2)
    result = quaternion_multiply(q2, q)
    np.testing.assert_array_almost_equal(result, q2, decimal=5)


def test_small_angle_quaternion_zero():
    """Zero rotation vector → identity quaternion."""
    q = small_angle_quaternion(np.zeros(3))
    np.testing.assert_array_almost_equal(q, [1, 0, 0, 0], decimal=10)


# ---------------------------------------------------------------------------
# ESKF class tests
# ---------------------------------------------------------------------------


def test_identity_initialization():
    """Default ESKF state should be at origin with identity rotation."""
    ekf = ESKF()
    pose = ekf.get_pose()
    assert pose.shape == (4, 4)
    np.testing.assert_array_almost_equal(pose, np.eye(4))


def test_get_pose_returns_se3():
    """get_pose() should return valid SE(3): R orthogonal, det=1, last row [0,0,0,1]."""
    ekf = ESKF()
    T = np.eye(4)
    T[0, 3] = 5.0
    ekf.initialize_from_pose(T)

    pose = ekf.get_pose()
    R = pose[:3, :3]
    np.testing.assert_array_almost_equal(R @ R.T, np.eye(3), decimal=10)
    assert abs(np.linalg.det(R) - 1.0) < 1e-10
    np.testing.assert_array_equal(pose[3, :], [0, 0, 0, 1])


def test_predict_constant_velocity():
    """Prediction with velocity should advance position."""
    ekf = ESKF()
    ekf.position = np.array([0.0, 0.0, 0.0])
    ekf.velocity = np.array([1.0, 0.0, 0.0])

    ekf.predict(dt=1.0)
    np.testing.assert_array_almost_equal(ekf.position, [1.0, 0.0, 0.0])

    ekf.predict(dt=0.5)
    np.testing.assert_array_almost_equal(ekf.position, [1.5, 0.0, 0.0])


def test_update_corrects_position():
    """Update should pull position toward measurement."""
    ekf = ESKF(measurement_noise_pos=0.01)  # trust measurement
    ekf.position = np.array([0.0, 0.0, 0.0])
    ekf.P = np.eye(9) * 10.0  # high uncertainty → trust measurement more

    T_meas = np.eye(4)
    T_meas[0, 3] = 5.0  # measurement at x=5

    ekf.update(T_meas)
    # Position should move significantly toward 5.0
    assert ekf.position[0] > 3.0


def test_update_corrects_rotation():
    """Update should pull rotation toward measurement."""
    ekf = ESKF(measurement_noise_rot=0.001)
    ekf.P = np.eye(9) * 10.0

    # Measurement: 30-degree rotation around z
    angle = np.pi / 6
    T_meas = np.eye(4)
    T_meas[0, 0] = np.cos(angle)
    T_meas[0, 1] = -np.sin(angle)
    T_meas[1, 0] = np.sin(angle)
    T_meas[1, 1] = np.cos(angle)

    ekf.update(T_meas)

    # Recovered rotation should be close to measurement
    R_est = matrix_from_quaternion(ekf.quaternion)
    np.testing.assert_array_almost_equal(R_est, T_meas[:3, :3], decimal=1)


def test_run_returns_correct_count():
    """run() output length should match input length."""
    poses = [np.eye(4) for _ in range(10)]
    timestamps = np.linspace(0, 1, 10)

    ekf = ESKF()
    result = ekf.run(poses, timestamps)
    assert len(result) == 10


def test_run_smooths_noisy_trajectory():
    """ESKF should reduce trajectory noise compared to input."""
    n = 50
    clean_poses = []
    noisy_poses = []
    for i in range(n):
        T = np.eye(4)
        T[0, 3] = float(i) * 0.5  # steady motion along x
        clean_poses.append(T)

        T_noisy = T.copy()
        T_noisy[0, 3] += np.random.normal(0, 0.3)  # add noise
        T_noisy[1, 3] += np.random.normal(0, 0.3)
        noisy_poses.append(T_noisy)

    timestamps = np.linspace(0, 5, n)

    ekf = ESKF(
        process_noise_pos=0.1,
        process_noise_vel=0.5,
        measurement_noise_pos=0.3,
    )
    smoothed = ekf.run(noisy_poses, timestamps)

    # Compute total error before and after
    err_noisy = sum(np.linalg.norm(noisy_poses[i][:3, 3] - clean_poses[i][:3, 3]) for i in range(n))
    err_smooth = sum(np.linalg.norm(smoothed[i][:3, 3] - clean_poses[i][:3, 3]) for i in range(n))
    assert err_smooth < err_noisy, f"ESKF should reduce noise: {err_noisy:.2f} → {err_smooth:.2f}"
