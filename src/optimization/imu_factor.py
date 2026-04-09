"""GTSAM IMU preintegration factor for tightly-coupled LiDAR-inertial fusion.

Implements the Forster 2017 IJRR preintegration formulation via
``gtsam.PreintegratedImuMeasurements``, inserting ``ImuFactor`` between
consecutive LiDAR keyframes in the pose graph.
"""

from __future__ import annotations

import gtsam
import numpy as np


def make_preintegration_params(
    gravity: float = 9.81,
    accel_noise_sigma: float = 5.0,
    gyro_noise_sigma: float = 0.5,
    accel_bias_sigma: float = 0.1,
    gyro_bias_sigma: float = 0.01,
    integration_sigma: float = 1e-3,
) -> gtsam.PreintegrationParams:
    """Create GTSAM preintegration parameters.

    Gravity points downward (-z in NED / Velodyne z-up convention).
    """
    params = gtsam.PreintegrationParams.MakeSharedU(gravity)
    params.setAccelerometerCovariance(np.eye(3) * accel_noise_sigma**2)
    params.setGyroscopeCovariance(np.eye(3) * gyro_noise_sigma**2)
    params.setIntegrationCovariance(np.eye(3) * integration_sigma**2)
    return params


class ImuPreintegrator:
    """Accumulates IMU measurements between two LiDAR keyframes.

    Usage::

        preint = ImuPreintegrator(params)
        for each IMU sample between LiDAR frames i and i+1:
            preint.add(acc, gyro, dt)
        factor = preint.make_factor(pose_i, vel_i, pose_j, vel_j, bias_i)
        preint.reset()
    """

    def __init__(
        self,
        params: gtsam.PreintegrationParams | None = None,
        initial_bias: gtsam.imuBias.ConstantBias | None = None,
    ) -> None:
        if params is None:
            params = make_preintegration_params()
        if initial_bias is None:
            initial_bias = gtsam.imuBias.ConstantBias()
        self._params = params
        self._bias = initial_bias
        self._pim = gtsam.PreintegratedImuMeasurements(params, initial_bias)

    def add(self, acc: np.ndarray, gyro: np.ndarray, dt: float) -> None:
        """Add a single IMU measurement.

        Args:
            acc: ``(3,)`` accelerometer [ax, ay, az] in m/s².
            gyro: ``(3,)`` gyroscope [wx, wy, wz] in rad/s.
            dt: Time interval in seconds.
        """
        self._pim.integrateMeasurement(acc, gyro, dt)

    def reset(self, bias: gtsam.imuBias.ConstantBias | None = None) -> None:
        """Reset the preintegrator for the next interval."""
        if bias is not None:
            self._bias = bias
        self._pim.resetIntegrationAndSetBias(self._bias)

    @property
    def preintegrated(self) -> gtsam.PreintegratedImuMeasurements:
        return self._pim

    def make_factor(
        self,
        pose_key_i: int,
        vel_key_i: int,
        pose_key_j: int,
        vel_key_j: int,
        bias_key: int,
    ) -> gtsam.ImuFactor:
        """Create a GTSAM ImuFactor from the accumulated measurements."""
        return gtsam.ImuFactor(
            pose_key_i,
            vel_key_i,
            pose_key_j,
            vel_key_j,
            bias_key,
            self._pim,
        )


def build_tight_coupled_graph(
    poses: list[np.ndarray],
    imu_acc: np.ndarray,
    imu_gyro: np.ndarray,
    imu_timestamps: np.ndarray,
    lidar_timestamps: np.ndarray,
    odom_sigmas: list[float] | None = None,
    prior_sigmas: list[float] | None = None,
    loop_closure_sigmas: list[float] | None = None,
    prior_indices: list[int] | None = None,
    gt_poses: list[np.ndarray] | None = None,
    loop_closures: list[tuple[int, int, np.ndarray]] | None = None,
    accel_noise_sigma: float = 5.0,
    gyro_noise_sigma: float = 0.5,
    accel_bias_sigma: float = 0.1,
    gyro_bias_sigma: float = 0.01,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Build and optimize a tightly-coupled LiDAR-IMU pose graph.

    Combines LiDAR BetweenFactors with IMU preintegration factors.

    Args:
        poses: Estimated LiDAR poses (list of 4×4).
        imu_acc: ``(M, 3)`` accelerometer data.
        imu_gyro: ``(M, 3)`` gyroscope data.
        imu_timestamps: ``(M,)`` IMU timestamps in seconds.
        lidar_timestamps: ``(N,)`` LiDAR frame timestamps in seconds.
        odom_sigmas: LiDAR odometry noise sigmas [tx,ty,tz,rx,ry,rz].
        prior_sigmas: Prior noise sigmas [tx,ty,tz,rx,ry,rz].
        loop_closure_sigmas: Loop closure noise sigmas [tx,ty,tz,rx,ry,rz].
            ICP-verified closures are typically tighter than odometry.
            Falls back to odom_sigmas when None.
        prior_indices: Frame indices for absolute priors (GNSS denial).
        gt_poses: Ground-truth poses for priors.
        accel_noise_sigma: Accelerometer noise σ (m/s²).
        gyro_noise_sigma: Gyroscope noise σ (rad/s).
        accel_bias_sigma: Accelerometer bias random walk σ.
        gyro_bias_sigma: Gyroscope bias random walk σ.

    Returns:
        Tuple of (optimized_poses, bias_history).
    """
    n = len(poses)
    if odom_sigmas is None:
        odom_sigmas = [0.1, 0.1, 0.1, 0.01, 0.01, 0.01]
    if prior_sigmas is None:
        prior_sigmas = [0.01, 0.01, 0.01, 0.001, 0.001, 0.001]

    # GTSAM sigma reorder: config [tx,ty,tz,rx,ry,rz] → [rx,ry,rz,tx,ty,tz]
    odom_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([*odom_sigmas[3:], *odom_sigmas[:3]]))
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([*prior_sigmas[3:], *prior_sigmas[:3]]))
    _lc_sigmas = loop_closure_sigmas if loop_closure_sigmas is not None else odom_sigmas
    lc_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([*_lc_sigmas[3:], *_lc_sigmas[:3]]))

    # Key scheme: pose=P(i), velocity=V(i), bias=B(i)
    def P(i):
        return gtsam.symbol("x", i)

    def V(i):
        return gtsam.symbol("v", i)

    def B(i):
        return gtsam.symbol("b", i)

    graph = gtsam.NonlinearFactorGraph()
    values = gtsam.Values()

    # IMU preintegration setup
    imu_params = make_preintegration_params(
        accel_noise_sigma=accel_noise_sigma,
        gyro_noise_sigma=gyro_noise_sigma,
    )
    preintegrator = ImuPreintegrator(imu_params)

    # Bias noise model
    bias_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([accel_bias_sigma] * 3 + [gyro_bias_sigma] * 3)
    )
    zero_bias = gtsam.imuBias.ConstantBias()

    # Prior set
    if prior_indices is None:
        prior_set = {0}
    else:
        prior_set = set(prior_indices)
        prior_set.add(0)
    prior_source = gt_poses if gt_poses is not None else poses

    # Velocity noise for prior
    vel_noise = gtsam.noiseModel.Isotropic.Sigma(3, 1.0)

    # Insert all nodes
    for i in range(n):
        pose_i = gtsam.Pose3(poses[i])
        values.insert(P(i), pose_i)
        # Estimate velocity from consecutive poses
        if i > 0:
            dt = float(lidar_timestamps[i] - lidar_timestamps[i - 1])
            if dt <= 0:
                dt = 0.1
            v = (poses[i][:3, 3] - poses[i - 1][:3, 3]) / dt
        else:
            v = np.zeros(3)
        values.insert(V(i), v)
        values.insert(B(i), zero_bias)

    # Add priors
    for idx in sorted(prior_set):
        if 0 <= idx < n:
            graph.add(gtsam.PriorFactorPose3(P(idx), gtsam.Pose3(prior_source[idx]), prior_noise))

    # Prior on initial velocity and bias
    graph.add(gtsam.PriorFactorVector(V(0), np.zeros(3), vel_noise))
    bias_prior_noise = gtsam.noiseModel.Isotropic.Sigma(6, 0.1)
    graph.add(gtsam.PriorFactorConstantBias(B(0), zero_bias, bias_prior_noise))

    # Add LiDAR odometry between factors and IMU factors
    for i in range(1, n):
        # LiDAR between factor
        delta = np.linalg.inv(poses[i - 1]) @ poses[i]
        graph.add(gtsam.BetweenFactorPose3(P(i - 1), P(i), gtsam.Pose3(delta), odom_noise))

        # IMU preintegration between LiDAR frames i-1 and i
        t_start = float(lidar_timestamps[i - 1])
        t_end = float(lidar_timestamps[i])

        # Find IMU samples in [t_start, t_end)
        mask = (imu_timestamps >= t_start) & (imu_timestamps < t_end)
        imu_indices = np.where(mask)[0]

        preintegrator.reset()
        has_imu = False
        if len(imu_indices) > 0:
            for j_idx in range(len(imu_indices)):
                j = imu_indices[j_idx]
                if j_idx == 0:
                    dt_imu = float(imu_timestamps[j] - t_start)
                else:
                    dt_imu = float(imu_timestamps[j] - imu_timestamps[imu_indices[j_idx - 1]])
                if dt_imu <= 0:
                    dt_imu = 0.01
                preintegrator.add(imu_acc[j], imu_gyro[j], dt_imu)

            # Add remaining dt to t_end
            dt_tail = t_end - float(imu_timestamps[imu_indices[-1]])
            if dt_tail > 0.001:
                preintegrator.add(imu_acc[imu_indices[-1]], imu_gyro[imu_indices[-1]], dt_tail)

            graph.add(preintegrator.make_factor(P(i - 1), V(i - 1), P(i), V(i), B(i - 1)))
            has_imu = True

        if not has_imu:
            # No IMU data for this interval — add velocity prior to prevent
            # unconstrained V(i) from making the system singular.
            graph.add(gtsam.PriorFactorVector(V(i), values.atVector(V(i)), vel_noise))

        # Bias between factor (constant bias assumption)
        graph.add(gtsam.BetweenFactorConstantBias(B(i - 1), B(i), zero_bias, bias_noise))

    # Add loop closure factors
    if loop_closures:
        for lc_i, lc_j, rel_pose in loop_closures:
            graph.add(gtsam.BetweenFactorPose3(P(lc_i), P(lc_j), gtsam.Pose3(rel_pose), lc_noise))

    # Optimize
    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(100)
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, values, params)
    result = optimizer.optimize()

    # Extract results
    optimized_poses = []
    bias_history = []
    for i in range(n):
        optimized_poses.append(result.atPose3(P(i)).matrix())
        b = result.atConstantBias(B(i))
        bias_history.append(np.concatenate([b.accelerometer(), b.gyroscope()]))

    return optimized_poses, bias_history
