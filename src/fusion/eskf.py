"""Error-State Kalman Filter (ESKF) for pose smoothing.

Fuses sequential pose observations using a constant-velocity motion model.
Uses error-state formulation to handle SO(3) rotation properly.

Nominal state: position(3) + velocity(3) + quaternion(4) = 10 dims
Error state:   δposition(3) + δvelocity(3) + δrotation(3) = 9 dims
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Quaternion utilities (wxyz convention)
# ---------------------------------------------------------------------------


def quaternion_from_matrix(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to unit quaternion [w, x, y, z]."""
    tr = np.trace(R)
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def matrix_from_quaternion(q: np.ndarray) -> np.ndarray:
    """Convert unit quaternion [w, x, y, z] to 3x3 rotation matrix."""
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two quaternions [w, x, y, z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def small_angle_quaternion(delta_theta: np.ndarray) -> np.ndarray:
    """Convert small rotation vector to quaternion [w, x, y, z].

    Uses first-order approximation: q ≈ [1, δθ/2] normalized.
    """
    q = np.array([1.0, delta_theta[0] / 2, delta_theta[1] / 2, delta_theta[2] / 2])
    return q / np.linalg.norm(q)


def rotation_error(R_meas: np.ndarray, R_nom: np.ndarray) -> np.ndarray:
    """Compute rotation error as a 3-vector in tangent space.

    Returns the rotation vector δθ such that R_meas ≈ R_nom @ exp(δθ).
    Uses logarithmic map approximation for small angles.
    """
    R_err = R_nom.T @ R_meas
    # Rodrigues: extract angle-axis from rotation matrix
    angle = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1))
    if abs(angle) < 1e-10:
        return np.zeros(3)
    axis = np.array(
        [
            R_err[2, 1] - R_err[1, 2],
            R_err[0, 2] - R_err[2, 0],
            R_err[1, 0] - R_err[0, 1],
        ]
    ) / (2 * np.sin(angle))
    return axis * angle


# ---------------------------------------------------------------------------
# ESKF
# ---------------------------------------------------------------------------


class ESKF:
    """Error-State Kalman Filter for pose smoothing.

    Uses a constant-velocity motion model for prediction and
    6-DOF pose observations for update.
    """

    def __init__(
        self,
        process_noise_pos: float = 0.1,
        process_noise_vel: float = 0.5,
        process_noise_rot: float = 0.01,
        measurement_noise_pos: float = 0.05,
        measurement_noise_rot: float = 0.01,
    ) -> None:
        """Initialize ESKF.

        Args:
            process_noise_pos: Position process noise σ (m).
            process_noise_vel: Velocity process noise σ (m/s).
            process_noise_rot: Rotation process noise σ (rad).
            measurement_noise_pos: Position measurement noise σ (m).
            measurement_noise_rot: Rotation measurement noise σ (rad).
        """
        self.sigma_p = process_noise_pos
        self.sigma_v = process_noise_vel
        self.sigma_r = process_noise_rot
        self.sigma_mp = measurement_noise_pos
        self.sigma_mr = measurement_noise_rot

        # Nominal state
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.quaternion = np.array([1.0, 0.0, 0.0, 0.0])  # wxyz, identity

        # Error-state covariance (9x9)
        self.P = np.eye(9) * 1.0

    def initialize_from_pose(self, pose: np.ndarray) -> None:
        """Initialize nominal state from a 4x4 SE(3) matrix."""
        self.position = pose[:3, 3].copy()
        self.quaternion = quaternion_from_matrix(pose[:3, :3])
        self.velocity = np.zeros(3)

    def predict(self, dt: float) -> None:
        """Constant-velocity prediction step.

        Propagates nominal state and error-state covariance.
        """
        # Propagate nominal state
        self.position += self.velocity * dt
        # velocity and quaternion unchanged (constant velocity, no angular rate)

        # State transition for error state: F (9x9)
        F = np.eye(9)
        F[0:3, 3:6] = np.eye(3) * dt  # δp += δv * dt

        # Process noise Q (9x9)
        Q = np.zeros((9, 9))
        Q[0:3, 0:3] = np.eye(3) * (self.sigma_p * dt) ** 2
        Q[3:6, 3:6] = np.eye(3) * (self.sigma_v * dt) ** 2
        Q[6:9, 6:9] = np.eye(3) * (self.sigma_r * dt) ** 2

        # Covariance propagation
        self.P = F @ self.P @ F.T + Q

    def update(self, pose_measurement: np.ndarray) -> None:
        """Update with a 4x4 SE(3) pose observation.

        Computes innovation, Kalman gain, updates error state,
        then injects into nominal state.
        """
        # Measurement: position + rotation error
        z_pos = pose_measurement[:3, 3]
        R_meas = pose_measurement[:3, :3]
        R_nom = matrix_from_quaternion(self.quaternion)

        # Innovation (6x1): [position_error, rotation_error]
        y = np.zeros(6)
        y[0:3] = z_pos - self.position
        y[3:6] = rotation_error(R_meas, R_nom)

        # Observation matrix H (6x9): maps error state → observation
        H = np.zeros((6, 9))
        H[0:3, 0:3] = np.eye(3)  # position
        H[3:6, 6:9] = np.eye(3)  # rotation

        # Measurement noise R (6x6)
        R = np.zeros((6, 6))
        R[0:3, 0:3] = np.eye(3) * self.sigma_mp**2
        R[3:6, 3:6] = np.eye(3) * self.sigma_mr**2

        # Kalman gain
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        # Error state update
        dx = K @ y

        # Inject into nominal state
        self.position += dx[0:3]
        self.velocity += dx[3:6]
        delta_q = small_angle_quaternion(dx[6:9])
        self.quaternion = quaternion_multiply(self.quaternion, delta_q)
        self.quaternion /= np.linalg.norm(self.quaternion)

        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(9) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

    def get_pose(self) -> np.ndarray:
        """Return current estimate as a 4x4 SE(3) matrix."""
        T = np.eye(4)
        T[:3, :3] = matrix_from_quaternion(self.quaternion)
        T[:3, 3] = self.position
        return T

    def run(
        self,
        poses: list[np.ndarray],
        timestamps: np.ndarray | list[float],
    ) -> list[np.ndarray]:
        """Run ESKF over a full trajectory.

        Args:
            poses: List of 4x4 SE(3) pose observations.
            timestamps: Per-frame timestamps in seconds.

        Returns:
            List of smoothed 4x4 SE(3) poses.
        """
        timestamps = np.asarray(timestamps)
        n = len(poses)

        # Initialize from first pose
        self.initialize_from_pose(poses[0])
        # Estimate initial velocity from first two poses
        if n >= 2:
            dt0 = timestamps[1] - timestamps[0]
            if dt0 > 0:
                self.velocity = (poses[1][:3, 3] - poses[0][:3, 3]) / dt0

        smoothed = [self.get_pose()]

        for i in range(1, n):
            dt = float(timestamps[i] - timestamps[i - 1])
            if dt <= 0:
                dt = 0.1  # fallback

            self.predict(dt)
            self.update(poses[i])
            smoothed.append(self.get_pose())

        return smoothed
