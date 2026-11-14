"""KITTI dataset loader for LiDAR point clouds, GPS/IMU, and calibration data."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_velodyne_bin(path: Path) -> np.ndarray:
    """Load a Velodyne binary point cloud file.

    Args:
        path: Path to .bin file.

    Returns:
        Point cloud as (N, 4) array [x, y, z, reflectance].
    """
    points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    return points


# The HDL-64E intrinsic calibration shipped with KITTI places every beam
# slightly too low, so flat ground appears to drop by ~3.6 mm per metre of
# range. Each scan-to-map registration absorbs part of that distortion as a
# pitch error, and the errors accumulate into vertical drift. The value is the
# one KISS-ICP applies in its own KITTI dataloader.
KITTI_VERTICAL_ANGLE_OFFSET_DEG = 0.205


def correct_kitti_scan(points: np.ndarray) -> np.ndarray:
    """Raise every point's elevation angle by ``KITTI_VERTICAL_ANGLE_OFFSET_DEG``.

    Range and azimuth are unchanged; extra columns (reflectance) are copied.

    Args:
        points: ``(N, 3)`` or ``(N, 4)`` array in the Velodyne frame.

    Returns:
        Array of the same shape and dtype.
    """
    # Rotating p by delta about the horizontal axis p x z_hat moves it towards
    # the zenith along its own vertical plane. Rodrigues' formula reduces to
    #   x' = x (cos d - sin d z / r_xy),  y' likewise,  z' = z cos d + sin d r_xy
    # so only r_xy is needed per point.
    xyz = points[:, :3].astype(np.float64)
    r_xy = np.hypot(xyz[:, 0], xyz[:, 1])
    c = np.cos(np.radians(KITTI_VERTICAL_ANGLE_OFFSET_DEG))
    s = np.sin(np.radians(KITTI_VERTICAL_ANGLE_OFFSET_DEG))
    # Points on the sensor's z-axis have no defined azimuth; leave them as is.
    on_axis = r_xy <= 0.0
    z_over_r = np.divide(xyz[:, 2], r_xy, out=np.zeros_like(r_xy), where=~on_axis)
    horizontal_scale = np.where(on_axis, 1.0, c - s * z_over_r)
    out = points.copy()
    out[:, 0] = xyz[:, 0] * horizontal_scale
    out[:, 1] = xyz[:, 1] * horizontal_scale
    out[:, 2] = np.where(on_axis, xyz[:, 2], xyz[:, 2] * c + s * r_xy)
    return out


def load_calibration(path: Path) -> dict[str, np.ndarray]:
    """Load calibration matrices from KITTI calib.txt.

    Each line has format "KEY: v1 v2 ... v12" representing a 3x4 matrix.
    Returns 4x4 matrices with [0, 0, 0, 1] appended as the last row.

    Args:
        path: Path to calib.txt.

    Returns:
        Dictionary mapping matrix names (P0..P3, Tr) to 4x4 numpy arrays.
    """
    calibration = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            key, _, values = line.partition(":")
            mat = np.eye(4)
            mat[:3, :] = np.fromstring(values, sep=" ").reshape(3, 4)
            calibration[key.strip()] = mat
    return calibration


def load_poses(path: Path) -> np.ndarray:
    """Load ground truth poses from a KITTI poses file.

    Each line contains 12 space-separated floats representing the 3x4
    row-major transformation matrix. Only sequences 00-10 have ground truth.

    Args:
        path: Path to poses .txt file (e.g., poses/00.txt).

    Returns:
        Array of shape (M, 4, 4) containing SE(3) transformation matrices.
    """
    raw = np.loadtxt(path)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    n = raw.shape[0]
    poses = np.zeros((n, 4, 4))
    poses[:, :3, :] = raw.reshape(n, 3, 4)
    poses[:, 3, 3] = 1.0
    return poses


def load_timestamps(path: Path) -> np.ndarray:
    """Load timestamps from KITTI times.txt.

    Args:
        path: Path to times.txt (one float per line, seconds from start).

    Returns:
        1D array of timestamps in seconds.
    """
    return np.atleast_1d(np.loadtxt(path))


class KITTIDataset:
    """KITTI odometry dataset wrapper.

    Provides indexed access to point clouds, ground truth poses,
    timestamps, and calibration for a given sequence.
    """

    def __init__(
        self, root_path: str | Path, sequence: str = "00", correct_scan: bool = False
    ) -> None:
        """Initialize dataset.

        Args:
            root_path: Root path to KITTI odometry dataset.
            sequence: Sequence number (e.g. "00", "01", ..., "21").
            correct_scan: Apply :func:`correct_kitti_scan` to every scan.
        """
        self.root = Path(root_path).expanduser()
        self.sequence = sequence
        self.correct_scan = correct_scan

        seq_dir = self.root / "sequences" / sequence
        self.velodyne_dir = seq_dir / "velodyne"
        self.calib_path = seq_dir / "calib.txt"
        self.times_path = seq_dir / "times.txt"
        self.poses_path = self.root / "poses" / f"{sequence}.txt"

        if self.velodyne_dir.exists():
            self.scan_files = sorted(self.velodyne_dir.glob("*.bin"))
        else:
            self.scan_files = []

        # Eagerly load small metadata files
        self.calibration = load_calibration(self.calib_path) if self.calib_path.exists() else None
        self.timestamps = load_timestamps(self.times_path) if self.times_path.exists() else None
        self.poses = load_poses(self.poses_path) if self.poses_path.exists() else None

    @classmethod
    def from_config(cls, data_cfg: dict, sequence: str) -> KITTIDataset:
        """Build from the ``data`` section of a pipeline config."""
        return cls(
            root_path=data_cfg["kitti_root"],
            sequence=sequence,
            correct_scan=bool(data_cfg.get("correct_kitti_scan", False)),
        )

    def __len__(self) -> int:
        return len(self.scan_files)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray | None, float | None]:
        """Get a single frame.

        Args:
            idx: Frame index.

        Returns:
            Tuple of (pointcloud, pose, timestamp).
            - pointcloud: (N, 4) array [x, y, z, reflectance]
            - pose: (4, 4) ground truth SE(3) matrix, or None if unavailable
            - timestamp: seconds from start, or None if unavailable
        """
        pointcloud = load_velodyne_bin(self.scan_files[idx])
        if self.correct_scan:
            pointcloud = correct_kitti_scan(pointcloud)

        pose = None
        if self.poses is not None and idx < len(self.poses):
            pose = self.poses[idx]

        timestamp = None
        if self.timestamps is not None and idx < len(self.timestamps):
            timestamp = float(self.timestamps[idx])

        return pointcloud, pose, timestamp
