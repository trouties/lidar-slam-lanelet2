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


def load_oxts(path: Path) -> dict:
    """Load GPS/IMU (OxTS) data from a text file.

    Args:
        path: Path to OxTS .txt file.

    Returns:
        Dictionary with lat, lon, alt, roll, pitch, yaw, and velocity fields.
    """
    pass


def load_calibration(path: Path) -> dict[str, np.ndarray]:
    """Load calibration matrices from KITTI calib.txt.

    Args:
        path: Path to calib.txt.

    Returns:
        Dictionary mapping matrix names (P0..P3, Tr) to 4x4 numpy arrays.
    """
    pass


class KITTIDataset:
    """KITTI odometry dataset wrapper.

    Provides indexed access to point clouds, GPS poses, and IMU data
    for a given sequence.
    """

    def __init__(self, root_path: str | Path, sequence: str = "00") -> None:
        """Initialize dataset.

        Args:
            root_path: Root path to KITTI odometry dataset.
            sequence: Sequence number (e.g. "00", "01", ..., "21").
        """
        self.root = Path(root_path).expanduser()
        self.sequence = sequence
        self.velodyne_dir = self.root / "sequences" / sequence / "velodyne"
        self.oxts_dir = self.root / "sequences" / sequence / "oxts" / "data"
        self.calib_path = self.root / "sequences" / sequence / "calib.txt"

        if self.velodyne_dir.exists():
            self.scan_files = sorted(self.velodyne_dir.glob("*.bin"))
        else:
            self.scan_files = []

    def __len__(self) -> int:
        return len(self.scan_files)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, dict | None, dict | None]:
        """Get a single frame.

        Args:
            idx: Frame index.

        Returns:
            Tuple of (pointcloud, gps_pose, imu_data).
            gps_pose and imu_data may be None if not available.
        """
        pointcloud = load_velodyne_bin(self.scan_files[idx])

        gps_pose = None
        oxts_file = self.oxts_dir / f"{idx:010d}.txt"
        if oxts_file.exists():
            gps_pose = load_oxts(oxts_file)

        return pointcloud, gps_pose, None
