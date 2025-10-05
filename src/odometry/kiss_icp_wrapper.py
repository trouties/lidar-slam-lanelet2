"""KISS-ICP odometry wrapper."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class KissICPOdometry:
    """Wrapper around the KISS-ICP pipeline for LiDAR odometry.

    Encapsulates KISS-ICP configuration and provides a simple interface
    for running odometry on a dataset.
    """

    def __init__(
        self,
        max_range: float = 100.0,
        min_range: float = 5.0,
        voxel_size: float = 1.0,
    ) -> None:
        """Initialize KISS-ICP odometry.

        Args:
            max_range: Maximum point range in meters.
            min_range: Minimum point range in meters.
            voxel_size: Voxel size for downsampling in meters.
        """
        self.max_range = max_range
        self.min_range = min_range
        self.voxel_size = voxel_size

    def run(self, dataset) -> list[np.ndarray]:
        """Run KISS-ICP odometry on a dataset.

        Args:
            dataset: Iterable yielding (pointcloud, ...) tuples.

        Returns:
            List of 4x4 pose matrices (one per frame).
        """
        pass

    @staticmethod
    def save_poses_kitti_format(poses: list[np.ndarray], path: str | Path) -> None:
        """Save poses in KITTI format (12 values per line, row-major 3x4).

        Args:
            poses: List of 4x4 pose matrices.
            path: Output file path.
        """
        path = Path(path)
        with path.open("w") as f:
            for pose in poses:
                row = pose[:3, :].flatten()
                f.write(" ".join(f"{v:.6e}" for v in row) + "\n")
