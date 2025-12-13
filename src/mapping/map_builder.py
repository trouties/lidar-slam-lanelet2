"""Point cloud map construction.

Accumulates registered point clouds into a global map
with voxel downsampling for memory efficiency.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d

from src.data.transforms import apply_transform


class MapBuilder:
    """Accumulate per-frame point clouds into a global map.

    Points are transformed using provided poses, range-filtered,
    and incrementally voxel-downsampled to keep memory bounded.
    Reflectance (intensity) is preserved by encoding it into the
    Open3D ``colors`` channel, which voxel_down_sample averages.
    """

    def __init__(
        self,
        voxel_size: float = 0.1,
        max_range: float = 50.0,
        downsample_every: int = 50,
    ) -> None:
        """Initialize builder.

        Args:
            voxel_size: Voxel edge length (m) for downsampling.
            max_range: Drop points farther than this (m) from the sensor
                before transforming to world. Reduces noise and memory.
            downsample_every: Trigger incremental voxel downsample every N frames.
        """
        self.voxel_size = voxel_size
        self.max_range = max_range
        self.downsample_every = downsample_every

        self._pcd = o3d.geometry.PointCloud()
        self._frames_since_downsample = 0

    def add_frame(self, points: np.ndarray, pose: np.ndarray) -> None:
        """Transform a frame to world and append to the accumulator.

        Args:
            points: (N, 4) [x, y, z, reflectance] in Velodyne frame.
            pose: (4, 4) SE(3) world pose of the sensor for this frame.
        """
        if points.shape[0] == 0:
            return

        # Range filter in the sensor frame (before world transform).
        xyz = points[:, :3]
        ranges = np.linalg.norm(xyz, axis=1)
        mask = ranges < self.max_range
        if not np.any(mask):
            return
        filtered = points[mask]

        # Transform (N, 4) → world. apply_transform preserves reflectance.
        world = apply_transform(filtered, pose)

        world_xyz = world[:, :3].astype(np.float64)
        intensity = world[:, 3].astype(np.float64)
        # Encode intensity in colors channel; voxel_down_sample averages it.
        colors = np.tile(intensity[:, None], (1, 3))

        frame_pcd = o3d.geometry.PointCloud()
        frame_pcd.points = o3d.utility.Vector3dVector(world_xyz)
        frame_pcd.colors = o3d.utility.Vector3dVector(colors)

        self._pcd += frame_pcd
        self._frames_since_downsample += 1

        if self._frames_since_downsample >= self.downsample_every:
            self._pcd = self._pcd.voxel_down_sample(voxel_size=self.voxel_size)
            self._frames_since_downsample = 0

    def finalize(self) -> o3d.geometry.PointCloud:
        """Apply the final voxel downsample and return the global map."""
        self._pcd = self._pcd.voxel_down_sample(voxel_size=self.voxel_size)
        self._frames_since_downsample = 0
        return self._pcd

    def build(
        self,
        dataset,
        poses: list[np.ndarray],
    ) -> o3d.geometry.PointCloud:
        """Run end-to-end: iterate dataset frames and accumulate into a global map.

        Args:
            dataset: Indexable dataset yielding (pointcloud, _, _) tuples,
                where pointcloud is (N, 4) in Velodyne frame.
            poses: List of (4, 4) SE(3) world poses, one per frame.

        Returns:
            Voxel-downsampled Open3D point cloud in world (Velodyne) frame.
        """
        n = min(len(dataset), len(poses))
        for i in range(n):
            points, _, _ = dataset[i]
            self.add_frame(points, poses[i])
        return self.finalize()

    @staticmethod
    def save(pcd: o3d.geometry.PointCloud, path: Path) -> None:
        """Save a point cloud to disk as .pcd (or whatever Open3D infers)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_point_cloud(str(path), pcd)
