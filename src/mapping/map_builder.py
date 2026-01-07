"""Point cloud map construction.

Accumulates registered point clouds into a global map using
streaming voxel aggregation with exact per-voxel means.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d

from src.data.transforms import apply_transform


def _voxel_aggregate(
    points: np.ndarray,
    intensities: np.ndarray,
    voxel_size: float,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Group points into voxels and return per-voxel weighted means.

    When ``weights`` is ``None`` each row is treated as a single raw point
    (weight = 1), so this behaves like a standard voxel downsample and
    reproduces Open3D's mean-over-voxel output. When ``weights`` is given,
    the inputs are interpreted as previously-aggregated voxel centroids
    with associated counts, and the output is the correctly-weighted
    merge — which makes the function usable in a streaming fashion.

    Args:
        points: (N, 3) float32/float64 world-frame XYZ.
        intensities: (N,) reflectance aligned with ``points``.
        voxel_size: Voxel edge length (m).
        weights: Optional (N,) per-row weight (count of underlying raw
            points that ``points[i]`` represents).

    Returns:
        Tuple ``(mean_points, mean_intensities, total_weights)`` where the
        first two are float32 and the last is float32.
    """
    if points.shape[0] == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    # Pack (x, y, z) voxel index into a single int64 key without
    # materializing a (N, 3) int64 intermediate. ±2^20 voxels per axis
    # (±~100 km at voxel=0.1m) is far beyond KITTI sequences.
    bias = 1 << 20
    inv_voxel = 1.0 / voxel_size

    keys = np.floor(points[:, 0] * inv_voxel).astype(np.int64)
    keys += bias
    keys <<= 42

    tmp = np.floor(points[:, 1] * inv_voxel).astype(np.int64)
    tmp += bias
    tmp <<= 21
    keys |= tmp

    tmp = np.floor(points[:, 2] * inv_voxel).astype(np.int64)
    tmp += bias
    keys |= tmp
    del tmp

    unique_keys, inverse = np.unique(keys, return_inverse=True)
    del keys
    n_voxels = unique_keys.shape[0]
    del unique_keys

    # Unweighted path: points themselves are the weights.
    if weights is None:
        total_w = np.bincount(inverse, minlength=n_voxels).astype(np.float64)
        sum_x = np.bincount(inverse, weights=points[:, 0], minlength=n_voxels)
        sum_y = np.bincount(inverse, weights=points[:, 1], minlength=n_voxels)
        sum_z = np.bincount(inverse, weights=points[:, 2], minlength=n_voxels)
        sum_i = np.bincount(inverse, weights=intensities, minlength=n_voxels)
    else:
        # Weighted path: reuse a single scratch buffer for (axis * weight).
        w = weights.astype(np.float64, copy=False)
        total_w = np.bincount(inverse, weights=w, minlength=n_voxels)
        scratch = np.empty(points.shape[0], dtype=np.float64)
        np.multiply(points[:, 0], w, out=scratch)
        sum_x = np.bincount(inverse, weights=scratch, minlength=n_voxels)
        np.multiply(points[:, 1], w, out=scratch)
        sum_y = np.bincount(inverse, weights=scratch, minlength=n_voxels)
        np.multiply(points[:, 2], w, out=scratch)
        sum_z = np.bincount(inverse, weights=scratch, minlength=n_voxels)
        np.multiply(intensities, w, out=scratch)
        sum_i = np.bincount(inverse, weights=scratch, minlength=n_voxels)
        del scratch, w

    del inverse

    sum_x /= total_w
    sum_y /= total_w
    sum_z /= total_w
    sum_i /= total_w

    mean_xyz = np.empty((n_voxels, 3), dtype=np.float32)
    mean_xyz[:, 0] = sum_x
    mean_xyz[:, 1] = sum_y
    mean_xyz[:, 2] = sum_z
    mean_i = sum_i.astype(np.float32)
    total_w_f32 = total_w.astype(np.float32)
    del sum_x, sum_y, sum_z, sum_i, total_w

    return mean_xyz, mean_i, total_w_f32


class MapBuilder:
    """Accumulate per-frame point clouds into a global map.

    Each call to :py:meth:`add_frame` immediately voxel-aggregates the
    frame (a small, fast operation). The per-frame aggregates pile up in
    an accumulator; every ``downsample_every`` frames the accumulator is
    consolidated into a single aggregate, and the same consolidation
    runs once more in :py:meth:`finalize`. This avoids re-sorting a
    large growing state on every frame and keeps peak memory bounded.

    Reflectance (intensity) is averaged per voxel alongside XYZ. The
    final :py:meth:`finalize` returns an Open3D point cloud with
    intensity encoded in the ``colors`` channel as ``[I, I, I]`` so
    downstream code can read it via ``np.asarray(pcd.colors)[:, 0]``.
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
            downsample_every: Trigger a consolidation merge every N frames.
        """
        self.voxel_size = voxel_size
        self.max_range = max_range
        self.downsample_every = downsample_every

        self._points_chunks: list[np.ndarray] = []  # list of (n_i, 3) float32
        self._intensity_chunks: list[np.ndarray] = []  # list of (n_i,)   float32
        self._weight_chunks: list[np.ndarray] = []  # list of (n_i,)   float32
        self._frames_since_merge = 0

    def add_frame(self, points: np.ndarray, pose: np.ndarray) -> None:
        """Transform a frame to world, voxel-aggregate, and append.

        Args:
            points: (N, 4) [x, y, z, reflectance] in Velodyne frame.
            pose: (4, 4) SE(3) world pose of the sensor for this frame.
        """
        if points.shape[0] == 0:
            return

        # Range filter in the sensor frame (before world transform).
        xyz_sensor = points[:, :3]
        ranges = np.linalg.norm(xyz_sensor, axis=1)
        mask = ranges < self.max_range
        if not np.any(mask):
            return
        filtered = points[mask]

        # Transform (N, 4) → world. apply_transform preserves reflectance.
        world = apply_transform(filtered, pose)

        frame_xyz = np.ascontiguousarray(world[:, :3], dtype=np.float32)
        frame_i = np.ascontiguousarray(world[:, 3], dtype=np.float32)

        mean_xyz, mean_i, counts = _voxel_aggregate(
            frame_xyz, frame_i, self.voxel_size, weights=None
        )

        self._points_chunks.append(mean_xyz)
        self._intensity_chunks.append(mean_i)
        self._weight_chunks.append(counts)
        self._frames_since_merge += 1

        if self._frames_since_merge >= self.downsample_every:
            self._merge()

    def _merge(self) -> None:
        """Collapse all pending aggregates into a single aggregate."""
        if not self._points_chunks:
            self._frames_since_merge = 0
            return

        if len(self._points_chunks) == 1:
            self._frames_since_merge = 0
            return

        all_xyz = np.concatenate(self._points_chunks, axis=0)
        all_i = np.concatenate(self._intensity_chunks, axis=0)
        all_w = np.concatenate(self._weight_chunks, axis=0)

        # Release the per-chunk references before the heavy step.
        self._points_chunks.clear()
        self._intensity_chunks.clear()
        self._weight_chunks.clear()

        merged_xyz, merged_i, merged_w = _voxel_aggregate(
            all_xyz, all_i, self.voxel_size, weights=all_w
        )
        del all_xyz, all_i, all_w

        self._points_chunks.append(merged_xyz)
        self._intensity_chunks.append(merged_i)
        self._weight_chunks.append(merged_w)
        self._frames_since_merge = 0

    def finalize(self) -> o3d.geometry.PointCloud:
        """Merge everything and return the global map as an Open3D cloud."""
        self._merge()

        if not self._points_chunks:
            return o3d.geometry.PointCloud()

        # After _merge there is exactly one chunk left.
        final_points = self._points_chunks[0].astype(np.float64, copy=False)
        final_intens = self._intensity_chunks[0].astype(np.float64, copy=False)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(final_points)
        colors = np.tile(final_intens[:, None], (1, 3))
        pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd

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

    @staticmethod
    def downsample_existing(
        pcd: o3d.geometry.PointCloud, voxel_size: float
    ) -> o3d.geometry.PointCloud:
        """Re-voxelize an already-aggregated map to a coarser resolution.

        Used to go from the cached ``master_voxel_size`` (e.g. 0.05 m) master
        cloud to the Stage 5 working ``voxel_size`` (e.g. 0.15 m) without
        re-running the full per-frame accumulation. Reuses the numpy-native
        ``_voxel_aggregate`` for memory safety — ``pipeline-notes.md:82-99``
        forbids ``o3d.voxel_down_sample`` on aggregated state because it
        thrashes sort buffers on large inputs.

        Reflectance is preserved via the ``colors`` channel convention used
        by :py:meth:`finalize`.
        """
        if len(pcd.points) == 0:
            return o3d.geometry.PointCloud()

        points = np.ascontiguousarray(np.asarray(pcd.points), dtype=np.float32)
        colors = np.asarray(pcd.colors)
        if colors.size == 0:
            intensities = np.zeros(points.shape[0], dtype=np.float32)
        else:
            intensities = np.ascontiguousarray(colors[:, 0], dtype=np.float32)

        mean_xyz, mean_i, _ = _voxel_aggregate(points, intensities, voxel_size, weights=None)

        out = o3d.geometry.PointCloud()
        out.points = o3d.utility.Vector3dVector(mean_xyz.astype(np.float64))
        out.colors = o3d.utility.Vector3dVector(
            np.repeat(mean_i.astype(np.float64)[:, None], 3, axis=1)
        )
        return out
