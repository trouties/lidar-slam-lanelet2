"""Rigid-body transform of point clouds."""

from __future__ import annotations

import numpy as np


def apply_transform(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply a 4x4 rigid-body transform to a point cloud.

    Args:
        points: (N, 3) or (N, 4) point cloud.
        T: (4, 4) homogeneous transformation matrix.

    Returns:
        Transformed points with same shape as input.
    """
    xyz = points[:, :3]
    ones = np.ones((xyz.shape[0], 1), dtype=xyz.dtype)
    homogeneous = np.hstack([xyz, ones])  # (N, 4)
    transformed = (T @ homogeneous.T).T  # (N, 4)

    if points.shape[1] == 4:
        return np.hstack([transformed[:, :3], points[:, 3:]])
    return transformed[:, :3]
