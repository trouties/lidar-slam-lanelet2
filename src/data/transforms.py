"""Coordinate transforms and point cloud operations."""

from __future__ import annotations

import math

import numpy as np


def latlon_to_mercator(lat: float, lon: float) -> tuple[float, float]:
    """Convert GPS latitude/longitude to local Mercator coordinates.

    Uses a simple Mercator projection suitable for small-area mapping.

    Args:
        lat: Latitude in degrees.
        lon: Longitude in degrees.

    Returns:
        Tuple of (x, y) in meters.
    """
    earth_radius = 6378137.0  # WGS84 equatorial radius
    x = math.radians(lon) * earth_radius
    y = math.log(math.tan(math.pi / 4 + math.radians(lat) / 2)) * earth_radius
    return x, y


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
