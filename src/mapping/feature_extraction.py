"""Semantic feature extraction from point cloud maps.

Extracts road boundaries, lane markings, and other HD Map features
from the accumulated point cloud using geometric and intensity cues.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import open3d as o3d


def extract_road_surface(
    points: np.ndarray,
    intensities: np.ndarray,
    z_min: float,
    z_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract candidate road-surface points by height filtering.

    Args:
        points: (N, 3) XYZ in world frame (Velodyne convention, z = up).
        intensities: (N,) reflectance values aligned with ``points``.
        z_min: Lower bound on z (inclusive).
        z_max: Upper bound on z (inclusive).

    Returns:
        Tuple of (road_points, road_intensities) where z is within [z_min, z_max].
    """
    if points.shape[0] == 0:
        return points.copy(), intensities.copy()

    mask = (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    return points[mask], intensities[mask]


def extract_lane_markings(
    road_points: np.ndarray,
    road_intensities: np.ndarray,
    intensity_threshold: float,
) -> np.ndarray:
    """Filter road points by reflectance to isolate lane-marking candidates.

    Args:
        road_points: (N, 3) road-surface points.
        road_intensities: (N,) reflectance values aligned with ``road_points``.
        intensity_threshold: Keep points with reflectance >= this value.

    Returns:
        (M, 3) array of lane-marking candidate points.
    """
    if road_points.shape[0] == 0:
        return road_points.copy()

    mask = road_intensities >= intensity_threshold
    return road_points[mask]


def cluster_points(
    points: np.ndarray,
    eps: float,
    min_points: int,
) -> list[np.ndarray]:
    """Cluster 3D points with DBSCAN and return one array per cluster.

    Uses Open3D's built-in ``cluster_dbscan``. Noise points (label = -1)
    are discarded.

    Args:
        points: (N, 3) points to cluster.
        eps: Neighborhood radius (m).
        min_points: Minimum points for a core sample.

    Returns:
        List of (k_i, 3) arrays, one per cluster.
    """
    if points.shape[0] == 0:
        return []

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))

    labels = np.asarray(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    if labels.size == 0:
        return []

    clusters: list[np.ndarray] = []
    max_label = int(labels.max()) if labels.max() >= 0 else -1
    for label in range(max_label + 1):
        cluster = points[labels == label]
        if cluster.shape[0] > 0:
            clusters.append(cluster)
    return clusters


def save_features_geojson(
    clusters: list[np.ndarray],
    path: Path,
    feature_type: str = "lane_marking",
) -> None:
    """Write clustered feature points to a GeoJSON FeatureCollection.

    Coordinates are kept in the local Velodyne metric frame (not lat/lon),
    since KITTI Odometry has no GPS and real georeferencing happens in
    Stage 6 (Lanelet2 export).

    Args:
        clusters: List of (k_i, 3) point arrays, one per feature cluster.
        path: Destination .geojson path.
        feature_type: Value stored under ``properties.type`` for each feature.
    """
    features = []
    for idx, cluster in enumerate(clusters):
        coordinates = [[float(x), float(y), float(z)] for x, y, z in cluster]
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "MultiPoint",
                    "coordinates": coordinates,
                },
                "properties": {
                    "id": idx,
                    "type": feature_type,
                    "point_count": int(cluster.shape[0]),
                },
            }
        )

    feature_collection = {
        "type": "FeatureCollection",
        "crs": {
            "type": "name",
            "properties": {"name": "local:velodyne_meters"},
        },
        "features": features,
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(feature_collection, f, indent=2)
