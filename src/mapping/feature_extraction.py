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


def extract_curbs(
    points: np.ndarray,
    grid_size: float = 0.30,
    z_min: float = -2.0,
    z_max: float = -1.2,
    height_min: float = 0.10,
    height_max: float = 0.25,
    top_band: float = 0.03,
) -> np.ndarray:
    """Detect curb-top points via xy-grid height-step filtering.

    Returns points that sit on the *upper* edge of a short vertical step
    in the road-adjacent z band — i.e. curb tops, which form the
    continuous line along a road boundary that DBSCAN can then cluster.

    The project spec describes this feature as "法向量 + 曲率：路缘检测
    （曲率突变 = 路面边界）". On a voxelized ground map (working voxel
    0.15 m) the signal of interest reduces to a z discontinuity between
    adjacent xy cells, so this implementation uses a pure-numpy 2D grid
    instead of Open3D normal estimation. See ``refs/pipeline-notes.md``
    (Stage 5 MapBuilder memory discipline) — running ``estimate_normals``
    on a 30M-point working map is not memory-safe; this function only
    allocates ``O(N)`` int64 keys plus ``O(C)`` per-cell arrays
    (``C`` = unique xy cells ``<< N``).

    Args:
        points: ``(N, 3)`` XYZ in world (Velodyne) frame.
        grid_size: xy bin edge length (m). Default 0.30 is 2× the working
            voxel to avoid snap aliasing.
        z_min: Lower bound on the z pre-filter (m). Defaults to the same
            value as ``road_z_min`` so the curb search band aligns with
            the road-surface band.
        z_max: Upper bound on the z pre-filter (m). Extends above the
            road-surface band to include curb tops.
        height_min: Minimum per-cell ``zmax - zmin`` to count as a curb
            step. Default 0.10 stays clear of ~0.03 m residual z noise
            from 0.15 m voxel snap.
        height_max: Maximum per-cell ``zmax - zmin`` — larger jumps get
            rejected as walls / vehicles / vegetation.
        top_band: Keep points within this distance (m) of the cell ``zmax``.
            Default 0.03 m captures the top ~3 cm slab of the step.

    Returns:
        ``(M, 3)`` array of curb-top candidate points. ``M`` may be 0.
    """
    if points.shape[0] == 0:
        return points.copy()

    # Stage 1: prune to the z window — most of the working map is above
    # this band (buildings, foliage) and is irrelevant for curbs.
    z_mask = (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    if not np.any(z_mask):
        return np.zeros((0, 3), dtype=points.dtype)
    band = points[z_mask]

    # Stage 2: pack (x, y) cell index into a single int64 key using the
    # same bias+shift trick as ``_voxel_aggregate`` in map_builder.py.
    # ±2^20 cells per axis is far beyond any KITTI sequence footprint.
    bias = 1 << 20
    inv_grid = 1.0 / grid_size

    keys = np.floor(band[:, 0] * inv_grid).astype(np.int64)
    keys += bias
    keys <<= 21

    tmp = np.floor(band[:, 1] * inv_grid).astype(np.int64)
    tmp += bias
    keys |= tmp
    del tmp

    unique_keys, inverse = np.unique(keys, return_inverse=True)
    del keys
    n_cells = unique_keys.shape[0]
    del unique_keys

    # Stage 3: per-cell zmin / zmax via unbuffered ufunc.at. These are
    # ~10× slower than bincount but are the idiomatic numpy way to get
    # per-bin extrema; ~3-5 s on 30M points is acceptable.
    zmin = np.full(n_cells, np.inf, dtype=np.float64)
    zmax = np.full(n_cells, -np.inf, dtype=np.float64)
    np.minimum.at(zmin, inverse, band[:, 2])
    np.maximum.at(zmax, inverse, band[:, 2])

    cell_range = zmax - zmin
    cell_mask = (cell_range >= height_min) & (cell_range <= height_max)
    if not np.any(cell_mask):
        return np.zeros((0, 3), dtype=points.dtype)

    # Stage 4: keep only points within ``top_band`` of their cell's zmax,
    # restricted to cells that passed the height-step mask.
    point_cell_zmax = zmax[inverse]
    point_mask = cell_mask[inverse] & (band[:, 2] >= point_cell_zmax - top_band)
    return band[point_mask]


def _trim_cluster_minor_axis(cluster: np.ndarray, k: float = 2.5) -> np.ndarray:
    """Remove per-cluster outliers along the PCA minor axis using MAD.

    A side effect of the ``dbscan_eps=0.7`` setting used by
    :func:`cluster_points` is that virtual-lane fragments get bridged, which
    widens the minor-axis spread of otherwise-thin lane-marking clusters.
    Normal-consistent MAD (``1.4826 * median absolute deviation``) is robust
    to the heavy-tailed residuals those bridges produce.

    ``k=2.5`` keeps ~99% of a clean Gaussian line while still clipping the
    DBSCAN bridge residuals (typically > 1.2 sigma_hat off-axis). A single
    pass is sufficient; Stage 6 ``_pca_stats`` (``lanelet2_export.py``) uses
    2/98 percentile spans as a second safety net.

    Args:
        cluster: ``(N, 3)`` points in world-frame meters.
        k: MAD multiplier. Points with ``|resid| > k * sigma_hat`` on the
            minor axis are dropped.

    Returns:
        A possibly-smaller ``(M, 3)`` cluster (``M <= N``).
    """
    if cluster.shape[0] < 10:
        # PCA is unstable on tiny clusters; leave them untouched. Downstream
        # ``min_points`` filtering catches genuinely-small noise clusters.
        return cluster

    xy = cluster[:, :2]
    centered = xy - xy.mean(axis=0)
    cov = np.cov(centered.T)
    _, eigvecs = np.linalg.eigh(cov)  # ascending eigenvalues
    minor_axis = eigvecs[:, 0]
    resid = centered @ minor_axis
    med = np.median(resid)
    mad = np.median(np.abs(resid - med)) + 1e-9
    sigma_hat = 1.4826 * mad
    mask = np.abs(resid - med) <= k * sigma_hat
    return cluster[mask]


def cluster_points(
    points: np.ndarray,
    eps: float,
    min_points: int,
    trim_k: float | None = 2.5,
) -> list[np.ndarray]:
    """Cluster 3D points with DBSCAN and return one array per cluster.

    Uses Open3D's built-in ``cluster_dbscan``. Noise points (label = -1)
    are discarded. Each surviving cluster is optionally trimmed along its
    PCA minor axis with MAD-based outlier rejection — see
    :func:`_trim_cluster_minor_axis`. Clusters that fall below
    ``min_points`` after trimming are dropped entirely.

    Args:
        points: (N, 3) points to cluster.
        eps: Neighborhood radius (m).
        min_points: Minimum points for a core sample; also the minimum
            cluster size that survives the post-trim re-check.
        trim_k: MAD multiplier for the minor-axis outlier trim. ``None``
            or ``0`` disables the trim entirely (used by tests as a
            regression anchor).

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
        if cluster.shape[0] == 0:
            continue
        if trim_k is not None and trim_k > 0:
            cluster = _trim_cluster_minor_axis(cluster, k=trim_k)
            if cluster.shape[0] < min_points:
                continue
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
