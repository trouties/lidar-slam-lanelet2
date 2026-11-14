"""Curb extraction from the accumulated point-cloud map."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def heights_in_nearest_pose_frame(points: np.ndarray, poses: np.ndarray) -> np.ndarray:
    """Return the z coordinate of each point in the frame of its nearest pose.

    The road and curb thresholds are heights in the sensor frame: the road is
    about 1.73 m below the KITTI Velodyne. The world frame is the first
    scan's sensor frame, and along a drive the road rises and falls by tens
    of metres in it, so a fixed world-z band only finds the road where the
    car happens to be at its starting height. Expressing every map point in
    the sensor frame of the trajectory pose closest to it (horizontally)
    measures its height relative to the road the car was driving on, and the
    pose's pitch and roll follow the local slope and camber of that road.

    Args:
        points: ``(N, 3)`` world-frame XYZ.
        poses: ``(M, 4, 4)`` sensor-to-world poses of the trajectory the map
            was built from.

    Returns:
        ``(N,)`` heights in metres.
    """
    poses = np.asarray(poses, dtype=np.float64)
    if points.shape[0] == 0:
        return np.zeros(0)
    t = poses[:, :3, 3]
    _, idx = cKDTree(t[:, :2]).query(points[:, :2], workers=-1)
    up = poses[idx, :3, 2]
    return np.einsum("ij,ij->i", points - t[idx], up)


def extract_curbs(
    points: np.ndarray,
    grid_size: float = 0.30,
    z_min: float = -2.0,
    z_max: float = -1.2,
    height_min: float = 0.10,
    height_max: float = 0.25,
    road_z_top: float = -1.55,
    top_band: float = 0.03,
    heights: np.ndarray | None = None,
) -> np.ndarray:
    """Detect curb-top points via xy-grid height-step filtering.

    Returns points that sit on the *upper* edge of a short vertical step
    that **straddles the road-surface upper boundary** — i.e. curb tops,
    which form the continuous line along a road boundary that DBSCAN can
    then cluster.

    On a voxelized map the curb reduces to a height discontinuity between
    neighbouring xy cells, so this uses a 2D grid instead of normal
    estimation: ``estimate_normals`` on a 30M-point map does not fit in
    memory, while this allocates ``O(N)`` int64 keys plus ``O(C)`` per-cell
    arrays (``C`` = occupied xy cells ``<< N``).

    The ``road_z_top`` parameter is what makes the result semantically a
    *curb* and not just any short height step. A cell qualifies only if
    its lowest point sits at or below ``road_z_top`` (the cell contains road
    points) AND its highest point is strictly above it (the cell also
    contains points just above the road). Without this constraint the
    detector lights up on every wall / parked car / road dip whose
    *internal* range happens to fall in ``[height_min, height_max]``.

    All height tests use ``heights`` when given (see
    :func:`heights_in_nearest_pose_frame`), otherwise ``points[:, 2]``.

    Args:
        points: ``(N, 3)`` XYZ in world (Velodyne) frame.
        grid_size: xy bin edge length (m). Default 0.30 is 2× the working
            voxel to avoid snap aliasing.
        z_min: Lower bound on the height pre-filter (m), about 0.27 m
            below the road under the car (the sensor is 1.73 m above it).
        z_max: Upper bound on the z pre-filter (m). Extends above the
            road-surface band to include curb tops.
        height_min: Minimum per-cell ``zmax - zmin`` to count as a curb
            step. Default 0.10 stays clear of ~0.03 m residual z noise
            from 0.15 m voxel snap.
        height_max: Maximum per-cell ``zmax - zmin`` — larger jumps get
            rejected as walls / vehicles / vegetation.
        road_z_top: Height (m) a cell must straddle to count as a curb cell,
            see above; -1.55 is 0.18 m above the road under the car. The
            road falls away towards its edges, so a fixed value misses many
            low curbs there (docs/map-quality.md).
        top_band: Keep points within this distance (m) of the cell ``zmax``.
            Default 0.03 m captures the top ~3 cm slab of the step.
        heights: ``(N,)`` height of each point used for every height test.
            Defaults to ``points[:, 2]``.

    Returns:
        ``(M, 3)`` array of curb-top candidate points in world coordinates.
        ``M`` may be 0.
    """
    if points.shape[0] == 0:
        return points.copy()

    h_all = points[:, 2] if heights is None else heights

    # Step 1: prune to the height window — most of the working map is above
    # this band (buildings, foliage) and is irrelevant for curbs.
    z_mask = (h_all >= z_min) & (h_all <= z_max)
    if not np.any(z_mask):
        return np.zeros((0, 3), dtype=points.dtype)
    band = points[z_mask]
    band_h = h_all[z_mask]

    # Step 2: pack (x, y) cell index into a single int64 key using the
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

    # Step 3: per-cell zmin / zmax via unbuffered ufunc.at. These are
    # ~10× slower than bincount but are the idiomatic numpy way to get
    # per-bin extrema; ~3-5 s on 30M points is acceptable.
    zmin = np.full(n_cells, np.inf, dtype=np.float64)
    zmax = np.full(n_cells, -np.inf, dtype=np.float64)
    np.minimum.at(zmin, inverse, band_h)
    np.maximum.at(zmax, inverse, band_h)

    cell_range = zmax - zmin
    cell_mask = (
        (cell_range >= height_min)
        & (cell_range <= height_max)
        & (zmin <= road_z_top)
        & (zmax > road_z_top)
    )
    if not np.any(cell_mask):
        return np.zeros((0, 3), dtype=points.dtype)

    # Step 4: keep only points within ``top_band`` of their cell's zmax,
    # restricted to cells that passed the height-step mask.
    point_cell_zmax = zmax[inverse]
    point_mask = cell_mask[inverse] & (band_h >= point_cell_zmax - top_band)
    return band[point_mask]


def _trim_cluster_minor_axis(cluster: np.ndarray, k: float = 2.5) -> np.ndarray:
    """Remove per-cluster outliers along the PCA minor axis using MAD.

    DBSCAN grows a cluster through any chain of neighbours, so a line of
    curb points picks up side branches into adjacent walls or vegetation,
    which widen it across its axis. The normal-consistent MAD
    (``1.4826 * median absolute deviation``) is robust to these heavy-tailed
    residuals. ``k=2.5`` keeps about 99 % of a clean Gaussian line.

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
    feature_type: str = "curb",
) -> None:
    """Write clustered feature points to a GeoJSON FeatureCollection.

    Coordinates are in the map frame (the Velodyne frame of the first scan),
    not latitude and longitude; georeferencing happens in the Lanelet2
    export.

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
