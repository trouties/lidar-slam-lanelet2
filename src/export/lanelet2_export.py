"""Lanelet2 HD Map export.

Stage 6: Convert Stage 5 lane-marking clusters into a Lanelet2-compatible
``.osm`` XML file.

Pipeline: cluster -> PCA morphology classify -> polyline / oriented-bbox
geometry -> OSM XML (nodes + ways).

Scope note: This stage emits **LineStrings and Areas only** -- no Lanelet
``<relation>`` elements. Stage 5 provides no lane topology / no left-right
pairing, so honest lanelet construction is not possible without inventing
heuristics. Downstream loaders (``lanelet2.io.load``) will see entries in
``map.lineStringLayer``, not ``map.laneletLayer``.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.dom import minidom

import numpy as np

EARTH_RADIUS_M = 6_371_000.0


def _pca_stats(xy: np.ndarray) -> dict:
    """Compute 2D PCA statistics for cluster classification.

    Uses 2nd/98th percentile span (not raw min/max) for length and thickness
    so single outlier points don't inflate the bounding extent.

    Returns a dict with keys: ``linearity``, ``length``, ``thickness``,
    ``u``, ``v`` (principal/minor unit vectors), ``mean_xy``, ``median_z``,
    ``degenerate``.
    """
    if xy.shape[0] < 3:
        return {"degenerate": True}

    mean_xy = xy.mean(axis=0)
    centered = xy - mean_xy
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # eigh returns ascending order; flip to descending so eigvals[0] = lambda1
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    lam1 = float(eigvals[0])
    lam2 = float(eigvals[1])
    if lam1 < 1e-9:
        return {"degenerate": True}

    linearity = (lam1 - lam2) / lam1
    u = eigvecs[:, 0]
    v = eigvecs[:, 1]

    proj_u = centered @ u
    proj_v = centered @ v
    length = float(np.percentile(proj_u, 98) - np.percentile(proj_u, 2))
    thickness = float(np.percentile(proj_v, 98) - np.percentile(proj_v, 2))

    return {
        "linearity": linearity,
        "length": length,
        "thickness": thickness,
        "u": u,
        "v": v,
        "mean_xy": mean_xy,
        "median_z": float(np.median(xy[:, -1])) if xy.shape[1] >= 3 else 0.0,
        "degenerate": False,
    }


def classify_cluster(
    cluster: np.ndarray,
    *,
    min_linearity: float = 0.75,
    min_length: float = 1.0,
    line_thin_max_thickness: float = 0.8,
    line_thick_max_thickness: float = 2.0,
) -> tuple[str, dict]:
    """Classify a Stage 5 cluster by xy morphology.

    Args:
        cluster: ``(N, 3)`` cluster point array (Velodyne meters).
        min_linearity: PCA linearity required for line classes.
        min_length: Reject clusters whose principal-axis span is below this.
        line_thin_max_thickness: Upper bound on minor-axis span for ``line_thin``.
        line_thick_max_thickness: Upper bound on minor-axis span for ``line_thick``.

    Returns:
        Tuple of (label, stats) where label is one of
        ``"line_thin"``, ``"line_thick"``, ``"area"``, ``"noise"``.
    """
    if cluster.shape[0] < 3:
        return "noise", {"degenerate": True}

    # Build (N, 3) view: cluster carries xyz, _pca_stats only needs xy + z.
    xy_with_z = cluster[:, :3]
    stats = _pca_stats(xy_with_z[:, :2])
    if stats.get("degenerate"):
        # Still try to surface a usable median_z so callers don't crash.
        stats["median_z"] = float(np.median(cluster[:, 2]))
        return "noise", stats

    # Re-attach median_z computed from full cluster (not just xy slice).
    stats["median_z"] = float(np.median(cluster[:, 2]))

    if stats["length"] < min_length:
        return "noise", stats

    if stats["linearity"] >= min_linearity:
        if stats["thickness"] <= line_thin_max_thickness:
            return "line_thin", stats
        if stats["thickness"] <= line_thick_max_thickness:
            return "line_thick", stats

    if stats["thickness"] > line_thick_max_thickness:
        return "area", stats

    return "noise", stats


def cluster_to_polyline(
    cluster: np.ndarray,
    stats: dict,
    bin_size: float = 0.5,
) -> np.ndarray | None:
    """Reduce a linear cluster to an ordered polyline.

    Bins points along the principal axis and takes the median (x, y, z) of
    each non-empty bin. Returns ``None`` if fewer than 2 bins survive.
    """
    u = stats["u"]
    mean_xy = stats["mean_xy"]
    centered_xy = cluster[:, :2] - mean_xy
    proj = centered_xy @ u

    proj_min = float(proj.min())
    proj_max = float(proj.max())
    span = proj_max - proj_min
    if span < 1e-6:
        return None

    n_bins = max(1, int(math.ceil(span / bin_size)))
    # np.digitize: returns 1..n_bins for in-range values
    edges = np.linspace(proj_min, proj_max, n_bins + 1)
    bin_idx = np.clip(np.digitize(proj, edges) - 1, 0, n_bins - 1)

    vertices: list[tuple[float, float, float, float]] = []
    for b in range(n_bins):
        mask = bin_idx == b
        if not np.any(mask):
            continue
        pts = cluster[mask]
        median_xyz = np.median(pts[:, :3], axis=0)
        center_proj = float(np.median(proj[mask]))
        vertices.append(
            (center_proj, float(median_xyz[0]), float(median_xyz[1]), float(median_xyz[2]))
        )

    if len(vertices) < 2:
        return None

    vertices.sort(key=lambda v: v[0])
    return np.array([[v[1], v[2], v[3]] for v in vertices], dtype=np.float64)


def cluster_to_polygon(cluster: np.ndarray, stats: dict) -> np.ndarray:
    """Reduce a blob cluster to a 4-corner PCA-oriented bounding rectangle.

    Pure numpy (no scipy). Returns ``(4, 3)`` corners in counter-clockwise
    order in the local xy frame, all sharing the cluster's median z.
    """
    u = stats["u"]
    v = stats["v"]
    mean_xy = stats["mean_xy"]
    centered_xy = cluster[:, :2] - mean_xy
    pu = centered_xy @ u
    pv = centered_xy @ v

    u_min, u_max = float(pu.min()), float(pu.max())
    v_min, v_max = float(pv.min()), float(pv.max())

    # Counter-clockwise corners in (u, v).
    uv_corners = [
        (u_min, v_min),
        (u_max, v_min),
        (u_max, v_max),
        (u_min, v_max),
    ]
    z = stats["median_z"]
    corners = []
    for cu, cv in uv_corners:
        xy = mean_xy + cu * u + cv * v
        corners.append([float(xy[0]), float(xy[1]), z])
    return np.array(corners, dtype=np.float64)


def _local_to_latlon(x: float, y: float, lat0: float, lon0: float) -> tuple[float, float]:
    """Equirectangular projection from local meters to WGS84 lat/lon.

    Sub-meter error within ~5 km of the origin -- adequate for KITTI-scale
    sequences. KITTI Odometry has no real GPS, so ``lat0/lon0`` is a fake
    Karlsruhe-area origin chosen so the file can still be loaded by
    Lanelet2's WGS84-based loaders.
    """
    deg_per_m_lat = 180.0 / (math.pi * EARTH_RADIUS_M)
    deg_per_m_lon = 180.0 / (math.pi * EARTH_RADIUS_M * math.cos(math.radians(lat0)))
    lat = lat0 + y * deg_per_m_lat
    lon = lon0 + x * deg_per_m_lon
    return lat, lon


def _add_node(
    parent: ET.Element,
    node_id: int,
    x: float,
    y: float,
    z: float,
    lat0: float,
    lon0: float,
) -> tuple[float, float]:
    """Append a ``<node>`` element with WGS84 + local-frame audit tags.

    Returns the (lat, lon) actually emitted so callers can update bounds.
    """
    lat, lon = _local_to_latlon(x, y, lat0, lon0)
    node = ET.SubElement(
        parent,
        "node",
        {
            "id": str(node_id),
            "lat": f"{lat:.10f}",
            "lon": f"{lon:.10f}",
            "visible": "true",
            "version": "1",
        },
    )
    ET.SubElement(node, "tag", {"k": "local_x", "v": f"{x:.6f}"})
    ET.SubElement(node, "tag", {"k": "local_y", "v": f"{y:.6f}"})
    ET.SubElement(node, "tag", {"k": "ele", "v": f"{z:.6f}"})
    return lat, lon


def _build_osm_xml(features: list[dict], lat0: float, lon0: float) -> ET.Element:
    """Build the OSM XML tree from classified features.

    ``features`` items: ``{"kind": "polyline"|"area", "type": str, "vertices": (P, 3)}``.

    Features are sorted by centroid (x, y) before emission so node/way IDs
    are deterministic across runs (test stability).
    """
    root = ET.Element("osm", {"version": "0.6", "generator": "lidar-slam-hdmap"})

    # Sort for determinism.
    def _centroid_key(f: dict) -> tuple[float, float]:
        c = f["vertices"].mean(axis=0)
        return (round(float(c[0]), 6), round(float(c[1]), 6))

    sorted_features = sorted(features, key=_centroid_key)

    next_id = 1
    lat_min = lat_max = lon_min = lon_max = None

    # We append nodes and ways into a temporary container, then prepend bounds
    # at the end so it's always the first child of <osm>.
    body_nodes: list[ET.Element] = []
    body_ways: list[ET.Element] = []
    container = ET.Element("_container")  # discarded; used so _add_node has a parent

    for feat in sorted_features:
        verts = feat["vertices"]
        kind = feat["kind"]
        type_tag = feat["type"]

        node_ids: list[int] = []
        for vx, vy, vz in verts:
            lat, lon = _add_node(container, next_id, float(vx), float(vy), float(vz), lat0, lon0)
            node_ids.append(next_id)
            next_id += 1
            lat_min = lat if lat_min is None else min(lat_min, lat)
            lat_max = lat if lat_max is None else max(lat_max, lat)
            lon_min = lon if lon_min is None else min(lon_min, lon)
            lon_max = lon if lon_max is None else max(lon_max, lon)

        way = ET.Element("way", {"id": str(next_id), "visible": "true", "version": "1"})
        next_id += 1
        for nid in node_ids:
            ET.SubElement(way, "nd", {"ref": str(nid)})
        if kind == "area":
            # Lanelet2 area: explicitly close by repeating the first node ref.
            ET.SubElement(way, "nd", {"ref": str(node_ids[0])})
            ET.SubElement(way, "tag", {"k": "area", "v": "yes"})
            ET.SubElement(way, "tag", {"k": "type", "v": type_tag})
        else:
            ET.SubElement(way, "tag", {"k": "type", "v": type_tag})
            ET.SubElement(way, "tag", {"k": "subtype", "v": "solid"})
        body_ways.append(way)

    # Move nodes from container into ordered list (preserving creation order).
    body_nodes.extend(list(container))

    if lat_min is not None:
        ET.SubElement(
            root,
            "bounds",
            {
                "minlat": f"{lat_min:.10f}",
                "minlon": f"{lon_min:.10f}",
                "maxlat": f"{lat_max:.10f}",
                "maxlon": f"{lon_max:.10f}",
            },
        )
    for n in body_nodes:
        root.append(n)
    for w in body_ways:
        root.append(w)

    return root


def export_lanelet2_osm(
    clusters: list[np.ndarray],
    path: Path,
    lat0: float = 49.0,
    lon0: float = 8.4,
    min_linearity: float = 0.75,
    min_length: float = 1.0,
    line_thin_max_thickness: float = 0.8,
    line_thick_max_thickness: float = 2.0,
    polyline_bin_size: float = 0.5,
) -> dict:
    """Classify Stage 5 clusters and write a Lanelet2-compatible OSM file.

    Args:
        clusters: List of ``(k_i, 3)`` cluster point arrays in Velodyne meters.
        path: Output ``.osm`` file path.
        lat0, lon0: Fake WGS84 origin for the equirectangular projection.
            KITTI Odometry has no GPS; defaults pin to the Karlsruhe area.
        min_linearity: PCA xy linearity threshold for ``line_*`` classes.
        min_length: Minimum principal-axis span (m) for any non-noise label.
        line_thin_max_thickness: Minor-axis cap (m) for ``line_thin``.
        line_thick_max_thickness: Minor-axis cap (m) for ``line_thick``;
            wider clusters become ``area``.
        polyline_bin_size: Bin width (m) when reducing a line cluster
            to an ordered polyline.

    Returns:
        Counts dict with keys ``line_thin``, ``line_thick``, ``area``,
        ``dropped``, ``total_input``. Conservation invariant:
        ``line_thin + line_thick + area + dropped == total_input``.

    Note:
        Emits LineStrings and Areas only -- no Lanelet ``<relation>`` is
        produced because Stage 5 provides no left/right pairing or topology.
    """
    counts = {
        "line_thin": 0,
        "line_thick": 0,
        "area": 0,
        "dropped": 0,
        "total_input": len(clusters),
    }
    features: list[dict] = []

    for cluster in clusters:
        if cluster.shape[0] < 3:
            counts["dropped"] += 1
            continue
        label, stats = classify_cluster(
            cluster,
            min_linearity=min_linearity,
            min_length=min_length,
            line_thin_max_thickness=line_thin_max_thickness,
            line_thick_max_thickness=line_thick_max_thickness,
        )
        if label == "noise":
            counts["dropped"] += 1
            continue

        if label in ("line_thin", "line_thick"):
            polyline = cluster_to_polyline(cluster, stats, bin_size=polyline_bin_size)
            if polyline is None:
                counts["dropped"] += 1
                continue
            features.append({"kind": "polyline", "type": label, "vertices": polyline})
            counts[label] += 1
        elif label == "area":
            polygon = cluster_to_polygon(cluster, stats)
            features.append({"kind": "area", "type": "zebra_marking", "vertices": polygon})
            counts["area"] += 1

    root = _build_osm_xml(features, lat0=lat0, lon0=lon0)
    raw = ET.tostring(root, encoding="utf-8")
    pretty = minidom.parseString(raw).toprettyxml(indent="  ", encoding="utf-8")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(pretty)

    return counts
