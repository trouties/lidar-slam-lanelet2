"""Lanelet2 export of the curb clusters.

Each curb cluster is checked for being a line (PCA linearity, length,
thickness), reduced to an ordered polyline and written as an OSM way of
``type=curbstone``, the Lanelet2 line type of a curb. No ``subtype``: Lanelet2
splits curbstones into ``high`` and ``low`` (passable), and the extraction
does not tell the two apart. Only linestrings are written, no lanelet
``<relation>``: the map holds no lane boundaries to pair, so
``lanelet2.io.load`` returns the curbs in ``lineStringLayer`` and an empty
``laneletLayer``.
"""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.dom import minidom

import numpy as np

from src.export.georef import GeoReference, nominal_georeference


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


def _trim_curb_minor_axis(cluster: np.ndarray, stats: dict, k: float = 1.0) -> np.ndarray:
    """MAD-based minor-axis outlier trim for curb clusters.

    Drops points whose minor-axis projection lies further than ``k`` robust
    sigmas (1.4826 * MAD) from the minor-axis median. Pure-numpy and idempotent
    on already-clean clusters. Returns the original cluster if there are too
    few points or the trim would leave fewer than 3 survivors.
    """
    if cluster.shape[0] < 5:
        return cluster
    centered = cluster[:, :2] - stats["mean_xy"]
    pv = centered @ stats["v"]
    median_v = float(np.median(pv))
    mad = float(np.median(np.abs(pv - median_v)))
    if mad < 1e-9:
        return cluster
    sigma = 1.4826 * mad  # MAD -> robust stddev estimate
    mask = np.abs(pv - median_v) <= k * sigma
    if int(mask.sum()) < 3:
        return cluster
    return cluster[mask]


def _curb_classify_with_trim(
    cluster: np.ndarray,
    *,
    min_linearity: float,
    min_length: float,
    max_thickness: float,
    trim_k: float,
) -> tuple[str, dict, np.ndarray]:
    """Internal: classify curb cluster, optionally rescuing thick-tail clusters
    via a single minor-axis MAD trim pass.

    Returns ``(label, stats, effective_cluster)``:
        - ``label``: ``"curb"`` or ``"noise"``
        - ``stats``: PCA stats from the *effective* cluster (post-trim if
          rescued, otherwise from the original); includes ``median_z`` and,
          when applicable, ``trim_applied=True``
        - ``effective_cluster``: the points to use for downstream geometry
          (polyline). Equals the trimmed cluster on rescue, otherwise the
          original.

    Rescue policy: clustering already trims the minor axis once
    (``mapping.curb_trim_k``). Some clusters still have a minor-axis spread
    above ``max_thickness`` because DBSCAN pulled in points of an adjacent
    wall, plant or vehicle; a second, tighter trim removes them without
    relaxing the thresholds. Only clusters that fail the first pass *solely*
    on thickness are eligible: trimming the minor axis cannot make a short
    cluster long or a curved one straight.
    """
    if cluster.shape[0] < 3:
        return "noise", {"degenerate": True}, cluster

    stats = _pca_stats(cluster[:, :2])
    if stats.get("degenerate"):
        stats["median_z"] = float(np.median(cluster[:, 2]))
        return "noise", stats, cluster

    stats["median_z"] = float(np.median(cluster[:, 2]))

    # First pass.
    if stats["length"] < min_length or stats["linearity"] < min_linearity:
        # Length / linearity failures are not recoverable via minor-axis trim.
        return "noise", stats, cluster

    if stats["thickness"] <= max_thickness:
        return "curb", stats, cluster

    # Thickness-only failure: try one trim pass on the minor axis.
    trimmed = _trim_curb_minor_axis(cluster, stats, k=trim_k)
    if trimmed.shape[0] < 3 or trimmed.shape[0] == cluster.shape[0]:
        # Either too few survivors, or trim was a no-op (cluster already clean).
        return "noise", stats, cluster

    stats2 = _pca_stats(trimmed[:, :2])
    if stats2.get("degenerate"):
        return "noise", stats, cluster

    stats2["median_z"] = float(np.median(trimmed[:, 2]))

    if (
        stats2["length"] >= min_length
        and stats2["linearity"] >= min_linearity
        and stats2["thickness"] <= max_thickness
    ):
        stats2["trim_applied"] = True
        return "curb", stats2, trimmed

    return "noise", stats2, cluster


def classify_curb_cluster(
    cluster: np.ndarray,
    *,
    min_linearity: float = 0.75,
    min_length: float = 1.0,
    max_thickness: float = 0.7,
    trim_k: float = 1.0,
) -> tuple[str, dict]:
    """Classify a Stage 5 curb cluster.

    Returns ``"curb"`` or ``"noise"``.

    Internally, clusters that fail the first PCA pass *only* on thickness
    get one minor-axis MAD-trim retry (controlled by ``trim_k``); see
    ``_curb_classify_with_trim`` for the rescue policy.

    Args:
        cluster: ``(N, 3)`` cluster point array (Velodyne meters).
        min_linearity: PCA xy linearity floor.
        min_length: Reject clusters whose principal-axis span is below this.
        max_thickness: Reject clusters whose minor-axis span exceeds this
            (after the optional trim retry).
        trim_k: Sigma multiplier of the second-pass minor-axis MAD trim.

    Returns:
        Tuple ``(label, stats)`` where label is ``"curb"`` or ``"noise"``.
        ``stats`` carries ``trim_applied=True`` when the second pass was
        used to rescue the cluster.
    """
    label, stats, _ = _curb_classify_with_trim(
        cluster,
        min_linearity=min_linearity,
        min_length=min_length,
        max_thickness=max_thickness,
        trim_k=trim_k,
    )
    return label, stats


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


def _simplify_polyline_rdp(polyline: np.ndarray, epsilon: float) -> np.ndarray:
    """Ramer-Douglas-Peucker polyline simplification (pure numpy, iterative).

    Drops vertices whose perpendicular distance to the line connecting their
    neighbors is below ``epsilon`` meters. Operates in 3D using
    point-to-line distance via cross product. The first and last vertices
    are always preserved.

    Args:
        polyline: ``(P, 3)`` ordered vertex array. Returned unchanged when
            ``P <= 2`` or when ``epsilon <= 0``.
        epsilon: Tolerance in meters. Recommended ~0.05m (below Velodyne
            noise) so geometric meaning is preserved while flat segments
            collapse to their endpoints.

    Returns:
        ``(P', 3)`` simplified polyline with ``P' <= P``. ``P' >= 2`` always.
    """
    if epsilon <= 0.0 or polyline.shape[0] <= 2:
        return polyline

    n = polyline.shape[0]
    keep = np.zeros(n, dtype=bool)
    keep[0] = True
    keep[-1] = True

    # Iterative stack-based RDP to avoid Python recursion limits on long
    # polylines. Each stack entry is the (start, end) index pair of an
    # open segment that still needs to be evaluated.
    stack: list[tuple[int, int]] = [(0, n - 1)]
    while stack:
        start, end = stack.pop()
        if end - start < 2:
            continue
        a = polyline[start]
        b = polyline[end]
        ab = b - a
        ab_norm = float(np.linalg.norm(ab))
        if ab_norm < 1e-12:
            # Degenerate segment: distance is just |p - a|.
            seg = polyline[start + 1 : end]
            dists = np.linalg.norm(seg - a, axis=1)
        else:
            seg = polyline[start + 1 : end]
            cross = np.cross(seg - a, ab)
            # ``cross`` is (P, 3) for 3D inputs; its magnitude / |ab| gives
            # the perpendicular distance from each point to line a-b.
            cross_norm = np.linalg.norm(cross, axis=1)
            dists = cross_norm / ab_norm
        max_idx_local = int(np.argmax(dists))
        if dists[max_idx_local] > epsilon:
            split = start + 1 + max_idx_local
            keep[split] = True
            stack.append((start, split))
            stack.append((split, end))

    return polyline[keep]


def _add_node(
    parent: ET.Element,
    node_id: int,
    xyz: np.ndarray,
    lat: float,
    lon: float,
    ele: float,
) -> None:
    """Append a ``<node>`` with WGS84 position, altitude and map coordinates.

    ``local_x`` / ``local_y`` are the coordinates in the SLAM map frame, kept
    so the file can be related back to the point cloud.
    """
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
    ET.SubElement(node, "tag", {"k": "local_x", "v": f"{xyz[0]:.6f}"})
    ET.SubElement(node, "tag", {"k": "local_y", "v": f"{xyz[1]:.6f}"})
    ET.SubElement(node, "tag", {"k": "ele", "v": f"{ele:.6f}"})


def _build_osm_xml(features: list[dict], georef: GeoReference) -> ET.Element:
    """Build the OSM XML tree from classified features.

    ``features`` items::

        {
            "type": str,                # OSM type tag value
            "vertices": (P, 3) ndarray,
            "extra_tags": dict[str, str],   # optional, stable-ordered metadata
        }

    Features are sorted by centroid (x, y) before emission so node/way IDs
    are deterministic across runs (test stability). ``extra_tags`` keys are
    written in their dict insertion order so XML output is stable.
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
        type_tag = feat["type"]

        node_ids: list[int] = []
        lats, lons, eles = georef.to_latlon(verts)
        for xyz, lat, lon, ele in zip(verts, lats, lons, eles):
            lat, lon = float(lat), float(lon)
            _add_node(container, next_id, xyz, lat, lon, float(ele))
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
        ET.SubElement(way, "tag", {"k": "type", "v": type_tag})
        for k, v in feat.get("extra_tags", {}).items():
            ET.SubElement(way, "tag", {"k": k, "v": v})
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


_DEFAULT_CURB_CFG = {
    "min_linearity": 0.75,
    "min_length": 1.0,
    "max_thickness": 0.7,
    "trim_k": 1.0,
}


def _stats_tags(stats: dict, vertex_count: int, source_points: int) -> dict[str, str]:
    """Build a deterministic, downstream-readable tag dict from PCA stats.

    Lets consumers filter or inspect the ways without rerunning the export.
    Values have fixed precision, so the OSM file is byte-stable for identical
    inputs.
    """
    return {
        "length_m": f"{stats['length']:.2f}",
        "thickness_m": f"{stats['thickness']:.2f}",
        "linearity": f"{stats['linearity']:.3f}",
        "vertex_count": str(int(vertex_count)),
        "source_points": str(int(source_points)),
    }


def _classify_curb_features(
    curb_clusters: list[np.ndarray],
    *,
    cfg: dict,
    polyline_bin_size: float,
    polyline_simplify_epsilon: float,
) -> tuple[list[dict], dict]:
    """Curb channel: single-label classification, polyline geometry only.

    Uses ``_curb_classify_with_trim`` so that thick-tail clusters get one
    minor-axis MAD-trim retry. The polyline is built from the *effective*
    cluster (post-trim when rescued) so its vertices reflect the cleaned
    geometry, not the original outlier-contaminated points.
    """
    counts = {
        "kept": 0,
        "dropped": 0,
        "rescued": 0,
        "total_input": len(curb_clusters),
        "total_length_m": 0.0,
    }
    features: list[dict] = []

    for cluster in curb_clusters:
        if cluster.shape[0] < 3:
            counts["dropped"] += 1
            continue
        label, stats, effective = _curb_classify_with_trim(cluster, **cfg)
        if label == "noise":
            counts["dropped"] += 1
            continue
        polyline = cluster_to_polyline(effective, stats, bin_size=polyline_bin_size)
        if polyline is None:
            counts["dropped"] += 1
            continue
        polyline = _simplify_polyline_rdp(polyline, polyline_simplify_epsilon)
        rescued = bool(stats.get("trim_applied"))
        extra_tags = _stats_tags(stats, polyline.shape[0], effective.shape[0])
        extra_tags["rescued"] = "true" if rescued else "false"
        features.append(
            {
                "type": "curbstone",
                "vertices": polyline,
                "extra_tags": extra_tags,
            }
        )
        counts["kept"] += 1
        counts["total_length_m"] += float(stats["length"])
        if rescued:
            counts["rescued"] += 1

    counts["total_length_m"] = round(counts["total_length_m"], 2)
    return features, counts


def export_lanelet2_osm(
    curb_clusters: list[np.ndarray],
    path: Path,
    *,
    georef: GeoReference | None = None,
    lat0: float = 49.0,
    lon0: float = 8.4,
    polyline_bin_size: float = 0.5,
    polyline_simplify_epsilon: float = 0.05,
    curb: dict | None = None,
) -> dict:
    """Classify the curb clusters and write them to a Lanelet2 OSM file.

    Args:
        curb_clusters: Curb clusters, ``(k_i, 3)`` each, in the map frame.
        path: Output ``.osm`` file path. A ``<stem>.georef.json`` file next
            to it records the transformation and the origin to give
            Lanelet2's ``UtmProjector`` when loading.
        georef: Map-to-UTM transformation (see :mod:`src.export.georef`).
            Latitude and longitude are computed from UTM, so loading the
            file with ``UtmProjector`` reproduces the UTM geometry.
        lat0, lon0: Used only without ``georef``: the map origin is placed
            here with its axes on the UTM grid, and the map is not
            georeferenced.
        polyline_bin_size: Bin width (m) when reducing a linear cluster to
            an ordered polyline.
        polyline_simplify_epsilon: Ramer-Douglas-Peucker tolerance (m)
            applied after binning; 0 disables it.
        curb: Classifier overrides. Recognized keys: ``min_linearity``,
            ``min_length``, ``max_thickness``, ``trim_k``.

    Returns:
        ``{"curb": {"kept", "dropped", "rescued", "total_input",
        "total_length_m"}}`` with ``kept + dropped == total_input``;
        ``rescued`` counts the kept clusters that needed the second
        minor-axis trim.
    """
    curb_cfg = {**_DEFAULT_CURB_CFG, **(curb or {})}
    curb_features, curb_counts = _classify_curb_features(
        curb_clusters,
        cfg=curb_cfg,
        polyline_bin_size=polyline_bin_size,
        polyline_simplify_epsilon=polyline_simplify_epsilon,
    )

    if georef is None:
        georef = nominal_georeference(lat0, lon0)
    root = _build_osm_xml(curb_features, georef)
    raw = ET.tostring(root, encoding="utf-8")
    pretty = minidom.parseString(raw).toprettyxml(indent="  ", encoding="utf-8")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(pretty)

    origin_lat, origin_lon, _ = georef.to_latlon(np.zeros((1, 3)))
    sidecar = {
        **georef.summary(),
        "lanelet2_origin": {"lat": float(origin_lat[0]), "lon": float(origin_lon[0])},
    }
    with open(path.with_suffix(".georef.json"), "w") as f:
        json.dump(sidecar, f, indent=2)

    return {"curb": curb_counts}
