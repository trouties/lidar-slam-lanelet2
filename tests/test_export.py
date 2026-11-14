"""Tests for Stage 6: Lanelet2 OSM export."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from collections import Counter

import numpy as np
import pytest

from src.export.georef import GeoReference, nominal_georeference
from src.export.lanelet2_export import (
    _pca_stats,
    _simplify_polyline_rdp,
    classify_curb_cluster,
    cluster_to_polyline,
    export_lanelet2_osm,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _thin_line(
    n: int = 50,
    length: float = 5.0,
    sigma_y: float = 0.05,
    seed: int = 0,
) -> np.ndarray:
    """Straight line along the x-axis with small Gaussian noise on y."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, length, n)
    y = rng.normal(0.0, sigma_y, n)
    z = np.full(n, -1.73)
    return np.column_stack([x, y, z])


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def test_cluster_to_polyline_orders_along_axis():
    cluster = _thin_line(n=80, length=8.0)
    rng = np.random.default_rng(7)
    rng.shuffle(cluster)  # in-place row shuffle
    stats = _pca_stats(cluster[:, :2])
    assert not stats["degenerate"]

    polyline = cluster_to_polyline(cluster, stats, bin_size=0.5)
    assert polyline is not None
    assert polyline.shape[1] == 3
    assert polyline.shape[0] >= 2

    # Project polyline onto principal axis; expect monotonic order.
    centered = polyline[:, :2] - stats["mean_xy"]
    proj = centered @ stats["u"]
    diffs = np.diff(proj)
    assert np.all(diffs > 0) or np.all(diffs < 0)


def test_simplify_polyline_collapses_straight_line():
    # 11 collinear points along x; epsilon 0.05m should leave only the
    # endpoints.
    poly = np.column_stack([np.linspace(0.0, 10.0, 11), np.zeros(11), np.full(11, -1.73)])
    out = _simplify_polyline_rdp(poly, epsilon=0.05)
    assert out.shape == (2, 3)
    assert np.allclose(out[0], poly[0])
    assert np.allclose(out[-1], poly[-1])


def test_simplify_polyline_preserves_kink():
    # L-shape: 5 points along +x, then 5 along +y. Vertex at the corner
    # has perpendicular distance ~3.5m to the chord between endpoints,
    # well above epsilon.
    poly = np.array(
        [
            [0.0, 0.0, -1.73],
            [1.0, 0.0, -1.73],
            [2.0, 0.0, -1.73],
            [3.0, 0.0, -1.73],
            [4.0, 0.0, -1.73],
            [4.0, 1.0, -1.73],
            [4.0, 2.0, -1.73],
            [4.0, 3.0, -1.73],
            [4.0, 4.0, -1.73],
        ]
    )
    out = _simplify_polyline_rdp(poly, epsilon=0.05)
    # Endpoints + corner = 3 vertices
    assert out.shape == (3, 3)
    # Corner must be (4, 0, -1.73)
    assert np.allclose(out[1], [4.0, 0.0, -1.73])


def test_simplify_polyline_disabled_when_epsilon_zero():
    poly = np.column_stack([np.linspace(0.0, 5.0, 8), np.zeros(8), np.full(8, -1.73)])
    out = _simplify_polyline_rdp(poly, epsilon=0.0)
    assert out.shape == poly.shape


def test_simplify_polyline_short_input_passthrough():
    poly = np.array([[0.0, 0.0, -1.73], [1.0, 0.0, -1.73]])
    out = _simplify_polyline_rdp(poly, epsilon=0.05)
    assert out.shape == (2, 3)


# ---------------------------------------------------------------------------
# Curb classification
# ---------------------------------------------------------------------------


def _curb_line(
    n: int = 80,
    length: float = 4.0,
    sigma_y: float = 0.10,
    seed: int = 20,
) -> np.ndarray:
    """Synthetic curbstone-shape cluster: long, thin, near-linear.

    Defaults are tuned to match Stage 5 v4 'good' rate definition:
    length >= 1.5 m, thickness <= 0.5 m, linearity >= 0.80. The z value
    sits ~0.27m above the ground (matches v4 measured -1.46 mean).
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, length, n)
    y = rng.normal(0.0, sigma_y, n)
    z = np.full(n, -1.46)
    return np.column_stack([x, y, z])


def test_classify_curb_cluster_kept():
    label, stats = classify_curb_cluster(_curb_line())
    assert label == "curb"
    assert stats["linearity"] >= 0.80
    assert stats["length"] >= 1.5
    assert stats["thickness"] <= 0.6


def test_classify_curb_too_thick_dropped():
    # 4m × 1m blob -- linear enough but exceeds curb max_thickness (0.6m).
    label, _ = classify_curb_cluster(_curb_line(sigma_y=0.4, seed=21))
    assert label == "noise"


def test_classify_curb_too_short_dropped():
    # 0.8m strip -- fails min_length (1.5m).
    label, _ = classify_curb_cluster(_curb_line(length=0.8, seed=22))
    assert label == "noise"


def _curb_line_with_outliers(
    n_clean: int = 60,
    n_outlier: int = 12,
    length: float = 4.0,
    sigma_y_clean: float = 0.10,
    outlier_offset: float = 1.5,
    seed: int = 60,
) -> np.ndarray:
    """Long thin curb body plus a wad of points pulled away on the minor axis.

    Mimics the seq 06/01/10 failure mode: a real curbstone line with adjacent
    wall / vegetation points that DBSCAN swept into the same cluster. The
    body is curb-shaped (length ~4m, sigma_y 0.10m) but the outliers push
    the raw thickness above 0.7m. A second-pass minor-axis trim recovers the
    body.
    """
    rng = np.random.default_rng(seed)
    x_body = np.linspace(0.0, length, n_clean)
    y_body = rng.normal(0.0, sigma_y_clean, n_clean)
    z_body = np.full(n_clean, -1.46)
    body = np.column_stack([x_body, y_body, z_body])

    x_out = rng.uniform(0.0, length, n_outlier)
    y_out = rng.normal(outlier_offset, 0.05, n_outlier)
    z_out = np.full(n_outlier, -1.46)
    outliers = np.column_stack([x_out, y_out, z_out])

    return np.vstack([body, outliers])


def test_classify_curb_thick_but_trimmable_is_rescued():
    cluster = _curb_line_with_outliers()
    label, stats = classify_curb_cluster(cluster)
    assert label == "curb"
    assert stats.get("trim_applied") is True
    # Post-trim thickness should be well below the gate.
    assert stats["thickness"] <= 0.7
    # Length should be roughly preserved (trim is on minor axis).
    assert stats["length"] >= 3.5


def test_classify_curb_short_cluster_not_rescued_by_trim():
    # 0.5m clean line: fails min_length (1.0m). Trim only fixes thickness,
    # not length, so the rescue branch must not be taken.
    short = _curb_line(n=20, length=0.5, seed=23)
    label, stats = classify_curb_cluster(short)
    assert label == "noise"
    assert stats.get("trim_applied") is not True


def test_export_writes_valid_osm_with_curb_ways(tmp_path):
    curb_clusters = [
        _curb_line(n=80, length=4.0, seed=40),
        _curb_line(n=80, length=3.0, seed=41),
        _curb_line(length=0.8, seed=42),  # too short -> noise
    ]
    out = tmp_path / "map.osm"
    curb = export_lanelet2_osm(curb_clusters, out)["curb"]

    # Conservation; clean synthetic curbs do not need the trim retry.
    assert curb["total_input"] == 3
    assert curb["kept"] + curb["dropped"] == 3
    assert curb["kept"] == 2
    assert curb["dropped"] == 1
    assert curb["rescued"] == 0

    root = ET.parse(out).getroot()
    assert root.tag == "osm"
    assert root.attrib.get("version") == "0.6"
    assert len(root.findall("bounds")) == 1
    ways = root.findall("way")
    assert len(ways) == 2
    for w in ways:
        tags = {t.get("k"): t.get("v") for t in w.findall("tag")}
        assert tags["type"] == "curbstone"
        assert "subtype" not in tags
        assert "area" not in tags
        for key in ("length_m", "thickness_m", "linearity", "vertex_count", "source_points"):
            assert key in tags
        assert int(tags["vertex_count"]) >= 2


def test_export_with_no_clusters_writes_empty_map(tmp_path):
    out = tmp_path / "empty.osm"
    curb = export_lanelet2_osm([], out)["curb"]
    assert curb == {"kept": 0, "dropped": 0, "rescued": 0, "total_input": 0, "total_length_m": 0.0}
    assert ET.parse(out).getroot().findall("way") == []


def test_export_rescued_counter_tracks_trim_recoveries(tmp_path):
    # Mix one clean curb (no trim needed) with one thick-but-trimmable curb
    # (trim required). Expect kept=2, rescued=1, dropped=0.
    curb_clusters = [
        _curb_line(n=80, length=4.0, seed=70),
        _curb_line_with_outliers(seed=71),
    ]
    out = tmp_path / "map_rescue.osm"
    counts = export_lanelet2_osm(curb_clusters, out)

    curb = counts["curb"]
    assert curb["total_input"] == 2
    assert curb["kept"] == 2
    assert curb["dropped"] == 0
    assert curb["rescued"] == 1

    # Both should appear as type=curbstone ways in the OSM file.
    tree = ET.parse(out)
    root = tree.getroot()
    way_types = Counter(
        tag.get("v")
        for w in root.findall("way")
        for tag in w.findall("tag")
        if tag.get("k") == "type"
    )
    assert way_types["curbstone"] == 2


def test_export_curb_way_rescued_tag_reflects_trim(tmp_path):
    curb_clusters = [
        _curb_line(n=80, length=4.0, seed=90),  # clean -> rescued=false
        _curb_line_with_outliers(seed=91),  # thick-tail -> rescued=true
    ]
    out = tmp_path / "map_curb_meta.osm"
    export_lanelet2_osm(curb_clusters, out)

    tree = ET.parse(out)
    curb_ways = [
        w
        for w in tree.getroot().findall("way")
        if any(t.get("k") == "type" and t.get("v") == "curbstone" for t in w.findall("tag"))
    ]
    assert len(curb_ways) == 2

    rescued_flags = []
    for way in curb_ways:
        tag_dict: dict[str, str] = {}
        for t in way.findall("tag"):
            k, v = t.get("k"), t.get("v")
            if k is not None and v is not None:
                tag_dict[k] = v
        assert tag_dict.get("rescued") in {"true", "false"}
        rescued_flags.append(tag_dict["rescued"])
        # Geometry metadata also present on curbs.
        assert "length_m" in tag_dict
        assert float(tag_dict["thickness_m"]) <= 0.7  # Post-trim thickness gate.

    assert sorted(rescued_flags) == ["false", "true"]


def test_export_total_length_metric_accumulates_kept_features(tmp_path):
    # Two clean ~4m curbs and one too-short curb. Expect total_length_m to
    # roughly equal 2 * 4 = 8 m (within polyline reduction tolerance).
    curb_clusters = [
        _curb_line(n=80, length=4.0, seed=100),
        _curb_line(n=80, length=4.0, seed=101),
        _curb_line(length=0.5, seed=102),  # too short -> dropped
    ]
    out = tmp_path / "map_length.osm"
    counts = export_lanelet2_osm(curb_clusters, out)

    curb = counts["curb"]
    assert curb["kept"] == 2
    assert curb["dropped"] == 1
    # Length is the PCA principal-axis 2-98 percentile span; expect
    # ~7.5-8.0m total for two ~4m segments.
    assert 7.0 <= curb["total_length_m"] <= 8.5


def test_export_total_length_excludes_dropped_clusters(tmp_path):
    # Single short cluster -> all dropped, length must stay zero.
    out = tmp_path / "map_length_zero.osm"
    counts = export_lanelet2_osm([_curb_line(length=0.5, seed=110)], out)
    assert counts["curb"]["kept"] == 0
    assert counts["curb"]["dropped"] == 1
    assert counts["curb"]["total_length_m"] == 0.0


def test_export_deterministic_ids(tmp_path):
    curb_clusters = [_curb_line(n=80, length=4.0, seed=51), _curb_line(length=3.0, seed=52)]
    out_a = tmp_path / "a.osm"
    out_b = tmp_path / "b.osm"

    export_lanelet2_osm(curb_clusters, out_a)
    export_lanelet2_osm(curb_clusters, out_b)

    # Same input should produce byte-identical OSM (deterministic IDs).
    assert out_a.read_bytes() == out_b.read_bytes()


# ---------------------------------------------------------------------------
# Georeferencing and loading with Lanelet2
# ---------------------------------------------------------------------------


def _georef_rotated() -> GeoReference:
    """Map frame rotated by 30 deg and tilted slightly against UTM."""
    yaw, pitch = np.radians(30.0), 0.01
    rz = np.array(
        [[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]]
    )
    ry = np.array(
        [[np.cos(pitch), 0.0, np.sin(pitch)], [0.0, 1.0, 0.0], [-np.sin(pitch), 0.0, np.cos(pitch)]]
    )
    T = np.eye(4)
    T[:3, :3] = rz @ ry
    T[:3, 3] = [455395.5, 5425695.5, 117.1]
    return GeoReference(map_to_utm=T, epsg=32632, source="gnss")


def test_nominal_georeference_places_origin_on_the_utm_grid():
    g = nominal_georeference(49.0, 8.4)
    assert g.epsg == 32632
    lat, lon, ele = g.to_latlon(np.array([[0.0, 0.0, -1.73]]))
    assert abs(lat[0] - 49.0) < 1e-9 and abs(lon[0] - 8.4) < 1e-9
    assert ele[0] == -1.73


@pytest.mark.parametrize("georef", [None, _georef_rotated()], ids=["nominal", "gnss"])
def test_lanelet2_loads_export_with_intended_geometry(tmp_path, georef):
    """Lanelet2 reads every way, and its UTM projector reproduces the map-to-UTM geometry."""
    lanelet2 = pytest.importorskip("lanelet2")
    from lanelet2.io import Origin
    from lanelet2.projection import UtmProjector

    curb_clusters = [_curb_line(length=4.0, seed=12), _curb_line(length=3.0, seed=13)]
    out = tmp_path / "map.osm"
    export_lanelet2_osm(curb_clusters, out, georef=georef)

    side = json.loads(out.with_suffix(".georef.json").read_text())
    origin = side["lanelet2_origin"]
    projector = UtmProjector(Origin(origin["lat"], origin["lon"]))
    lmap, errors = lanelet2.io.loadRobust(str(out), projector)
    assert errors == []

    n_ways = len(ET.parse(out).getroot().findall("way"))
    assert len(lmap.lineStringLayer) == n_ways == 2
    assert len(lmap.laneletLayer) == 0

    # Lanelet2 local = UTM - UTM(origin); the exporter's intended UTM is map_to_utm @ local_xyz.
    T = np.asarray(side["map_to_utm"])
    origin_utm = T[:3, 3]
    # Lanelet2 turns the ``ele`` tag into the point's z. Recover the map-frame z from it
    # through the third row of map_to_utm, then check the projected x, y.
    r = T[2, :3]
    for p in lmap.pointLayer:
        x, y = float(p.attributes["local_x"]), float(p.attributes["local_y"])
        z_map = (p.z - T[2, 3] - r[0] * x - r[1] * y) / r[2]
        expected = T[:3, :3] @ np.array([x, y, z_map]) + T[:3, 3] - origin_utm
        assert abs(p.x - expected[0]) < 1e-3 and abs(p.y - expected[1]) < 1e-3
