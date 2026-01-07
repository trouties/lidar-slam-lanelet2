"""Tests for Stage 6: Lanelet2 OSM export."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import Counter

import numpy as np

from src.export.lanelet2_export import (
    classify_cluster,
    cluster_to_polygon,
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


def _thick_line(
    n: int = 200,
    length: float = 5.0,
    sigma_y: float = 0.4,
    seed: int = 1,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, length, n)
    y = rng.normal(0.0, sigma_y, n)
    z = np.full(n, -1.73)
    return np.column_stack([x, y, z])


def _blob(n: int = 200, side: float = 4.0, seed: int = 2) -> np.ndarray:
    """Roughly square blob in xy."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-side / 2.0, side / 2.0, size=(n, 2))
    z = np.full((n, 1), -1.73)
    return np.hstack([xy, z])


# ---------------------------------------------------------------------------
# classify_cluster
# ---------------------------------------------------------------------------


def test_classify_thin_line_cluster():
    label, stats = classify_cluster(_thin_line())
    assert label == "line_thin"
    assert stats["linearity"] > 0.9
    assert stats["length"] >= 4.0
    assert stats["thickness"] <= 0.8


def test_classify_thick_line_cluster():
    label, _ = classify_cluster(_thick_line())
    assert label == "line_thick"


def test_classify_area_blob():
    label, stats = classify_cluster(_blob())
    assert label == "area"
    assert stats["thickness"] > 2.0


def test_classify_too_short_returns_noise():
    short = _thin_line(n=10, length=0.3)
    label, _ = classify_cluster(short)
    assert label == "noise"


def test_classify_degenerate_returns_noise():
    two = np.array([[0.0, 0.0, -1.73], [1.0, 0.0, -1.73]])
    label, stats = classify_cluster(two)
    assert label == "noise"
    assert stats.get("degenerate") is True


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def test_cluster_to_polyline_orders_along_axis():
    cluster = _thin_line(n=80, length=8.0)
    rng = np.random.default_rng(7)
    rng.shuffle(cluster)  # in-place row shuffle
    label, stats = classify_cluster(cluster)
    assert label == "line_thin"

    polyline = cluster_to_polyline(cluster, stats, bin_size=0.5)
    assert polyline is not None
    assert polyline.shape[1] == 3
    assert polyline.shape[0] >= 2

    # Project polyline onto principal axis; expect monotonic order.
    centered = polyline[:, :2] - stats["mean_xy"]
    proj = centered @ stats["u"]
    diffs = np.diff(proj)
    assert np.all(diffs > 0) or np.all(diffs < 0)


def test_cluster_to_polygon_returns_four_corners():
    cluster = _blob()
    label, stats = classify_cluster(cluster)
    assert label == "area"

    polygon = cluster_to_polygon(cluster, stats)
    assert polygon.shape == (4, 3)

    # Polygon should bound the input points (within slack since we use
    # full min/max in the helper, no percentile clipping).
    px = polygon[:, 0]
    py = polygon[:, 1]
    slack = 1e-6
    assert px.min() <= cluster[:, 0].min() + slack
    assert px.max() >= cluster[:, 0].max() - slack
    assert py.min() <= cluster[:, 1].min() + slack
    assert py.max() >= cluster[:, 1].max() - slack


# ---------------------------------------------------------------------------
# End-to-end XML export
# ---------------------------------------------------------------------------


def test_export_lanelet2_osm_writes_valid_xml(tmp_path):
    clusters = [
        _thin_line(n=80, length=6.0, seed=10),
        _blob(n=300, side=5.0, seed=11),
        _thin_line(n=10, length=0.3, seed=12),  # too short -> noise
    ]
    out = tmp_path / "map_test.osm"

    counts = export_lanelet2_osm(clusters, out)

    # File exists and parses.
    assert out.exists()
    tree = ET.parse(out)
    root = tree.getroot()
    assert root.tag == "osm"
    assert root.attrib.get("version") == "0.6"

    # Exactly one bounds element.
    bounds = root.findall("bounds")
    assert len(bounds) == 1

    # Way and tag inventory.
    ways = root.findall("way")
    assert len(ways) == 2  # one polyline + one area; the noise cluster is dropped
    way_types = Counter(
        tag.get("v") for w in ways for tag in w.findall("tag") if tag.get("k") == "type"
    )
    assert way_types["line_thin"] == 1
    assert way_types["zebra_marking"] == 1

    # Area way must close: first and last <nd ref> identical, and tagged area=yes.
    area_way = next(
        w
        for w in ways
        if any(t.get("k") == "type" and t.get("v") == "zebra_marking" for t in w.findall("tag"))
    )
    nd_refs = [nd.get("ref") for nd in area_way.findall("nd")]
    assert len(nd_refs) >= 4
    assert nd_refs[0] == nd_refs[-1]
    assert any(t.get("k") == "area" and t.get("v") == "yes" for t in area_way.findall("tag"))

    # Conservation invariant.
    assert counts["total_input"] == 3
    assert counts["line_thin"] + counts["line_thick"] + counts["area"] + counts["dropped"] == 3
    assert counts["line_thin"] == 1
    assert counts["area"] == 1
    assert counts["dropped"] == 1
