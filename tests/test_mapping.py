"""Tests for Stage 5: map assembly and feature extraction."""

from __future__ import annotations

import json

import numpy as np

from src.mapping import (
    MapBuilder,
    cluster_points,
    extract_lane_markings,
    extract_road_surface,
    save_features_geojson,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_flat_ground(
    n: int,
    z: float = -1.73,
    size: float = 10.0,
    intensity: float = 0.2,
    seed: int = 0,
) -> np.ndarray:
    """Create a (n, 4) flat patch of ground at fixed z with uniform intensity."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-size, size, (n, 2))
    z_col = np.full((n, 1), z)
    i_col = np.full((n, 1), intensity)
    return np.hstack([xy, z_col, i_col]).astype(np.float32)


def _translate(tx: float = 0.0, ty: float = 0.0, tz: float = 0.0) -> np.ndarray:
    T = np.eye(4)
    T[0, 3] = tx
    T[1, 3] = ty
    T[2, 3] = tz
    return T


class _FakeDataset:
    """Minimal dataset stub that yields (points, None, None) per index."""

    def __init__(self, frames: list[np.ndarray]) -> None:
        self._frames = frames

    def __len__(self) -> int:
        return len(self._frames)

    def __getitem__(self, idx: int):
        return self._frames[idx], None, None


# ---------------------------------------------------------------------------
# MapBuilder tests
# ---------------------------------------------------------------------------


def test_map_builder_applies_pose():
    """Two identical frames at different poses should produce two spatially
    separated clusters in the accumulated map."""
    frame = _make_flat_ground(n=500, z=-1.73, size=2.0)
    dataset = _FakeDataset([frame.copy(), frame.copy()])
    poses = [_translate(tx=0.0), _translate(tx=100.0)]

    builder = MapBuilder(voxel_size=0.2, max_range=50.0, downsample_every=10)
    pcd = builder.build(dataset, poses)

    xyz = np.asarray(pcd.points)
    assert xyz.shape[0] > 0
    assert xyz[:, 0].min() < 10.0  # first frame near origin
    assert xyz[:, 0].max() > 90.0  # second frame shifted +100 m


def test_map_builder_voxel_downsample_reduces_count():
    """Dense random cloud should shrink after voxel downsampling."""
    frame = _make_flat_ground(n=5000, z=-1.73, size=1.0)
    dataset = _FakeDataset([frame])
    builder = MapBuilder(voxel_size=0.5, max_range=50.0, downsample_every=10)
    pcd = builder.build(dataset, [np.eye(4)])

    assert len(pcd.points) < 5000
    assert len(pcd.points) > 0


def test_map_builder_preserves_intensity_via_colors():
    """Intensity encoded in colors channel should survive voxel downsampling."""
    frame = _make_flat_ground(n=2000, z=-1.73, size=2.0, intensity=0.7)
    dataset = _FakeDataset([frame])
    builder = MapBuilder(voxel_size=0.2, max_range=50.0, downsample_every=10)
    pcd = builder.build(dataset, [np.eye(4)])

    colors = np.asarray(pcd.colors)
    assert colors.shape[0] > 0
    np.testing.assert_allclose(colors[:, 0], 0.7, atol=1e-5)
    np.testing.assert_allclose(colors[:, 1], colors[:, 0])
    np.testing.assert_allclose(colors[:, 2], colors[:, 0])


def test_map_builder_max_range_filter():
    """Points beyond max_range should be dropped before accumulation."""
    close = np.array([[1.0, 0.0, -1.73, 0.2]], dtype=np.float32)
    far = np.array([[100.0, 0.0, -1.73, 0.2]], dtype=np.float32)
    frame = np.vstack([close, far])
    dataset = _FakeDataset([frame])

    builder = MapBuilder(voxel_size=0.1, max_range=10.0, downsample_every=10)
    pcd = builder.build(dataset, [np.eye(4)])

    xyz = np.asarray(pcd.points)
    assert xyz.shape[0] == 1
    np.testing.assert_allclose(xyz[0], [1.0, 0.0, -1.73], atol=1e-5)


# ---------------------------------------------------------------------------
# feature_extraction tests
# ---------------------------------------------------------------------------


def test_extract_road_surface_filters_by_height():
    """Only points within [z_min, z_max] should be kept."""
    ground = np.array([[0.0, 0.0, -1.73]])
    wall = np.array([[0.0, 0.0, 0.5]])
    roof = np.array([[0.0, 0.0, 3.0]])
    points = np.vstack([ground, wall, roof])
    intensities = np.array([0.3, 0.1, 0.1])

    road_pts, road_int = extract_road_surface(points, intensities, z_min=-1.95, z_max=-1.45)

    assert road_pts.shape == (1, 3)
    np.testing.assert_allclose(road_pts[0], ground[0])
    np.testing.assert_allclose(road_int, [0.3])


def test_extract_lane_markings_intensity_threshold():
    """Only points at/above intensity_threshold should survive."""
    points = np.array(
        [
            [0.0, 0.0, -1.73],
            [1.0, 0.0, -1.73],
            [2.0, 0.0, -1.73],
            [3.0, 0.0, -1.73],
        ]
    )
    intensities = np.array([0.1, 0.5, 0.2, 0.9])

    lane = extract_lane_markings(points, intensities, intensity_threshold=0.35)

    assert lane.shape == (2, 3)
    np.testing.assert_allclose(lane[0], [1.0, 0.0, -1.73])
    np.testing.assert_allclose(lane[1], [3.0, 0.0, -1.73])


def test_cluster_points_finds_two_groups():
    """Two spatially separated blobs should yield exactly two clusters."""
    rng = np.random.default_rng(42)
    blob_a = rng.normal(loc=[0.0, 0.0, 0.0], scale=0.05, size=(50, 3))
    blob_b = rng.normal(loc=[10.0, 10.0, 0.0], scale=0.05, size=(50, 3))
    points = np.vstack([blob_a, blob_b])

    clusters = cluster_points(points, eps=0.5, min_points=5)

    assert len(clusters) == 2
    sizes = sorted(c.shape[0] for c in clusters)
    assert sizes[0] >= 30 and sizes[1] >= 30


def test_save_features_geojson_writes_valid_json(tmp_path):
    """GeoJSON file should be valid JSON with the expected structure."""
    clusters = [
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        np.array([[10.0, 10.0, 0.0]]),
    ]
    out = tmp_path / "features.geojson"

    save_features_geojson(clusters, out, feature_type="lane_marking")

    assert out.exists()
    with open(out) as f:
        data = json.load(f)

    assert data["type"] == "FeatureCollection"
    assert len(data["features"]) == 2
    assert data["features"][0]["geometry"]["type"] == "MultiPoint"
    assert len(data["features"][0]["geometry"]["coordinates"]) == 2
    assert data["features"][0]["properties"]["type"] == "lane_marking"
    assert data["features"][0]["properties"]["point_count"] == 2
    assert data["features"][1]["properties"]["point_count"] == 1
