"""Tests for Stage 5: map assembly and feature extraction."""

from __future__ import annotations

import json

import numpy as np

from src.mapping import (
    MapBuilder,
    cluster_points,
    extract_curbs,
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


def test_cluster_points_trims_minor_axis_outliers():
    """A clean thin line along x plus a few y=0.5 bridge fragments should
    come out as one cluster with the bridge points removed by the MAD trim.

    The bridge points are close enough (y=0.5 < eps=0.7) to get absorbed
    into the DBSCAN cluster, but they lie far enough outside the ~3 cm
    MAD of the real paint line that the minor-axis trim should catch them.
    """
    rng = np.random.default_rng(7)
    line = np.column_stack(
        [
            np.linspace(0.0, 5.0, 200),
            rng.normal(0.0, 0.03, size=200),  # paint-width sigma ~3 cm
            np.full(200, -1.73),
        ]
    )
    # Bridge points: near-enough on x to be connected into the cluster, but
    # offset ~0.5 m on y (far beyond the line's real thickness).
    bridge = np.array(
        [
            [1.0, 0.5, -1.73],
            [2.0, 0.52, -1.73],
            [3.0, 0.48, -1.73],
            [4.0, 0.51, -1.73],
        ]
    )
    points = np.vstack([line, bridge])

    clusters = cluster_points(points, eps=0.7, min_points=40)

    assert len(clusters) == 1
    result = clusters[0]
    # Bridge outliers gone: minor axis span well under their 0.5 m offset.
    assert float(np.ptp(result[:, 1])) < 0.3
    # Line inliers preserved; tiny MAD clip on the ~3cm tails is acceptable.
    assert 190 <= result.shape[0] <= 204


def test_cluster_points_trim_preserves_clean_line():
    """Without outliers, the trim should leave a clean line nearly untouched."""
    rng = np.random.default_rng(11)
    line = np.column_stack(
        [
            np.linspace(0.0, 5.0, 200),
            rng.normal(0.0, 0.03, size=200),
            np.full(200, -1.73),
        ]
    )

    clusters = cluster_points(line, eps=0.7, min_points=40)

    assert len(clusters) == 1
    assert clusters[0].shape[0] >= int(0.96 * line.shape[0])


def test_cluster_points_drops_below_min_points_after_trim():
    """A cluster that survives DBSCAN but shrinks below min_points after
    outlier trim should be dropped entirely."""
    rng = np.random.default_rng(3)
    # Dense core of 42 points (just above min_points=40) on a short line.
    core = np.column_stack(
        [
            np.linspace(0.0, 1.0, 42),
            rng.normal(0.0, 0.03, size=42),
            np.full(42, -1.73),
        ]
    )
    # 8 bridge outliers connected by a chain on x.
    outliers = np.column_stack(
        [
            np.linspace(0.1, 0.9, 8),
            np.full(8, 0.5),
            np.full(8, -1.73),
        ]
    )
    points = np.vstack([core, outliers])

    # The raw union (50 pts) passes DBSCAN; after MAD trim the 8 bridge
    # points get chopped, leaving ~42 inliers. Still >= min_points=40 so
    # the cluster survives. To force the drop, raise min_points to 45.
    clusters = cluster_points(points, eps=0.7, min_points=45, trim_k=2.5)

    assert clusters == []


def test_cluster_points_trim_disabled_is_regression_anchor():
    """trim_k=None reproduces the pre-trim behavior — used as a regression
    anchor so we can verify the new default against the legacy path."""
    rng = np.random.default_rng(4)
    line = np.column_stack(
        [
            np.linspace(0.0, 5.0, 200),
            rng.normal(0.0, 0.03, size=200),
            np.full(200, -1.73),
        ]
    )
    outliers = np.column_stack(
        [
            np.linspace(0.5, 4.5, 10),
            np.full(10, 0.5),
            np.full(10, -1.73),
        ]
    )
    points = np.vstack([line, outliers])

    clusters_disabled = cluster_points(points, eps=0.7, min_points=40, trim_k=None)
    clusters_default = cluster_points(points, eps=0.7, min_points=40)

    # Disabled path keeps all 210 points in one cluster; default path trims.
    assert len(clusters_disabled) == 1
    assert clusters_disabled[0].shape[0] == 210
    assert len(clusters_default) == 1
    assert clusters_default[0].shape[0] < 210


# ---------------------------------------------------------------------------
# Curb detection tests
# ---------------------------------------------------------------------------


def _flat_ground_xyz(
    n: int,
    z: float = -1.73,
    x_range: tuple[float, float] = (-5.0, 5.0),
    y_range: tuple[float, float] = (-2.5, 2.5),
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.uniform(*x_range, size=n)
    y = rng.uniform(*y_range, size=n)
    z_col = np.full(n, z, dtype=np.float64)
    return np.column_stack([x, y, z_col])


def test_extract_curbs_detects_height_step():
    """A 0.15 m ridge along y=0 on a flat ground patch should light up the
    curb detector in cells covering the ridge and return only the top-of-
    step slab."""
    ground = _flat_ground_xyz(n=4000, z=-1.73, seed=1)
    # 10 m long, 0.4 m wide raised strip at y in [0.0, 0.4], z = -1.58.
    rng = np.random.default_rng(2)
    n_ridge = 1500
    ridge = np.column_stack(
        [
            rng.uniform(-5.0, 5.0, size=n_ridge),
            rng.uniform(0.0, 0.4, size=n_ridge),
            np.full(n_ridge, -1.58),
        ]
    )
    points = np.vstack([ground, ridge])

    curb_pts = extract_curbs(
        points,
        grid_size=0.3,
        z_min=-2.0,
        z_max=-1.2,
        height_min=0.10,
        height_max=0.25,
        top_band=0.03,
    )

    assert curb_pts.shape[0] > 50, f"expected many curb-top points, got {curb_pts.shape[0]}"
    # All returned points should be on the ridge (y near [0, 0.4], z near -1.58).
    assert curb_pts[:, 1].min() >= -0.05
    assert curb_pts[:, 1].max() <= 0.45
    np.testing.assert_allclose(curb_pts[:, 2], -1.58, atol=0.05)


def test_extract_curbs_ignores_flat_ground():
    """A pure flat-ground patch has no height steps, so the detector must
    return an empty point set."""
    ground = _flat_ground_xyz(n=10_000, z=-1.73, seed=3)

    curb_pts = extract_curbs(
        ground,
        grid_size=0.3,
        z_min=-2.0,
        z_max=-1.2,
        height_min=0.10,
        height_max=0.25,
    )

    assert curb_pts.shape == (0, 3)


def test_extract_curbs_rejects_tall_walls():
    """A 0.5 m riser (above ``height_max``) must be rejected — curbs are
    specifically the short-step regime; taller jumps are walls / vehicles."""
    ground = _flat_ground_xyz(n=3000, z=-1.73, seed=4)
    rng = np.random.default_rng(5)
    n_wall = 1500
    wall = np.column_stack(
        [
            rng.uniform(-5.0, 5.0, size=n_wall),
            rng.uniform(0.0, 0.4, size=n_wall),
            # A 0.5 m riser: wall tops at z = -1.23, which is ABOVE
            # height_max=0.25 relative to the ground at -1.73.
            np.full(n_wall, -1.23),
        ]
    )
    points = np.vstack([ground, wall])

    curb_pts = extract_curbs(
        points,
        grid_size=0.3,
        z_min=-2.0,
        z_max=-1.2,
        height_min=0.10,
        height_max=0.25,
    )

    assert curb_pts.shape == (0, 3)


def test_extract_curbs_empty_input():
    """Empty input -> empty (0, 3) output, no crash."""
    out = extract_curbs(np.zeros((0, 3)))
    assert out.shape == (0, 3)


def test_extract_curbs_respects_z_window():
    """Points whose z is entirely outside [z_min, z_max] are filtered even
    if they would otherwise form a valid height step."""
    # A 0.15 m ridge, but the entire thing is above z_max=-1.2 -> rejected.
    rng = np.random.default_rng(6)
    high_floor = np.column_stack(
        [
            rng.uniform(-5.0, 5.0, size=2000),
            rng.uniform(-2.0, 2.0, size=2000),
            np.full(2000, -0.5),  # well above z_max
        ]
    )
    high_ridge = np.column_stack(
        [
            rng.uniform(-5.0, 5.0, size=1000),
            rng.uniform(0.0, 0.3, size=1000),
            np.full(1000, -0.35),
        ]
    )
    points = np.vstack([high_floor, high_ridge])

    curb_pts = extract_curbs(
        points,
        grid_size=0.3,
        z_min=-2.0,
        z_max=-1.2,
        height_min=0.10,
        height_max=0.25,
    )

    assert curb_pts.shape == (0, 3)


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
