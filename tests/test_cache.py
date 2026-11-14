"""Tests for the layered cache (src.cache.layered_cache)."""

from __future__ import annotations

import copy

import numpy as np
import open3d as o3d

from src.cache.layered_cache import (
    STAGE_ORDER,
    LayeredCache,
    _config_subtree,
    compute_hash,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _base_config() -> dict:
    return {
        "data": {"kitti_root": "~/data/kitti/odometry/dataset", "sequence": "00"},
        "kiss_icp": {"max_range": 100.0, "min_range": 5.0, "voxel_size": 1.0},
        "gtsam": {"odom_sigmas": [0.1] * 6, "prior_sigmas": [0.01] * 6},
        "loop_closure": {
            "min_frame_gap": 100,
            "icp_fitness_threshold": 0.9,
        },
        "mapping": {
            "voxel_size": 0.15,
            "max_range": 30.0,
            "downsample_every": 500,
            "master_voxel_size": 0.05,
            "curb_grid_size": 0.30,
            "curb_z_min": -2.0,
            "curb_z_max": -1.2,
            "curb_height_min": 0.10,
            "curb_height_max": 0.25,
            "curb_dbscan_eps": 0.5,
            "curb_dbscan_min_points": 40,
        },
    }


def _make_pcd(n: int = 500, seed: int = 0) -> o3d.geometry.PointCloud:
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-10.0, 10.0, (n, 3)).astype(np.float64)
    intensities = rng.uniform(0.0, 1.0, (n,)).astype(np.float64)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(np.repeat(intensities[:, None], 3, axis=1))
    return pcd


# ---------------------------------------------------------------------------
# Hash tests
# ---------------------------------------------------------------------------


def test_config_hash_stable():
    """Same config => same hash across repeat calls."""
    cfg = _base_config()
    h1 = compute_hash(_config_subtree("curbs", cfg))
    h2 = compute_hash(_config_subtree("curbs", copy.deepcopy(cfg)))
    assert h1 == h2
    assert len(h1) == 16


def test_config_hash_different_subtree():
    """Different stage subtrees => different hashes from same config."""
    cfg = _base_config()
    hashes = {s: LayeredCache.hash_for(s, cfg) for s in STAGE_ORDER}
    # All 5 stage hashes should be distinct (their subtrees don't overlap).
    assert len(set(hashes.values())) == len(hashes)


def test_config_hash_invalidates_on_curb_change():
    """Changing a curb parameter invalidates the curbs stage but not odometry."""
    cfg = _base_config()
    h_odom_before = LayeredCache.hash_for("odometry", cfg)
    h_curbs_before = LayeredCache.hash_for("curbs", cfg)

    cfg["mapping"]["curb_dbscan_eps"] = 0.6

    assert LayeredCache.hash_for("curbs", cfg) != h_curbs_before
    assert LayeredCache.hash_for("odometry", cfg) == h_odom_before


def test_config_hash_ignores_sequence():
    """Changing data.sequence does NOT change any hash (cache is sharded by seq on disk)."""
    cfg_a = _base_config()
    cfg_b = _base_config()
    cfg_b["data"]["sequence"] = "05"
    for stage in STAGE_ORDER:
        assert LayeredCache.hash_for(stage, cfg_a) == LayeredCache.hash_for(stage, cfg_b)


# ---------------------------------------------------------------------------
# Roundtrip tests
# ---------------------------------------------------------------------------


def test_roundtrip_odometry_cache(tmp_path):
    cfg = _base_config()
    cache = LayeredCache(tmp_path, sequence="00")

    assert cache.load_odometry(cfg) is None  # cold

    poses = np.stack([np.eye(4) for _ in range(5)]).astype(np.float64)
    poses[:, 0, 3] = np.arange(5, dtype=np.float64)
    timestamps = np.arange(5, dtype=np.float64) * 0.1

    cache.save_odometry(poses, timestamps, cfg, metrics={"frame_count": 5})
    loaded = cache.load_odometry(cfg)
    assert loaded is not None
    np.testing.assert_allclose(loaded[0], poses)
    np.testing.assert_allclose(loaded[1], timestamps)


def _seed_chain(cache: LayeredCache, cfg: dict, until: str) -> None:
    """Save every stage up to ``until`` so the upstream hashes line up."""
    dummy = np.stack([np.eye(4) for _ in range(3)]).astype(np.float64)
    cache.save_odometry(dummy, np.arange(3, dtype=np.float64), cfg)
    if until == "odometry":
        return
    cache.save_optimized(dummy, cfg)
    if until == "optimized":
        return
    cache.save_map(_make_pcd(), cfg)
    if until == "map":
        return
    cache.save_curbs(_make_pcd(seed=1), [np.array([[0.0, 0.0, 0.0]])], cfg)


def test_roundtrip_map_cache(tmp_path):
    cfg = _base_config()
    cache = LayeredCache(tmp_path, sequence="00")
    assert cache.load_map(cfg) is None
    _seed_chain(cache, cfg, "optimized")
    pcd = _make_pcd(n=300)
    cache.save_map(pcd, cfg, metrics={"point_count": 300})
    loaded = cache.load_map(cfg)
    assert loaded is not None
    np.testing.assert_allclose(np.asarray(loaded.points), np.asarray(pcd.points), atol=1e-6)
    # Reflectance survives at full float precision (the PCD writer would quantize it).
    np.testing.assert_allclose(
        np.asarray(loaded.colors)[:, 0], np.asarray(pcd.colors)[:, 0], atol=1e-6
    )


def test_roundtrip_curbs_cache(tmp_path):
    cfg = _base_config()
    cache = LayeredCache(tmp_path, sequence="00")
    _seed_chain(cache, cfg, "map")
    pcd = _make_pcd(n=200, seed=3)
    curbs = [np.array([[0.0, 0.0, -1.6], [1.0, 0.0, -1.6]]), np.array([[5.0, 5.0, -1.5]])]
    cache.save_curbs(pcd, curbs, cfg, metrics={"curb_cluster_count": 2})
    loaded = cache.load_curbs(cfg)
    assert loaded is not None
    working, loaded_curbs = loaded
    assert len(working.points) == 200
    assert len(loaded_curbs) == 2
    np.testing.assert_allclose(loaded_curbs[0], curbs[0])


def test_curbs_cache_invalidates_on_curb_param_change(tmp_path):
    cfg = _base_config()
    cache = LayeredCache(tmp_path, sequence="00")
    _seed_chain(cache, cfg, "curbs")
    assert cache.load_curbs(cfg) is not None
    cfg["mapping"]["curb_height_min"] = 0.08
    assert cache.load_curbs(cfg) is None
    # The map does not depend on the curb parameters.
    assert cache.load_map(cfg) is not None


# ---------------------------------------------------------------------------
# Invalidation tests
# ---------------------------------------------------------------------------


def test_cache_invalidates_on_upstream_change(tmp_path):
    """Changing odometry config (ie. kiss_icp params) invalidates optimized."""
    cfg = _base_config()
    cache = LayeredCache(tmp_path, sequence="00")

    dummy = np.stack([np.eye(4) for _ in range(3)]).astype(np.float64)
    cache.save_odometry(dummy, np.arange(3, dtype=np.float64), cfg)
    cache.save_optimized(dummy, cfg)

    # Optimized loads fine under unchanged config.
    assert cache.load_optimized(cfg) is not None

    # Change an upstream (Stage 2) param. Now optimized should refuse to load
    # because its stored upstream_hash no longer matches the current
    # odometry hash.
    cfg["kiss_icp"]["voxel_size"] = 0.5
    assert cache.load_optimized(cfg) is None


def test_cache_invalidates_all_downstream_on_upstream_change(tmp_path):
    """A Stage 2 change must also invalidate stages two or more levels down.

    Each stage's result depends on every upstream stage, so checking only the
    immediate upstream's config would let ``map`` (whose own subtree and whose
    upstream ``optimized`` subtree are both unchanged) load a map built from
    the old odometry.
    """
    cfg = _base_config()
    cache = LayeredCache(tmp_path, sequence="00")
    _seed_chain(cache, cfg, "curbs")
    assert cache.load_map(cfg) is not None
    assert cache.load_curbs(cfg) is not None

    cfg["data"]["correct_kitti_scan"] = True
    assert cache.load_map(cfg) is None
    assert cache.load_curbs(cfg) is None


def test_force_rebuild_propagates_downstream(tmp_path):
    """Invalidating the map also clears the curbs, but not the trajectories."""
    cfg = _base_config()
    cache = LayeredCache(tmp_path, sequence="00")
    _seed_chain(cache, cfg, "curbs")
    for load in (cache.load_odometry, cache.load_optimized, cache.load_map, cache.load_curbs):
        assert load(cfg) is not None

    cache.invalidate("map")

    assert cache.load_odometry(cfg) is not None
    assert cache.load_optimized(cfg) is not None
    assert cache.load_map(cfg) is None
    assert cache.load_curbs(cfg) is None


def test_force_rebuild_all_clears_everything(tmp_path):
    cfg = _base_config()
    cache = LayeredCache(tmp_path, sequence="00")

    dummy = np.stack([np.eye(4) for _ in range(3)]).astype(np.float64)
    cache.save_odometry(dummy, np.arange(3, dtype=np.float64), cfg)
    cache.save_optimized(dummy, cfg)

    cache.invalidate("all")

    assert cache.load_odometry(cfg) is None
    assert cache.load_optimized(cfg) is None
