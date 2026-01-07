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
            "distance_threshold": 15.0,
            "min_frame_gap": 100,
            "icp_fitness_threshold": 0.9,
        },
        "ekf": {"process_noise_pos": 0.1},
        "mapping": {
            "voxel_size": 0.15,
            "max_range": 30.0,
            "downsample_every": 500,
            "master_voxel_size": 0.05,
            "road_z_min": -2.0,
            "road_z_max": -1.5,
            "intensity_threshold": 0.40,
            "dbscan_eps": 0.7,
            "dbscan_min_points": 40,
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
    h1 = compute_hash(_config_subtree("stage5", cfg))
    h2 = compute_hash(_config_subtree("stage5", copy.deepcopy(cfg)))
    assert h1 == h2
    assert len(h1) == 16


def test_config_hash_different_subtree():
    """Different stage subtrees => different hashes from same config."""
    cfg = _base_config()
    hashes = {s: LayeredCache.hash_for(s, cfg) for s in STAGE_ORDER}
    # All 5 stage hashes should be distinct (their subtrees don't overlap).
    assert len(set(hashes.values())) == len(hashes)


def test_config_hash_invalidates_on_dbscan_change():
    """Changing mapping.dbscan_eps invalidates stage5 but not odometry."""
    cfg = _base_config()
    h_odom_before = LayeredCache.hash_for("odometry", cfg)
    h_stage5_before = LayeredCache.hash_for("stage5", cfg)

    cfg["mapping"]["dbscan_eps"] = 0.5

    assert LayeredCache.hash_for("odometry", cfg) == h_odom_before
    assert LayeredCache.hash_for("stage5", cfg) != h_stage5_before


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


def test_roundtrip_map_master_cache(tmp_path):
    cfg = _base_config()
    cache = LayeredCache(tmp_path, sequence="00")

    assert cache.load_global_map_master(cfg) is None

    # Map master also requires fused to have been saved so the upstream_hash
    # matches. Seed the chain up to map_master.
    dummy_poses = np.stack([np.eye(4) for _ in range(3)]).astype(np.float64)
    cache.save_odometry(dummy_poses, np.arange(3, dtype=np.float64), cfg)
    cache.save_optimized(dummy_poses, cfg)
    cache.save_fused(dummy_poses, cfg)

    pcd = _make_pcd(n=300, seed=1)
    cache.save_global_map_master(pcd, cfg, metrics={"point_count": 300})

    loaded = cache.load_global_map_master(cfg)
    assert loaded is not None
    assert len(loaded.points) == len(pcd.points)
    # Reflectance round-trips via the colors channel.
    np.testing.assert_allclose(
        np.asarray(loaded.colors)[:, 0],
        np.asarray(pcd.colors)[:, 0],
        atol=1e-5,
    )


def test_roundtrip_stage5_cache(tmp_path):
    cfg = _base_config()
    cache = LayeredCache(tmp_path, sequence="00")

    # Walk the chain so upstream_hash checks line up for stage5.
    dummy = np.stack([np.eye(4) for _ in range(3)]).astype(np.float64)
    cache.save_odometry(dummy, np.arange(3, dtype=np.float64), cfg)
    cache.save_optimized(dummy, cfg)
    cache.save_fused(dummy, cfg)
    cache.save_global_map_master(_make_pcd(), cfg)

    pcd = _make_pcd(n=100, seed=2)
    clusters = [
        np.array([[0.0, 0.0, -1.7], [1.0, 0.0, -1.7], [2.0, 0.0, -1.7]]),
        np.array([[5.0, 5.0, -1.7], [5.1, 5.0, -1.7]]),
    ]
    cache.save_stage5(pcd, clusters, cfg, metrics={"cluster_count": 2})

    loaded = cache.load_stage5(cfg)
    assert loaded is not None
    loaded_pcd, loaded_clusters = loaded
    assert len(loaded_pcd.points) == len(pcd.points)
    assert len(loaded_clusters) == 2
    np.testing.assert_allclose(loaded_clusters[0], clusters[0])
    np.testing.assert_allclose(loaded_clusters[1], clusters[1])


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


def test_force_rebuild_propagates_downstream(tmp_path):
    """Invalidating map_master also clears stage5."""
    cfg = _base_config()
    cache = LayeredCache(tmp_path, sequence="00")

    dummy = np.stack([np.eye(4) for _ in range(3)]).astype(np.float64)
    cache.save_odometry(dummy, np.arange(3, dtype=np.float64), cfg)
    cache.save_optimized(dummy, cfg)
    cache.save_fused(dummy, cfg)
    cache.save_global_map_master(_make_pcd(), cfg)
    cache.save_stage5(_make_pcd(), [np.array([[0.0, 0.0, 0.0]])], cfg)

    # Sanity: all four stages load.
    assert cache.load_odometry(cfg) is not None
    assert cache.load_optimized(cfg) is not None
    assert cache.load_fused(cfg) is not None
    assert cache.load_global_map_master(cfg) is not None
    assert cache.load_stage5(cfg) is not None

    cache.invalidate("map_master")

    # Map_master and stage5 are gone, but upstream survives.
    assert cache.load_odometry(cfg) is not None
    assert cache.load_optimized(cfg) is not None
    assert cache.load_fused(cfg) is not None
    assert cache.load_global_map_master(cfg) is None
    assert cache.load_stage5(cfg) is None


def test_force_rebuild_all_clears_everything(tmp_path):
    cfg = _base_config()
    cache = LayeredCache(tmp_path, sequence="00")

    dummy = np.stack([np.eye(4) for _ in range(3)]).astype(np.float64)
    cache.save_odometry(dummy, np.arange(3, dtype=np.float64), cfg)
    cache.save_optimized(dummy, cfg)

    cache.invalidate("all")

    assert cache.load_odometry(cfg) is None
    assert cache.load_optimized(cfg) is None
