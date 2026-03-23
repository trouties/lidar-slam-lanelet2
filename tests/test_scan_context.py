"""Tests for the Scan Context loop closure module."""

from __future__ import annotations

import numpy as np

from src.optimization.loop_closure import LoopClosureDetector
from src.optimization.scan_context import (
    ScanContextDatabase,
    compute_ring_key,
    make_scan_context,
    sc_distance,
)


def _make_cylinder_cloud(n: int = 5000, radius: float = 30.0) -> np.ndarray:
    """Generate a random cylindrical point cloud."""
    rng = np.random.RandomState(42)
    theta = rng.uniform(0, 2 * np.pi, n)
    r = rng.uniform(0, radius, n)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = rng.uniform(-1, 3, n)
    return np.column_stack([x, y, z])


class TestMakeScanContext:
    def test_shape(self):
        cloud = _make_cylinder_cloud()
        sc = make_scan_context(cloud, num_rings=20, num_sectors=60, max_range=40.0)
        assert sc.shape == (20, 60)
        assert sc.dtype == np.float32

    def test_empty_cloud(self):
        cloud = np.zeros((0, 3))
        sc = make_scan_context(cloud, num_rings=10, num_sectors=30)
        assert sc.shape == (10, 30)
        assert sc.max() == 0.0


class TestRingKey:
    def test_shape(self):
        sc = np.random.rand(20, 60).astype(np.float32)
        rk = compute_ring_key(sc)
        assert rk.shape == (20,)
        assert rk.dtype == np.float32

    def test_values(self):
        sc = np.ones((5, 10), dtype=np.float32) * 3.0
        rk = compute_ring_key(sc)
        np.testing.assert_allclose(rk, 3.0)


class TestSCDistance:
    def test_same_cloud(self):
        cloud = _make_cylinder_cloud()
        sc = make_scan_context(cloud)
        d = sc_distance(sc, sc)
        assert d < 0.01, f"Same cloud should have near-zero distance, got {d}"

    def test_rotation_invariance(self):
        cloud = _make_cylinder_cloud(n=10000)
        sc_a = make_scan_context(cloud)
        # Rotate 30° around z
        angle = np.radians(30)
        R = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ])
        rotated = cloud @ R.T
        sc_b = make_scan_context(rotated)
        d = sc_distance(sc_a, sc_b)
        assert d < 0.15, f"Rotated cloud should be similar, got {d}"

    def test_different_clouds(self):
        cloud_a = _make_cylinder_cloud(n=5000)
        rng = np.random.RandomState(99)
        cloud_b = rng.randn(5000, 3) * 20
        sc_a = make_scan_context(cloud_a)
        sc_b = make_scan_context(cloud_b)
        d = sc_distance(sc_a, sc_b)
        assert d > 0.1, f"Different clouds should have high distance, got {d}"


class TestScanContextDatabase:
    def test_add_and_query(self):
        db = ScanContextDatabase()
        # Make distinct clouds using different seeds
        clouds = []
        for i in range(5):
            rng = np.random.RandomState(i * 100)
            c = np.column_stack([
                rng.uniform(-30, 30, 3000),
                rng.uniform(-30, 30, 3000),
                rng.uniform(-1 + i, 3 + i, 3000),  # z offset per cloud
            ])
            clouds.append(c)
        for i, c in enumerate(clouds):
            sc = make_scan_context(c)
            rk = compute_ring_key(sc)
            db.add(sc, rk, i)

        # Query with same cloud → best match should be that frame
        query_sc = make_scan_context(clouds[2])
        query_rk = compute_ring_key(query_sc)
        results = db.query(query_sc, query_rk, top_k=3, min_frame_gap=0, current_frame=99)
        assert len(results) > 0
        assert results[0][0] == 2

    def test_min_frame_gap(self):
        db = ScanContextDatabase()
        cloud = _make_cylinder_cloud()
        for i in range(10):
            sc = make_scan_context(cloud)
            rk = compute_ring_key(sc)
            db.add(sc, rk, i)

        sc = make_scan_context(cloud)
        rk = compute_ring_key(sc)
        results = db.query(sc, rk, top_k=10, min_frame_gap=5, current_frame=7)
        for fid, _ in results:
            assert abs(fid - 7) >= 5


class _MockDataset:
    """Minimal dataset stub for LoopClosureDetector SC tests."""

    def __init__(self, clouds: list[np.ndarray]):
        self._clouds = clouds

    def __getitem__(self, idx):
        c = self._clouds[idx]
        # Return (pointcloud_with_reflectance, pose, timestamp)
        return np.column_stack([c, np.zeros(len(c))]), np.eye(4), 0.0

    def __len__(self):
        return len(self._clouds)


def _make_revisit_dataset(
    n_frames: int = 200,
    n_unique: int = 5,
    seed: int = 42,
) -> _MockDataset:
    """Create a dataset where frames cyclically revisit *n_unique* scenes.

    Frames 0..n_unique-1 each get a unique cloud; subsequent frames reuse
    them in round-robin order, so frame k has the same cloud as frame
    k % n_unique.  This guarantees multiple revisit pairs.
    """
    rng = np.random.RandomState(seed)
    base_clouds = []
    for i in range(n_unique):
        c = np.column_stack([
            rng.uniform(-30, 30, 3000),
            rng.uniform(-30, 30, 3000),
            rng.uniform(-1 + i * 2, 3 + i * 2, 3000),
        ])
        base_clouds.append(c)

    clouds = [base_clouds[k % n_unique] for k in range(n_frames)]
    return _MockDataset(clouds)


class TestDetectCandidatesSC:
    """Tests for LoopClosureDetector.detect_candidates_sc."""

    def test_returns_triples_with_distance(self):
        ds = _make_revisit_dataset(n_frames=150, n_unique=5)
        det = LoopClosureDetector(
            mode="v2", min_frame_gap=10,
            sc_distance_threshold=0.5, sc_query_stride=1,
            sc_max_matches_per_query=0,
        )
        cands = det.detect_candidates_sc(ds, len(ds))
        assert len(cands) > 0
        # Each candidate should be a (frame_i, frame_j, sc_dist) triple
        for item in cands:
            assert len(item) == 3
            i, j, d = item
            assert isinstance(i, (int, np.integer))
            assert isinstance(j, (int, np.integer))
            assert 0.0 <= d < 0.5

    def test_multi_match_per_query(self):
        ds = _make_revisit_dataset(n_frames=200, n_unique=3)
        det = LoopClosureDetector(
            mode="v2", min_frame_gap=10,
            sc_distance_threshold=0.5, sc_query_stride=1,
            sc_max_matches_per_query=0, sc_top_k=25,
        )
        cands = det.detect_candidates_sc(ds, len(ds))
        # With n_unique=3, many query frames should match multiple DB frames
        from collections import Counter
        query_counts = Counter(j for _, j, _ in cands)
        multi = [j for j, c in query_counts.items() if c > 1]
        assert len(multi) > 0, "Expected multiple matches per query frame"

    def test_max_matches_per_query_caps(self):
        ds = _make_revisit_dataset(n_frames=200, n_unique=3)
        det_unlimited = LoopClosureDetector(
            mode="v2", min_frame_gap=10,
            sc_distance_threshold=0.5, sc_query_stride=1,
            sc_max_matches_per_query=0, sc_top_k=25,
        )
        det_capped = LoopClosureDetector(
            mode="v2", min_frame_gap=10,
            sc_distance_threshold=0.5, sc_query_stride=1,
            sc_max_matches_per_query=2, sc_top_k=25,
        )
        cands_unlimited = det_unlimited.detect_candidates_sc(ds, len(ds))
        cands_capped = det_capped.detect_candidates_sc(ds, len(ds))

        # Capped should have fewer or equal candidates
        assert len(cands_capped) <= len(cands_unlimited)

        # No query frame should exceed max_matches_per_query
        from collections import Counter
        query_counts = Counter(j for _, j, _ in cands_capped)
        for j, count in query_counts.items():
            assert count <= 2, f"Frame {j} has {count} matches, expected <= 2"

    def test_query_stride(self):
        ds = _make_revisit_dataset(n_frames=200, n_unique=5)
        det_s1 = LoopClosureDetector(
            mode="v2", min_frame_gap=10,
            sc_distance_threshold=0.5, sc_query_stride=1,
            sc_max_matches_per_query=1,
        )
        det_s5 = LoopClosureDetector(
            mode="v2", min_frame_gap=10,
            sc_distance_threshold=0.5, sc_query_stride=5,
            sc_max_matches_per_query=1,
        )
        cands_s1 = det_s1.detect_candidates_sc(ds, len(ds))
        cands_s5 = det_s5.detect_candidates_sc(ds, len(ds))

        # Stride 1 should find more candidates than stride 5
        assert len(cands_s1) >= len(cands_s5)

        # All query frames in stride-5 result should be multiples of 5
        for _, j, _ in cands_s5:
            assert j % 5 == 0, f"Frame {j} is not a multiple of stride 5"
