"""Tests for the Scan Context loop closure module."""

from __future__ import annotations

import numpy as np

from src.optimization.loop_closure import LoopClosureDetector
from src.optimization.scan_context import (
    ScanContextDatabase,
    compute_ring_key,
    make_scan_context,
    sc_distance,
    sc_relative_yaw,
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
        R = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        )
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
            c = np.column_stack(
                [
                    rng.uniform(-30, 30, 3000),
                    rng.uniform(-30, 30, 3000),
                    rng.uniform(-1 + i, 3 + i, 3000),  # z offset per cloud
                ]
            )
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
        c = np.column_stack(
            [
                rng.uniform(-30, 30, 3000),
                rng.uniform(-30, 30, 3000),
                rng.uniform(-1 + i * 2, 3 + i * 2, 3000),
            ]
        )
        base_clouds.append(c)

    clouds = [base_clouds[k % n_unique] for k in range(n_frames)]
    return _MockDataset(clouds)


class TestDetectCandidatesSC:
    """Tests for LoopClosureDetector.detect_candidates_sc."""

    def test_returns_triples_with_distance(self):
        ds = _make_revisit_dataset(n_frames=150, n_unique=5)
        det = LoopClosureDetector(
            min_frame_gap=10,
            sc_distance_threshold=0.5,
            sc_query_stride=1,
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
            min_frame_gap=10,
            sc_distance_threshold=0.5,
            sc_query_stride=1,
            sc_max_matches_per_query=0,
            sc_top_k=25,
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
            min_frame_gap=10,
            sc_distance_threshold=0.5,
            sc_query_stride=1,
            sc_max_matches_per_query=0,
            sc_top_k=25,
        )
        det_capped = LoopClosureDetector(
            min_frame_gap=10,
            sc_distance_threshold=0.5,
            sc_query_stride=1,
            sc_max_matches_per_query=2,
            sc_top_k=25,
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
            min_frame_gap=10,
            sc_distance_threshold=0.5,
            sc_query_stride=1,
            sc_max_matches_per_query=1,
        )
        det_s5 = LoopClosureDetector(
            min_frame_gap=10,
            sc_distance_threshold=0.5,
            sc_query_stride=5,
            sc_max_matches_per_query=1,
        )
        cands_s1 = det_s1.detect_candidates_sc(ds, len(ds))
        cands_s5 = det_s5.detect_candidates_sc(ds, len(ds))

        # Stride 1 should find more candidates than stride 5
        assert len(cands_s1) >= len(cands_s5)

        # All query frames in stride-5 result should be multiples of 5
        for _, j, _ in cands_s5:
            assert j % 5 == 0, f"Frame {j} is not a multiple of stride 5"


class TestRelativeYaw:
    def test_recovers_rotation_and_direction(self):
        """Rz(yaw) must rotate query-frame points into the matched scan's frame."""
        rng = np.random.RandomState(7)
        # Objects of random height scattered at all ranges and bearings, so
        # that every sector has a distinct ring profile. (A scene where each
        # sector holds a single ring makes several shifts tie, because the
        # column distance ignores empty columns.)
        n_obj = 400
        r_obj = rng.uniform(4.0, 70.0, n_obj)
        a_obj = rng.uniform(-np.pi, np.pi, n_obj)
        h_obj = rng.uniform(-1.0, 6.0, n_obj)
        walls = [
            np.column_stack(
                [
                    r * np.cos(a) + rng.normal(0, 0.3, 20),
                    r * np.sin(a) + rng.normal(0, 0.3, 20),
                    rng.uniform(-1.7, h, 20),
                ]
            )
            for r, a, h in zip(r_obj, a_obj, h_obj)
        ]
        match = np.vstack(walls)
        for yaw_true in np.radians([30.0, 96.0, 222.0]):
            # Query frame rotated by yaw_true w.r.t. the matched frame:
            # p_match = Rz(yaw_true) p_query, so p_query = Rz(-yaw_true) p_match.
            c, s_ = np.cos(-yaw_true), np.sin(-yaw_true)
            query = match @ np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]]).T
            yaw = sc_relative_yaw(make_scan_context(query), make_scan_context(match))
            err = np.angle(np.exp(1j * (yaw - yaw_true)))
            assert abs(np.degrees(err)) <= 6.0


def _scene(seed: int) -> np.ndarray:
    """Ground plane plus posts and boxes of random height: structure for point-to-plane ICP."""
    rng = np.random.RandomState(seed)
    ground = np.column_stack(
        [rng.uniform(-40, 40, 6000), rng.uniform(-40, 40, 6000), rng.normal(-1.73, 0.01, 6000)]
    )
    objs = []
    for _ in range(120):
        cx, cy = rng.uniform(-35, 35, 2)
        w, d, h = rng.uniform(0.3, 4.0), rng.uniform(0.3, 4.0), rng.uniform(0.5, 6.0)
        n = 150
        face = rng.randint(0, 4, n)
        u = rng.uniform(-0.5, 0.5, n)
        x = np.where(face < 2, cx + (face - 0.5) * w, cx + u * w)
        y = np.where(face < 2, cy + u * d, cy + (face - 2.5) * d)
        objs.append(np.column_stack([x, y, rng.uniform(-1.73, -1.73 + h, n)]))
    return np.vstack([ground, *objs])


class TestRetryFromScanContext:
    """The last scan revisits the first, but the odometry puts it 15 m away."""

    yaw = np.radians(40.0)
    shift = np.array([1.0, 0.0, 0.0])

    def _setup(self):
        n = 40
        first = _scene(0)
        # p_first = Rz(yaw) p_last + shift  ->  p_last = Rz(-yaw) (p_first - shift)
        c, s = np.cos(-self.yaw), np.sin(-self.yaw)
        last = (first - self.shift) @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]).T
        clouds = [first] + [_scene(100 + k) for k in range(1, n - 1)] + [last]
        poses = []
        for k in range(n):
            T = np.eye(4)
            T[0, 3] = 20.0 * k
            poses.append(T)
        poses[-1] = np.eye(4)
        poses[-1][0, 3] = 15.0  # odometry: 15 m from the start, no rotation
        return _MockDataset(clouds), poses

    def _detector(self, **kw):
        return LoopClosureDetector(
            min_frame_gap=10,
            icp_fitness_threshold=0.9,
            sc_distance_threshold=0.3,
            refine_stages=[(2.0, 5.0), (1.0, 2.0), (0.5, 1.0)],
            **kw,
        )

    def test_without_retry_the_revisit_is_missed(self):
        ds, poses = self._setup()
        assert self._detector().detect(poses, ds) == []

    def test_retry_closes_it(self):
        ds, poses = self._setup()
        closures = self._detector(retry_from_scan_context=True, max_drift_ratio=0.02).detect(
            poses, ds
        )
        assert [(i, j) for i, j, _ in closures] == [(0, len(poses) - 1)]
        T = closures[0][2]
        assert np.linalg.norm(T[:3, 3] - self.shift) < 0.1
        assert abs(np.arctan2(T[1, 0], T[0, 0]) - self.yaw) < np.radians(0.5)

    def test_drift_check_rejects_it_when_too_tight(self):
        # The closure moves the last scan by about 14 m over about 1.5 km of path (0.9 %).
        ds, poses = self._setup()
        det = self._detector(retry_from_scan_context=True, max_drift_ratio=0.005)
        assert det.detect(poses, ds) == []

    def test_retry_needs_refinement(self):
        import pytest

        with pytest.raises(ValueError):
            LoopClosureDetector(retry_from_scan_context=True)
