"""Loop closure detection.

Detects revisited places using either distance-based candidate search (v1)
or Scan Context appearance matching (v2), then verifies with ICP point
cloud registration.
"""

from __future__ import annotations

import numpy as np
import open3d as o3d

from src.benchmarks.timing import StageTimer
from src.optimization.scan_context import (
    ScanContextDatabase,
    compute_ring_key,
    make_scan_context,
)


class LoopClosureDetector:
    """Loop closure detector with v1 (distance) and v2 (Scan Context) modes.

    v1: detects candidates by comparing pose translations.
    v2: detects candidates using Scan Context appearance descriptors.

    Both modes optionally verify with Open3D ICP.
    """

    def __init__(
        self,
        distance_threshold: float = 15.0,
        min_frame_gap: int = 100,
        icp_fitness_threshold: float = 0.3,
        mode: str = "v1",
        sc_num_rings: int = 20,
        sc_num_sectors: int = 60,
        sc_max_range: float = 80.0,
        sc_distance_threshold: float = 0.4,
        sc_top_k: int = 10,
        sc_query_stride: int = 1,
        sc_max_matches_per_query: int = 0,
        icp_downsample_voxel: float = 1.0,
    ) -> None:
        """Initialize detector.

        Args:
            distance_threshold: Maximum distance (m) between poses to be a candidate (v1).
            min_frame_gap: Minimum frame index gap to avoid detecting neighbors.
            icp_fitness_threshold: Minimum ICP fitness to accept a match.
            mode: ``'v1'`` (distance), ``'v2'`` (scan context), or ``'both'``.
            sc_num_rings: Number of radial bins for Scan Context.
            sc_num_sectors: Number of angular bins for Scan Context.
            sc_max_range: Maximum range for Scan Context descriptor.
            sc_distance_threshold: Maximum SC distance to consider a candidate.
            sc_top_k: Number of ring-key candidates to re-rank per query.
            sc_query_stride: Query every N-th frame (1 = every frame).
            sc_max_matches_per_query: Max matches per query frame (0 = unlimited).
            icp_downsample_voxel: Voxel size (m) used to downsample candidate
                point clouds before ICP. Defaults to 1.0 m (the SUP-03
                round-2 locked value on KITTI 64-beam Velodyne). Finer voxels
                trade speed for registration accuracy; coarser voxels flip
                the tradeoff. Cross-sensor work (e.g., nuScenes 32-beam)
                typically wants a smaller value.
        """
        self.distance_threshold = distance_threshold
        self.min_frame_gap = min_frame_gap
        self.icp_fitness_threshold = icp_fitness_threshold
        self.mode = mode
        self.sc_num_rings = sc_num_rings
        self.sc_num_sectors = sc_num_sectors
        self.sc_max_range = sc_max_range
        self.sc_distance_threshold = sc_distance_threshold
        self.sc_top_k = sc_top_k
        self.sc_query_stride = sc_query_stride
        self.sc_max_matches_per_query = sc_max_matches_per_query
        self.icp_downsample_voxel = icp_downsample_voxel
        # Sub-stage timers populated during detect(); read summary() after.
        # Reset on every detect() call so reusing a detector across sequences
        # does not accumulate stale measurements.
        self.sc_query_timer = StageTimer("stage3_sc_query")
        self.icp_verify_timer = StageTimer("stage3_icp_verify")
        # Per-detect() downsample cache: frame_id -> downsampled Open3D cloud.
        # Reset at the start of detect(). See _get_cached_downsampled_pcd.
        self._downsample_cache: dict[int, o3d.geometry.PointCloud] = {}

    def detect_candidates(self, poses: list[np.ndarray]) -> list[tuple[int, int]]:
        """Find loop closure candidates based on pose distance (v1).

        For each frame j, finds the single closest earlier frame i
        (with sufficient gap) below the distance threshold.

        Args:
            poses: List of 4x4 SE(3) poses.

        Returns:
            List of (i, j) pairs where i < j and poses are close.
        """
        n = len(poses)
        translations = np.array([p[:3, 3] for p in poses])
        candidates = []

        for j in range(self.min_frame_gap, n):
            # Compare against all earlier poses with sufficient gap
            search_end = j - self.min_frame_gap + 1
            diffs = translations[:search_end] - translations[j]
            dists = np.linalg.norm(diffs, axis=1)
            min_idx = int(np.argmin(dists))
            if dists[min_idx] < self.distance_threshold:
                candidates.append((min_idx, j))

        return candidates

    def detect_candidates_sc(
        self,
        dataset,
        n_frames: int,
    ) -> list[tuple[int, int, float]]:
        """Find loop closure candidates using Scan Context (v2).

        Builds the SC database incrementally and queries for each frame.

        Args:
            dataset: Indexable dataset returning ``(pointcloud, ...)``.
            n_frames: Number of frames to process.

        Returns:
            List of ``(i, j, sc_distance)`` candidate triples.
        """
        db = ScanContextDatabase(self.sc_num_rings, self.sc_num_sectors)
        candidates: list[tuple[int, int, float]] = []

        for j in range(n_frames):
            # `sc_query_timer` covers the full per-frame SC cost: descriptor
            # build, ring-key compute, query against db, and db.add for next
            # iteration. The dataset I/O (dataset[j][0]) sits outside because
            # it is owned by Stage 1 / data loader.
            pcl = dataset[j][0][:, :3]
            with self.sc_query_timer:
                sc = make_scan_context(
                    pcl, self.sc_num_rings, self.sc_num_sectors, self.sc_max_range
                )
                rk = compute_ring_key(sc)

                if j >= self.min_frame_gap and j % self.sc_query_stride == 0:
                    matches = db.query(
                        sc,
                        rk,
                        top_k=self.sc_top_k,
                        min_frame_gap=self.min_frame_gap,
                        current_frame=j,
                    )
                    n_matches = 0
                    for frame_idx, dist in matches:
                        if dist < self.sc_distance_threshold:
                            candidates.append((frame_idx, j, dist))
                            n_matches += 1
                            max_m = self.sc_max_matches_per_query
                            if max_m > 0 and n_matches >= max_m:
                                break

                db.add(sc, rk, j)

            if (j + 1) % 500 == 0:
                print(f"  SC: processed {j + 1}/{n_frames} frames, {len(candidates)} candidates")

        return candidates

    def _build_downsampled_pcd(self, points: np.ndarray) -> o3d.geometry.PointCloud:
        """Convert a raw (N, 3+) ndarray into a voxel-downsampled Open3D cloud.

        Uses :attr:`icp_downsample_voxel` set at construction time. This is
        the unit of work that the downsample cache amortizes across the
        ~3× redundant access per unique frame (8285 candidates ↔ ~1200
        unique frames on Seq 00 stride=1, giving ~6× downsample reduction).
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        return pcd.voxel_down_sample(voxel_size=self.icp_downsample_voxel)

    def _get_cached_downsampled_pcd(self, frame_id: int, dataset) -> o3d.geometry.PointCloud:
        """Return the downsampled cloud for ``frame_id``, building on miss.

        The cache is per-``detect()``-call, keyed on the frame index. It is
        reset at the start of :meth:`detect`. Cache misses read the raw
        point cloud from ``dataset[frame_id]`` and run voxel downsampling;
        hits return the stored Open3D cloud directly.
        """
        cached = self._downsample_cache.get(frame_id)
        if cached is not None:
            return cached
        raw = dataset[frame_id][0][:, :3]
        pcd = self._build_downsampled_pcd(raw)
        self._downsample_cache[frame_id] = pcd
        return pcd

    def verify_with_icp(
        self,
        source_points: np.ndarray,
        target_points: np.ndarray,
        initial_transform: np.ndarray,
        max_correspondence_distance: float = 2.0,
    ) -> tuple[np.ndarray, float] | None:
        """Verify a loop closure candidate using ICP (array-input API).

        Retained for callers that hold raw ndarrays and for unit tests.
        The production Stage 3 path in :meth:`detect` uses the cached,
        frame-id-keyed variant to avoid re-downsampling shared candidates.

        Args:
            source_points: (N, 3) source point cloud.
            target_points: (M, 3) target point cloud.
            initial_transform: 4x4 initial alignment guess.
            max_correspondence_distance: ICP max correspondence distance.

        Returns:
            (relative_pose, fitness) if fitness exceeds threshold, else None.
        """
        src_pcd = self._build_downsampled_pcd(source_points)
        tgt_pcd = self._build_downsampled_pcd(target_points)

        result = o3d.pipelines.registration.registration_icp(
            src_pcd,
            tgt_pcd,
            max_correspondence_distance,
            initial_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        )

        if result.fitness >= self.icp_fitness_threshold:
            return result.transformation, result.fitness
        return None

    def _verify_cached(
        self,
        src_frame_id: int,
        tgt_frame_id: int,
        dataset,
        initial_transform: np.ndarray,
        max_correspondence_distance: float = 2.0,
    ) -> tuple[np.ndarray, float] | None:
        """Cached ICP verification used by :meth:`detect`.

        Downsampled Open3D clouds are memoized per frame id within a single
        :meth:`detect` call, so when many SC candidates share endpoints the
        downsample work is paid once per frame instead of once per candidate.
        """
        src_pcd = self._get_cached_downsampled_pcd(src_frame_id, dataset)
        tgt_pcd = self._get_cached_downsampled_pcd(tgt_frame_id, dataset)

        result = o3d.pipelines.registration.registration_icp(
            src_pcd,
            tgt_pcd,
            max_correspondence_distance,
            initial_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        )

        if result.fitness >= self.icp_fitness_threshold:
            return result.transformation, result.fitness
        return None

    def detect(
        self,
        poses: list[np.ndarray],
        dataset=None,
    ) -> list[tuple[int, int, np.ndarray]]:
        """Run full loop closure detection.

        Args:
            poses: List of 4x4 estimated poses.
            dataset: Optional dataset for ICP verification and SC computation.
                If None, uses pose-derived relative transforms without ICP.

        Returns:
            List of (i, j, relative_pose_4x4) loop closure constraints.
        """
        # Reset sub-stage timers so each detect() call has a clean budget
        # (callers read .summary() right after this method returns).
        self.sc_query_timer = StageTimer("stage3_sc_query")
        self.icp_verify_timer = StageTimer("stage3_icp_verify")
        # Reset downsample cache — each detect() owns its own memoization
        # table. This keeps memory bounded across sequences and avoids
        # stale hits when poses change.
        self._downsample_cache = {}
        # Reset pre-ICP candidate snapshot; populated after dedupe below
        # so eval tooling can compare pre-ICP vs post-ICP TP in one pass.
        self.last_pre_icp_candidates: list[tuple[int, int]] = []

        # Gather candidates from selected mode(s)
        pairs: list[tuple[int, int]] = []

        if self.mode in ("v1", "both"):
            v1_cands = self.detect_candidates(poses)
            pairs.extend(v1_cands)
            if self.mode == "v1":
                print(f"  v1 distance candidates: {len(v1_cands)}")

        if self.mode in ("v2", "both") and dataset is not None:
            v2_cands = self.detect_candidates_sc(dataset, len(poses))
            pairs.extend([(i, j) for i, j, _ in v2_cands])
            print(f"  v2 SC candidates: {len(v2_cands)}")

        # Deduplicate
        seen: set[tuple[int, int]] = set()
        candidates: list[tuple[int, int]] = []
        for i, j in pairs:
            key = (min(i, j), max(i, j))
            if key not in seen:
                seen.add(key)
                candidates.append((i, j))

        self.last_pre_icp_candidates = list(candidates)

        if not candidates:
            return []

        closures = []
        for i, j in candidates:
            if dataset is not None:
                # ICP verification via cached downsampled clouds. The timer
                # covers cache lookup + (on miss) cloud build + Open3D ICP.
                # Cache hits return in microseconds, so the p50 of
                # `stage3_icp_verify` naturally splits into two modes:
                # "first touch for this frame" vs "cache hit".
                initial = np.linalg.inv(poses[i]) @ poses[j]
                with self.icp_verify_timer:
                    result = self._verify_cached(j, i, dataset, initial)
                if result is not None:
                    relative_pose, fitness = result
                    closures.append((i, j, relative_pose))
                    print(f"  Loop closure: {i} ↔ {j} (fitness={fitness:.3f})")
            else:
                # Without point clouds, use pose-derived relative transform
                relative_pose = np.linalg.inv(poses[i]) @ poses[j]
                closures.append((i, j, relative_pose))

        return closures
