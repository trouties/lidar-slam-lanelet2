"""Loop closure detection.

Detects revisited places with Scan Context appearance matching, then
verifies each candidate with ICP point cloud registration and optionally
refines the accepted ones.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import open3d as o3d

from src.benchmarks.timing import StageTimer
from src.optimization.scan_context import (
    ScanContextDatabase,
    compute_ring_key,
    make_scan_context,
    sc_relative_yaw,
)


class LoopClosureDetector:
    """Scan Context candidates, verified with Open3D ICP."""

    def __init__(
        self,
        min_frame_gap: int = 100,
        icp_fitness_threshold: float = 0.3,
        sc_num_rings: int = 20,
        sc_num_sectors: int = 60,
        sc_max_range: float = 80.0,
        sc_distance_threshold: float = 0.4,
        sc_top_k: int = 10,
        sc_query_stride: int = 1,
        sc_max_matches_per_query: int = 0,
        icp_downsample_voxel: float = 1.0,
        refine_stages: list[tuple[float, float]] | None = None,
        refine_max_iteration: int = 50,
        retry_from_scan_context: bool = False,
        max_drift_ratio: float | None = None,
    ) -> None:
        """Initialize detector.

        Args:
            min_frame_gap: Minimum frame index gap to avoid detecting neighbors.
            icp_fitness_threshold: Minimum ICP fitness to accept a match.
            sc_num_rings: Number of radial bins for Scan Context.
            sc_num_sectors: Number of angular bins for Scan Context.
            sc_max_range: Maximum range for Scan Context descriptor.
            sc_distance_threshold: Maximum SC distance to consider a candidate.
            sc_top_k: Number of ring-key candidates to re-rank per query.
            sc_query_stride: Query every N-th frame (1 = every frame).
            sc_max_matches_per_query: Max matches per query frame (0 = unlimited).
            icp_downsample_voxel: Voxel size (m) used to downsample candidate
                point clouds before the verification ICP. Finer voxels trade
                speed for registration accuracy. Sparser sensors (e.g. the
                32-beam nuScenes LiDAR) typically want a smaller value.
            refine_stages: ``[(voxel_m, max_correspondence_m), ...]`` for a
                coarse-to-fine point-to-plane ICP that replaces the relative
                pose of every accepted candidate. ``None`` keeps the
                verification result.
            refine_max_iteration: ICP iterations per refinement stage.
            retry_from_scan_context: When the verification fails from the
                odometry prior, refine again from the Scan Context alignment
                (descriptor yaw, zero translation) and verify from there.
                The prior is only as good as the odometry: after a few
                kilometres it can be metres off at the revisit, beyond the
                reach of the verification ICP, while Scan Context finds the
                place without it. Needs ``refine_stages``.
            max_drift_ratio: Reject a loop closure whose relative pose moves
                the second scan by more than this fraction of the path driven
                between the two scans, relative to the odometry. Odometry
                drifts by about half a percent of the distance travelled; a
                loop closure that needs several percent is a false match.
                ``None`` disables the check.
        """
        self.min_frame_gap = min_frame_gap
        self.icp_fitness_threshold = icp_fitness_threshold
        self.sc_num_rings = sc_num_rings
        self.sc_num_sectors = sc_num_sectors
        self.sc_max_range = sc_max_range
        self.sc_distance_threshold = sc_distance_threshold
        self.sc_top_k = sc_top_k
        self.sc_query_stride = sc_query_stride
        self.sc_max_matches_per_query = sc_max_matches_per_query
        self.icp_downsample_voxel = icp_downsample_voxel
        self.refine_stages = [tuple(st) for st in refine_stages] if refine_stages else None
        self.refine_max_iteration = refine_max_iteration
        if retry_from_scan_context and not self.refine_stages:
            raise ValueError("retry_from_scan_context needs refine_stages")
        self.retry_from_scan_context = retry_from_scan_context
        self.max_drift_ratio = max_drift_ratio
        # Scan Context descriptors by frame, kept by detect_candidates_sc for the retry.
        self._sc_by_frame: dict[int, np.ndarray] = {}
        # Sub-stage timers populated during detect(); read summary() after.
        # Reset on every detect() call so reusing a detector across sequences
        # does not accumulate stale measurements.
        self.sc_query_timer = StageTimer("stage3_sc_query")
        self.icp_verify_timer = StageTimer("stage3_icp_verify")
        self.icp_refine_timer = StageTimer("stage3_icp_refine")
        self.icp_retry_timer = StageTimer("stage3_icp_retry")
        # Per-detect() downsample cache: frame_id -> downsampled Open3D cloud.
        # Reset at the start of detect(). See _get_cached_downsampled_pcd.
        self._downsample_cache: dict[int, o3d.geometry.PointCloud] = {}
        # Refinement clouds (all stage voxel sizes, with normals), LRU by frame.
        self._refine_cache: OrderedDict[int, dict[float, o3d.geometry.PointCloud]] = OrderedDict()
        self._refine_cache_frames = 600

    @classmethod
    def from_config(cls, lc_cfg: dict) -> LoopClosureDetector:
        """Build a detector from the ``loop_closure`` config section."""
        sc_cfg = lc_cfg.get("scan_context", {})
        return cls(
            min_frame_gap=lc_cfg.get("min_frame_gap", 100),
            icp_fitness_threshold=lc_cfg.get("icp_fitness_threshold", 0.9),
            sc_num_rings=sc_cfg.get("num_rings", 20),
            sc_num_sectors=sc_cfg.get("num_sectors", 60),
            sc_max_range=sc_cfg.get("max_range", 80.0),
            sc_distance_threshold=sc_cfg.get("distance_threshold", 0.4),
            sc_top_k=sc_cfg.get("top_k", 10),
            sc_query_stride=sc_cfg.get("query_stride", 1),
            sc_max_matches_per_query=sc_cfg.get("max_matches_per_query", 0),
            icp_downsample_voxel=lc_cfg.get("icp_downsample_voxel", 1.0),
            refine_stages=(lc_cfg.get("refine") or {}).get("stages"),
            refine_max_iteration=(lc_cfg.get("refine") or {}).get("max_iteration", 50),
            retry_from_scan_context=lc_cfg.get("retry_from_scan_context", False),
            max_drift_ratio=lc_cfg.get("max_drift_ratio"),
        )

    def detect_candidates_sc(
        self,
        dataset,
        n_frames: int,
    ) -> list[tuple[int, int, float]]:
        """Find loop closure candidates using Scan Context.

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
                if self.retry_from_scan_context:
                    self._sc_by_frame[j] = sc

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
        result = self.register(
            src_frame_id, tgt_frame_id, dataset, initial_transform, max_correspondence_distance
        )
        if result.fitness >= self.icp_fitness_threshold:
            return result.transformation, result.fitness
        return None

    def register(
        self,
        src_frame_id: int,
        tgt_frame_id: int,
        dataset,
        initial_transform: np.ndarray,
        max_correspondence_distance: float = 2.0,
    ) -> o3d.pipelines.registration.RegistrationResult:
        """Point-to-point ICP of frame ``src`` onto frame ``tgt``, without the fitness check.

        The returned transformation maps source-frame points into the target
        frame, i.e. it estimates ``inv(T_tgt) @ T_src``.
        """
        src_pcd = self._get_cached_downsampled_pcd(src_frame_id, dataset)
        tgt_pcd = self._get_cached_downsampled_pcd(tgt_frame_id, dataset)
        return o3d.pipelines.registration.registration_icp(
            src_pcd,
            tgt_pcd,
            max_correspondence_distance,
            initial_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        )

    def _refine_cloud(self, frame_id: int, dataset, voxel: float) -> o3d.geometry.PointCloud:
        levels = self._refine_cache.get(frame_id)
        if levels is None:
            raw = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(dataset[frame_id][0][:, :3]))
            levels = {}
            for v in sorted({v for v, _ in self.refine_stages}):
                pcd = raw.voxel_down_sample(v)
                pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(3 * v, 30))
                levels[v] = pcd
            if len(self._refine_cache) >= self._refine_cache_frames:
                self._refine_cache.popitem(last=False)
            self._refine_cache[frame_id] = levels
        else:
            self._refine_cache.move_to_end(frame_id)
        return levels[voxel]

    def refine(
        self,
        src_frame_id: int,
        tgt_frame_id: int,
        dataset,
        initial_transform: np.ndarray,
    ) -> o3d.pipelines.registration.RegistrationResult:
        """Coarse-to-fine point-to-plane ICP; returns the result of the last stage.

        Starts again from ``initial_transform`` (the odometry prior), not from
        the verification result. Point-to-point on 1 m voxel centroids leaves
        the alignment soft in directions where the scene is mostly planar
        (road surface, facades), because centroids of differently sampled
        planes do not coincide; the plane normals remove that freedom. The
        coarse stage widens the basin of convergence when odometry has drifted
        by several metres at the revisit.
        """
        reg = o3d.pipelines.registration
        est = reg.TransformationEstimationPointToPlane()
        crit = reg.ICPConvergenceCriteria(max_iteration=self.refine_max_iteration)
        T = np.asarray(initial_transform)
        result = None
        for voxel, corr in self.refine_stages:
            src = self._refine_cloud(src_frame_id, dataset, voxel)
            tgt = self._refine_cloud(tgt_frame_id, dataset, voxel)
            result = reg.registration_icp(src, tgt, corr, T, est, crit)
            T = np.asarray(result.transformation)
        return result

    def scan_context_start(self, i: int, j: int) -> np.ndarray:
        """Relative pose of scan j in scan i from Scan Context alone.

        Rotation about z by the descriptor's best column shift, no
        translation: a Scan Context match means the two scans were taken at
        about the same place.
        """
        yaw = sc_relative_yaw(self._sc_by_frame[j], self._sc_by_frame[i])
        T = np.eye(4)
        T[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
        return T

    def retry(self, i: int, j: int, dataset) -> tuple[np.ndarray, float] | None:
        """Refine from the Scan Context alignment, then run the usual verification.

        Returns ``(relative_pose, fitness)`` of the refined registration if
        the verification ICP, started from it, passes the fitness threshold.
        """
        start = self.scan_context_start(i, j)
        refined = np.asarray(self.refine(j, i, dataset, start).transformation)
        result = self._verify_cached(j, i, dataset, refined)
        if result is None:
            return None
        return refined, result[1]

    def plausible(self, prior: np.ndarray, measured: np.ndarray, path_length: float) -> bool:
        """Whether a loop closure asks for no more drift than ``max_drift_ratio`` allows.

        The drift is how far the measured relative pose moves the second
        scan away from where the odometry puts it.
        """
        if self.max_drift_ratio is None:
            return True
        drift = float(np.linalg.norm(measured[:3, 3] - prior[:3, 3]))
        return drift <= self.max_drift_ratio * path_length

    def detect(
        self,
        poses: list[np.ndarray],
        dataset,
    ) -> list[tuple[int, int, np.ndarray]]:
        """Run full loop closure detection.

        Args:
            poses: List of 4x4 estimated poses.
            dataset: Indexable dataset returning ``(pointcloud, ...)``, for the
                Scan Context descriptors and the ICP verification.

        Returns:
            List of (i, j, relative_pose_4x4) loop closure constraints.
        """
        # Reset sub-stage timers so each detect() call has a clean budget
        # (callers read .summary() right after this method returns).
        self.sc_query_timer = StageTimer("stage3_sc_query")
        self.icp_verify_timer = StageTimer("stage3_icp_verify")
        self.icp_refine_timer = StageTimer("stage3_icp_refine")
        self.icp_retry_timer = StageTimer("stage3_icp_retry")
        self._refine_cache = OrderedDict()
        self._sc_by_frame = {}
        # Reset downsample cache — each detect() owns its own memoization
        # table. This keeps memory bounded across sequences and avoids
        # stale hits when poses change.
        self._downsample_cache = {}
        # Reset pre-ICP candidate snapshot; populated after dedupe below
        # so eval tooling can compare pre-ICP vs post-ICP TP in one pass.
        self.last_pre_icp_candidates: list[tuple[int, int]] = []

        sc_cands = self.detect_candidates_sc(dataset, len(poses))
        print(f"  SC candidates: {len(sc_cands)}")

        # Deduplicate
        seen: set[tuple[int, int]] = set()
        candidates: list[tuple[int, int]] = []
        for i, j, _ in sc_cands:
            key = (min(i, j), max(i, j))
            if key not in seen:
                seen.add(key)
                candidates.append((i, j))

        self.last_pre_icp_candidates = list(candidates)

        if not candidates:
            return []

        trans = np.array([p[:3, 3] for p in poses])
        path = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(trans, axis=0), axis=1))])

        closures = []
        for i, j in candidates:
            # ICP verification via cached downsampled clouds. The timer
            # covers cache lookup + (on miss) cloud build + Open3D ICP.
            # Cache hits return in microseconds, so the p50 of
            # `stage3_icp_verify` naturally splits into two modes:
            # "first touch for this frame" vs "cache hit".
            initial = np.linalg.inv(poses[i]) @ poses[j]
            with self.icp_verify_timer:
                result = self._verify_cached(j, i, dataset, initial)
            relative_pose = None
            if result is not None:
                relative_pose, fitness = result
                if self.refine_stages:
                    with self.icp_refine_timer:
                        refined = self.refine(j, i, dataset, initial)
                    relative_pose = np.asarray(refined.transformation)
            elif self.retry_from_scan_context and j in self._sc_by_frame:
                with self.icp_retry_timer:
                    retried = self.retry(i, j, dataset)
                if retried is not None:
                    relative_pose, fitness = retried
            if relative_pose is None:
                continue
            if not self.plausible(initial, relative_pose, path[j] - path[i]):
                print(f"  Rejected {i} ↔ {j}: implausible drift (fitness={fitness:.3f})")
                continue
            closures.append((i, j, relative_pose))
            print(f"  Loop closure: {i} ↔ {j} (fitness={fitness:.3f})")

        return closures
