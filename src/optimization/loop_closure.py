"""Loop closure detection.

Detects revisited places using either distance-based candidate search (v1)
or Scan Context appearance matching (v2), then verifies with ICP point
cloud registration.
"""

from __future__ import annotations

import numpy as np
import open3d as o3d

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
            pcl = dataset[j][0][:, :3]
            sc = make_scan_context(pcl, self.sc_num_rings, self.sc_num_sectors, self.sc_max_range)
            rk = compute_ring_key(sc)

            if j >= self.min_frame_gap and j % self.sc_query_stride == 0:
                matches = db.query(
                    sc, rk,
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

    def verify_with_icp(
        self,
        source_points: np.ndarray,
        target_points: np.ndarray,
        initial_transform: np.ndarray,
        max_correspondence_distance: float = 2.0,
    ) -> tuple[np.ndarray, float] | None:
        """Verify a loop closure candidate using ICP.

        Args:
            source_points: (N, 3) source point cloud.
            target_points: (M, 3) target point cloud.
            initial_transform: 4x4 initial alignment guess.
            max_correspondence_distance: ICP max correspondence distance.

        Returns:
            (relative_pose, fitness) if fitness exceeds threshold, else None.
        """
        src_pcd = o3d.geometry.PointCloud()
        src_pcd.points = o3d.utility.Vector3dVector(source_points[:, :3])

        tgt_pcd = o3d.geometry.PointCloud()
        tgt_pcd.points = o3d.utility.Vector3dVector(target_points[:, :3])

        # Downsample for speed
        src_pcd = src_pcd.voxel_down_sample(voxel_size=1.0)
        tgt_pcd = tgt_pcd.voxel_down_sample(voxel_size=1.0)

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

        if not candidates:
            return []

        closures = []
        for i, j in candidates:
            if dataset is not None:
                # ICP verification
                source_cloud = dataset[j][0][:, :3]
                target_cloud = dataset[i][0][:, :3]
                initial = np.linalg.inv(poses[i]) @ poses[j]

                result = self.verify_with_icp(source_cloud, target_cloud, initial)
                if result is not None:
                    relative_pose, fitness = result
                    closures.append((i, j, relative_pose))
                    print(f"  Loop closure: {i} ↔ {j} (fitness={fitness:.3f})")
            else:
                # Without point clouds, use pose-derived relative transform
                relative_pose = np.linalg.inv(poses[i]) @ poses[j]
                closures.append((i, j, relative_pose))

        return closures
