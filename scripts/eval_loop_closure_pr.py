#!/usr/bin/env python3
"""SUP-02: Evaluate loop closure precision/recall on a KITTI sequence.

Compares v1 (distance) and v2 (Scan Context) detectors against
ground-truth loop pairs derived from GT poses.

Features:
- Single-point P/R for v1 and v2 (post-ICP)
- PR curve by sweeping SC distance threshold on pre-ICP candidates
- GNSS-denied robustness test
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from scripts.run_pipeline import load_config
from src.data.kitti_loader import KITTIDataset
from src.optimization.loop_closure import LoopClosureDetector


def _build_gt_loops(
    gt_poses: list[np.ndarray],
    distance_threshold: float = 5.0,
    min_frame_gap: int = 100,
) -> set[tuple[int, int]]:
    """Build GT loop closure pairs from ground-truth poses."""
    translations = np.array([p[:3, 3] for p in gt_poses])
    n = len(translations)
    pairs = set()
    for j in range(min_frame_gap, n):
        diffs = translations[:j - min_frame_gap + 1] - translations[j]
        dists = np.linalg.norm(diffs, axis=1)
        close = np.where(dists < distance_threshold)[0]
        for i in close:
            pairs.add((int(i), j))
    return pairs


def _count_tp(
    detected_pairs: set[tuple[int, int]],
    gt_pairs: set[tuple[int, int]],
    tolerance_frames: int = 20,
) -> int:
    """Count true positives with frame tolerance (vectorized)."""
    if not detected_pairs or not gt_pairs:
        return 0
    det = np.array(list(detected_pairs))  # (N, 2)
    gt = np.array(list(gt_pairs))  # (M, 2)
    # Broadcast: |det[i] - gt[j]| for each pair component
    # Process in chunks to avoid memory explosion
    chunk = 2000
    tp = 0
    for start in range(0, len(det), chunk):
        d = det[start:start + chunk]  # (C, 2)
        # (C, 1, 2) - (1, M, 2) -> (C, M, 2)
        diff = np.abs(d[:, None, :] - gt[None, :, :])
        # Both components within tolerance
        match = (diff[:, :, 0] <= tolerance_frames) & (diff[:, :, 1] <= tolerance_frames)
        tp += int(match.any(axis=1).sum())
    return tp


def _eval_detector(
    detector: LoopClosureDetector,
    poses: list[np.ndarray],
    dataset,
    gt_pairs: set[tuple[int, int]],
) -> dict:
    """Evaluate a detector (post-ICP) against GT pairs."""
    closures = detector.detect(poses, dataset=dataset)
    detected_pairs = {(i, j) for i, j, _ in closures}
    tp = _count_tp(detected_pairs, gt_pairs)
    n = len(detected_pairs)
    return {
        "n_detected": n,
        "n_gt": len(gt_pairs),
        "n_tp": tp,
        "precision": tp / n if n else 0.0,
        "recall": tp / len(gt_pairs) if gt_pairs else 0.0,
    }


def _eval_pr_curve_sc(
    detector: LoopClosureDetector,
    dataset,
    n_frames: int,
    gt_pairs: set[tuple[int, int]],
    thresholds: list[float],
) -> list[dict]:
    """Compute PR curve by sweeping SC distance threshold on pre-ICP candidates.

    Runs SC detection once with a lenient threshold, then filters at each
    sweep point.  This evaluates the SC descriptor quality without ICP cost.
    """
    orig_thresh = detector.sc_distance_threshold
    orig_max = detector.sc_max_matches_per_query
    detector.sc_distance_threshold = max(thresholds) + 0.05
    detector.sc_max_matches_per_query = 0

    cands = detector.detect_candidates_sc(dataset, n_frames)

    detector.sc_distance_threshold = orig_thresh
    detector.sc_max_matches_per_query = orig_max

    rows = []
    for t in sorted(thresholds):
        filtered = {(i, j) for i, j, d in cands if d < t}
        tp = _count_tp(filtered, gt_pairs)
        n = len(filtered)
        rows.append({
            "threshold": round(t, 2),
            "n_detected": n,
            "n_tp": tp,
            "precision": tp / n if n else 0.0,
            "recall": tp / len(gt_pairs) if gt_pairs else 0.0,
        })
    return rows


def _eval_gnss_denied(
    detector_v1: LoopClosureDetector,
    detector_v2: LoopClosureDetector,
    poses: list[np.ndarray],
    dataset,
    drift_per_frame: float = 0.05,
) -> dict:
    """Test GNSS-denied robustness with artificially drifted poses.

    Adds cumulative translation drift to simulate dead reckoning without
    GNSS correction.  v1 (pose-distance) should fail; v2 (appearance)
    should still detect closures.
    """
    n = len(poses)
    drifted = []
    for k in range(n):
        p = poses[k].copy()
        p[0, 3] += drift_per_frame * k
        p[1, 3] += drift_per_frame * k * 0.3
        drifted.append(p)

    v1_cands = detector_v1.detect_candidates(drifted)
    v2_cands = detector_v2.detect_candidates_sc(dataset, n)

    return {
        "v1_triggers": len(v1_cands),
        "v2_triggers": len(v2_cands),
        "drift_per_frame_m": drift_per_frame,
        "total_drift_m": drift_per_frame * n,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate loop closure P/R")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--sequence", default="00")
    parser.add_argument(
        "--pr-curve", action="store_true",
        help="Compute pre-ICP PR curve by sweeping SC threshold",
    )
    parser.add_argument(
        "--gnss-denied", action="store_true",
        help="Test GNSS-denied robustness (artificially drifted poses)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    seq = args.sequence

    dataset = KITTIDataset(root_path=config["data"]["kitti_root"], sequence=seq)
    Tr = dataset.calibration["Tr"] if dataset.calibration else np.eye(4)
    Tr_inv = np.linalg.inv(Tr)
    gt_cam = dataset.poses
    if gt_cam is None:
        print(f"No GT poses for seq {seq}")
        return
    gt_velo = [Tr_inv @ gt_cam[i] @ Tr for i in range(len(gt_cam))]

    # Run KISS-ICP to get estimated poses
    from src.odometry.kiss_icp_wrapper import KissICPOdometry

    kiss_cfg = config.get("kiss_icp", {})
    odom = KissICPOdometry(
        max_range=kiss_cfg.get("max_range", 100.0),
        min_range=kiss_cfg.get("min_range", 5.0),
        voxel_size=kiss_cfg.get("voxel_size", 1.0),
    )

    # Try to load cached poses
    from src.cache import LayeredCache

    cache = None
    cache_cfg = config.get("cache", {})
    if cache_cfg.get("enabled", False):
        cache = LayeredCache(root=cache_cfg.get("root", "cache/kitti"), sequence=seq)
    odom_cached = cache.load_odometry(config) if cache else None
    if odom_cached is not None:
        poses_arr, _ = odom_cached
        poses = [poses_arr[i] for i in range(poses_arr.shape[0])]
        print(f"Loaded {len(poses)} cached poses")
    else:
        poses = odom.run(dataset)

    lc_cfg = config.get("loop_closure", {})
    sc_cfg = lc_cfg.get("scan_context", {})
    min_gap = lc_cfg.get("min_frame_gap", 100)

    # Build GT loop pairs (aligned with detector min_frame_gap)
    gt_pairs = _build_gt_loops(gt_velo, distance_threshold=5.0, min_frame_gap=min_gap)
    print(f"GT loop pairs: {len(gt_pairs)} (min_frame_gap={min_gap})")

    rows = []

    # --- Evaluate v1 (distance) ---
    print("\n--- Evaluating v1 (distance) ---")
    det_v1 = LoopClosureDetector(
        distance_threshold=lc_cfg.get("distance_threshold", 15.0),
        min_frame_gap=min_gap,
        icp_fitness_threshold=lc_cfg.get("icp_fitness_threshold", 0.9),
        mode="v1",
    )
    result_v1 = _eval_detector(det_v1, poses, dataset, gt_pairs)
    print(f"  v1: detected={result_v1['n_detected']} TP={result_v1['n_tp']} "
          f"P={result_v1['precision']:.3f} R={result_v1['recall']:.3f}")
    rows.append({"system": "v1", **result_v1})

    # --- Evaluate v2 (Scan Context) ---
    print("\n--- Evaluating v2 (Scan Context) ---")
    det_v2 = LoopClosureDetector(
        distance_threshold=lc_cfg.get("distance_threshold", 15.0),
        min_frame_gap=min_gap,
        icp_fitness_threshold=lc_cfg.get("icp_fitness_threshold", 0.9),
        mode="v2",
        sc_num_rings=sc_cfg.get("num_rings", 20),
        sc_num_sectors=sc_cfg.get("num_sectors", 60),
        sc_max_range=sc_cfg.get("max_range", 80.0),
        sc_distance_threshold=sc_cfg.get("distance_threshold", 0.4),
        sc_top_k=sc_cfg.get("top_k", 10),
        sc_query_stride=sc_cfg.get("query_stride", 1),
        sc_max_matches_per_query=sc_cfg.get("max_matches_per_query", 0),
    )
    result_v2 = _eval_detector(det_v2, poses, dataset, gt_pairs)
    print(f"  v2: detected={result_v2['n_detected']} TP={result_v2['n_tp']} "
          f"P={result_v2['precision']:.3f} R={result_v2['recall']:.3f}")
    rows.append({"system": "v2", **result_v2})

    # Write summary CSV
    out_path = Path("results") / f"loop_closure_pr_seq{seq}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults written to {out_path}")

    # --- PR curve (optional) ---
    if args.pr_curve:
        print("\n--- Computing SC pre-ICP PR curve ---")
        thresholds = [x * 0.05 for x in range(2, 17)]  # 0.10 to 0.80
        pr_rows = _eval_pr_curve_sc(det_v2, dataset, len(poses), gt_pairs, thresholds)

        # Find recall @ precision=0.95
        r_at_p95 = 0.0
        for row in pr_rows:
            if row["precision"] >= 0.95:
                r_at_p95 = max(r_at_p95, row["recall"])

        print(f"  recall @ P=0.95 (pre-ICP): {r_at_p95:.4f}")
        for row in pr_rows:
            print(f"    SC<{row['threshold']:.2f}: "
                  f"det={row['n_detected']} TP={row['n_tp']} "
                  f"P={row['precision']:.3f} R={row['recall']:.3f}")

        pr_path = Path("results") / f"loop_closure_pr_curve_seq{seq}.csv"
        with pr_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(pr_rows[0].keys()))
            writer.writeheader()
            writer.writerows(pr_rows)
        print(f"  PR curve written to {pr_path}")

    # --- GNSS-denied test (optional) ---
    if args.gnss_denied:
        print("\n--- GNSS-denied robustness test ---")
        result_gd = _eval_gnss_denied(
            det_v1, det_v2, poses, dataset, drift_per_frame=0.05,
        )
        print(f"  Drift: {result_gd['drift_per_frame_m']} m/frame, "
              f"total {result_gd['total_drift_m']:.0f} m")
        print(f"  v1 triggers: {result_gd['v1_triggers']}")
        print(f"  v2 triggers: {result_gd['v2_triggers']}")
        v1_pass = result_gd["v1_triggers"] == 0
        v2_pass = result_gd["v2_triggers"] >= 1
        print(f"  Criterion 3: v1=0 {'PASS' if v1_pass else 'FAIL'}, "
              f"v2>=1 {'PASS' if v2_pass else 'FAIL'}")


if __name__ == "__main__":
    main()
