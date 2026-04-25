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

# Shared default for Scan Context ring-key KNN fan-out. Matches
# ``configs/default.yaml::loop_closure.scan_context.top_k`` so that the main
# v2 evaluator and the SUP-02 Round-2 K-sweep operate at the same KNN width
# when a loaded config omits the key. Keep these in sync.
SC_DEFAULT_TOP_K = 25


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


def _count_gt_coverage(
    detected_pairs: set[tuple[int, int]],
    gt_pairs: set[tuple[int, int]],
    tolerance_frames: int = 20,
) -> int:
    """Count GT pairs covered by ≥1 detection within tolerance (GT-side).

    This is the proper numerator for recall: bounded above by ``len(gt_pairs)``,
    so the resulting ratio stays in [0, 1] regardless of how densely detections
    cluster around a single GT. ``_count_tp`` above counts the detection side
    and can exceed ``len(gt_pairs)``.

    Chunks detections along the D axis so the transient (chunk, M, 2) buffer
    stays under ~2 GB even at the permissive end of a PR sweep (n_detected
    can reach >100 k when SC threshold is lenient).
    """
    if not detected_pairs or not gt_pairs:
        return 0
    det = np.array(list(detected_pairs))  # (D, 2)
    gt = np.array(list(gt_pairs))  # (M, 2)
    covered = np.zeros(len(gt), dtype=bool)
    chunk = 2000
    for start in range(0, len(det), chunk):
        d = det[start:start + chunk]  # (C, 2)
        diff = np.abs(d[:, None, :] - gt[None, :, :])  # (C, M, 2)
        match = (diff[:, :, 0] <= tolerance_frames) & (diff[:, :, 1] <= tolerance_frames)
        covered |= match.any(axis=0)
    return int(covered.sum())


def _cluster_gt_pairs(
    gt_pairs: set[tuple[int, int]],
    cluster_gap: int = 50,
) -> list[frozenset[tuple[int, int]]]:
    """Cluster GT loop pairs into loop events via union-find.

    Two pairs (i1, j1) and (i2, j2) belong to the same event when both
    |i1 - i2| ≤ cluster_gap and |j1 - j2| ≤ cluster_gap (Chebyshev). This
    converts the dense per-pair GT into discrete revisit episodes, which
    is what place-recall counts against (Kim & Kim 2018 scoring style).

    The 50-frame default ≈ 5 s @ 10 Hz on KITTI, a typical dwell time
    inside a 5 m revisit ball.
    """
    if not gt_pairs:
        return []

    # Sort by i so a sliding window on the i-axis bounds neighbor search.
    arr = np.array(sorted(gt_pairs), dtype=np.int64)
    n = len(arr)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    left = 0
    for r in range(n):
        while arr[r, 0] - arr[left, 0] > cluster_gap:
            left += 1
        for left_idx in range(left, r):
            if abs(int(arr[r, 1]) - int(arr[left_idx, 1])) <= cluster_gap:
                union(left_idx, r)

    buckets: dict[int, list[tuple[int, int]]] = {}
    for k in range(n):
        root = find(k)
        buckets.setdefault(root, []).append((int(arr[k, 0]), int(arr[k, 1])))
    return [frozenset(v) for v in buckets.values()]


def _count_event_tp(
    detected_pairs: set[tuple[int, int]],
    gt_events: list[frozenset[tuple[int, int]]],
    tolerance_frames: int = 20,
) -> int:
    """Count loop events with at least one detected pair inside tolerance.

    An event is TP iff any detected pair lies within ±tolerance_frames
    (Chebyshev) of any GT pair in that event. This is the "place-recall"
    numerator — each event contributes at most one hit regardless of
    how many pairs the detector produced inside it.
    """
    if not detected_pairs or not gt_events:
        return 0
    det = np.array(list(detected_pairs))  # (D, 2)
    tp_events = 0
    chunk = 2000
    for event in gt_events:
        event_arr = np.array(list(event))  # (E, 2)
        hit = False
        for start in range(0, len(det), chunk):
            d = det[start:start + chunk]  # (C, 2)
            diff = np.abs(d[:, None, :] - event_arr[None, :, :])
            match = (diff[:, :, 0] <= tolerance_frames) & (diff[:, :, 1] <= tolerance_frames)
            if match.any():
                hit = True
                break
        if hit:
            tp_events += 1
    return tp_events


def _eval_detector(
    detector: LoopClosureDetector,
    poses: list[np.ndarray],
    dataset,
    gt_pairs: set[tuple[int, int]],
    gt_events: list[frozenset[tuple[int, int]]] | None = None,
    tolerance_frames: int = 20,
) -> dict:
    """Evaluate a detector against GT pairs.

    Reports both pre-ICP and post-ICP TP by reading
    :attr:`LoopClosureDetector.last_pre_icp_candidates` populated during
    ``detect()``. When ``gt_events`` is provided, also reports
    place-recall (one hit per event) and per-run ICP verify wall time.
    ``place_recall`` is the canonical loop closure recall (Kim & Kim 2018
    style); ``per_pair_recall`` is kept as a secondary indicator.
    ``recall`` aliases ``place_recall`` when events are provided, and
    falls back to ``per_pair_recall`` otherwise, for legacy readers.
    """
    closures = detector.detect(poses, dataset=dataset)
    post_pairs = {(i, j) for i, j, _ in closures}
    pre_pairs = set(detector.last_pre_icp_candidates)

    post_tp = _count_tp(post_pairs, gt_pairs, tolerance_frames=tolerance_frames)
    pre_tp = _count_tp(pre_pairs, gt_pairs, tolerance_frames=tolerance_frames)
    # GT-side coverage for per_pair_recall — bounded by n_gt.
    post_gt_cov = _count_gt_coverage(post_pairs, gt_pairs, tolerance_frames=tolerance_frames)
    n = len(post_pairs)
    n_gt = len(gt_pairs)

    icp_total_ms = detector.icp_verify_timer.summary().get("total_ms", 0.0)

    per_pair_recall = post_gt_cov / n_gt if n_gt else 0.0

    result: dict = {
        "n_detected": n,
        "n_gt": n_gt,
        "n_tp": post_tp,
        "precision": post_tp / n if n else 0.0,
        "per_pair_recall": per_pair_recall,
        "pre_icp_tp": pre_tp,
        "post_icp_tp": post_tp,
        "icp_time_s": icp_total_ms / 1000.0,
    }
    if gt_events is not None:
        place_tp = _count_event_tp(post_pairs, gt_events, tolerance_frames=tolerance_frames)
        place_recall = place_tp / len(gt_events) if gt_events else 0.0
        result["n_events"] = len(gt_events)
        result["place_tp"] = place_tp
        result["place_recall"] = place_recall
        # Canonical recall = place_recall when available.
        result["recall"] = place_recall
    else:
        # Legacy alias for backward-compatible readers.
        result["recall"] = per_pair_recall
    return result


def _eval_pr_curve_sc(
    detector: LoopClosureDetector,
    dataset,
    n_frames: int,
    gt_pairs: set[tuple[int, int]],
    thresholds: list[float],
    gt_events: list[frozenset[tuple[int, int]]] | None = None,
    tolerance_frames: int = 20,
) -> list[dict]:
    """Compute PR curve by sweeping SC distance threshold on pre-ICP candidates.

    Runs SC detection once with a lenient threshold, then filters at each
    sweep point. This evaluates the SC descriptor quality without ICP cost.
    When ``gt_events`` is provided, each row also carries the per-revisit
    (place) recall — the canonical place-recognition recall (Kim & Kim 2018).
    """
    orig_thresh = detector.sc_distance_threshold
    orig_max = detector.sc_max_matches_per_query
    detector.sc_distance_threshold = max(thresholds) + 0.05
    detector.sc_max_matches_per_query = 0

    cands = detector.detect_candidates_sc(dataset, n_frames)

    detector.sc_distance_threshold = orig_thresh
    detector.sc_max_matches_per_query = orig_max

    rows = []
    n_events = len(gt_events) if gt_events is not None else 0
    for t in sorted(thresholds):
        filtered = {(i, j) for i, j, d in cands if d < t}
        tp = _count_tp(filtered, gt_pairs, tolerance_frames=tolerance_frames)
        gt_cov = _count_gt_coverage(filtered, gt_pairs, tolerance_frames=tolerance_frames)
        n = len(filtered)
        per_pair_recall = gt_cov / len(gt_pairs) if gt_pairs else 0.0

        row: dict = {
            "threshold": round(t, 2),
            "n_detected": n,
            "n_tp": tp,
            "precision": tp / n if n else 0.0,
            "per_pair_recall": per_pair_recall,
        }
        if gt_events is not None:
            place_tp = _count_event_tp(filtered, gt_events, tolerance_frames=tolerance_frames)
            place_recall = place_tp / n_events if n_events else 0.0
            row["place_tp"] = place_tp
            row["n_events"] = n_events
            row["place_recall"] = place_recall
            # Canonical recall column tracks place_recall when events are provided.
            row["recall"] = place_recall
        else:
            row["recall"] = per_pair_recall
        rows.append(row)
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
    parser.add_argument(
        "--k-sweep",
        nargs="?",
        const="5,10,15",
        default=None,
        help=(
            "SUP-02 Round 2 max_matches_per_query sweep. "
            "Pass bare flag for default '5,10,15', or '--k-sweep 5,10' to customize. "
            "Writes results/sup02_round2_seq{seq}.csv with per-pair + place recall."
        ),
    )
    parser.add_argument(
        "--cluster-gap",
        type=int,
        default=50,
        help=(
            "Chebyshev gap (in frames) for clustering GT pairs into revisit "
            "events used by place_recall. 50 ≈ 5 s @10 Hz on KITTI, the "
            "typical dwell time inside a 5 m revisit ball."
        ),
    )
    parser.add_argument(
        "--tolerance-frames",
        type=int,
        default=20,
        help="Chebyshev tolerance (frames) when matching detected pairs against GT.",
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
    # Cluster GT pairs into revisit events for canonical place_recall. This
    # was previously only wired into --k-sweep; making it default ensures the
    # main CSV and the PR curve both report per-revisit recall.
    gt_events = _cluster_gt_pairs(gt_pairs, cluster_gap=args.cluster_gap)
    print(
        f"GT loop pairs: {len(gt_pairs)} (min_frame_gap={min_gap}); "
        f"revisit events: {len(gt_events)} (cluster_gap={args.cluster_gap})"
    )

    rows = []

    # --- Evaluate v1 (distance) ---
    print("\n--- Evaluating v1 (distance) ---")
    det_v1 = LoopClosureDetector(
        distance_threshold=lc_cfg.get("distance_threshold", 15.0),
        min_frame_gap=min_gap,
        icp_fitness_threshold=lc_cfg.get("icp_fitness_threshold", 0.9),
        mode="v1",
    )
    result_v1 = _eval_detector(
        det_v1, poses, dataset, gt_pairs,
        gt_events=gt_events, tolerance_frames=args.tolerance_frames,
    )
    print(
        f"  v1: detected={result_v1['n_detected']} TP={result_v1['n_tp']} "
        f"P={result_v1['precision']:.3f} "
        f"place_R={result_v1['place_recall']:.3f} "
        f"per_pair_R={result_v1['per_pair_recall']:.3f}"
    )
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
        sc_top_k=sc_cfg.get("top_k", SC_DEFAULT_TOP_K),
        sc_query_stride=sc_cfg.get("query_stride", 1),
        sc_max_matches_per_query=sc_cfg.get("max_matches_per_query", 0),
    )
    result_v2 = _eval_detector(
        det_v2, poses, dataset, gt_pairs,
        gt_events=gt_events, tolerance_frames=args.tolerance_frames,
    )
    print(
        f"  v2: detected={result_v2['n_detected']} TP={result_v2['n_tp']} "
        f"P={result_v2['precision']:.3f} "
        f"place_R={result_v2['place_recall']:.3f} "
        f"per_pair_R={result_v2['per_pair_recall']:.3f}"
    )
    rows.append({"system": "v2", **result_v2})

    # Write summary CSV with a stable header order. ``recall`` aliases
    # ``place_recall`` when revisit events are available, so legacy readers
    # that expect a ``recall`` column still see the canonical value.
    out_path = Path("results") / f"loop_closure_pr_seq{seq}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "system",
        "n_detected",
        "n_gt",
        "n_tp",
        "precision",
        "recall",
        "place_recall",
        "per_pair_recall",
        "n_events",
        "place_tp",
        "pre_icp_tp",
        "post_icp_tp",
        "icp_time_s",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults written to {out_path}")

    # --- PR curve (optional) ---
    if args.pr_curve:
        print("\n--- Computing SC pre-ICP PR curve ---")
        thresholds = [x * 0.05 for x in range(2, 17)]  # 0.10 to 0.80
        pr_rows = _eval_pr_curve_sc(
            det_v2, dataset, len(poses), gt_pairs, thresholds,
            gt_events=gt_events, tolerance_frames=args.tolerance_frames,
        )

        # SUP-02 acceptance point: recall @ precision=0.95. We report both
        # the canonical place (per-revisit) and the legacy per-pair values
        # so the threshold point can be read off either axis.
        r_place_at_p95 = 0.0
        r_pair_at_p95 = 0.0
        for row in pr_rows:
            if row["precision"] >= 0.95:
                r_place_at_p95 = max(r_place_at_p95, row.get("place_recall", 0.0))
                r_pair_at_p95 = max(r_pair_at_p95, row.get("per_pair_recall", 0.0))

        print(f"  place recall @ P=0.95 (pre-ICP): {r_place_at_p95:.4f}")
        print(f"  per-pair recall @ P=0.95 (pre-ICP): {r_pair_at_p95:.4f}")
        for row in pr_rows:
            print(
                f"    SC<{row['threshold']:.2f}: "
                f"det={row['n_detected']} TP={row['n_tp']} "
                f"P={row['precision']:.3f} "
                f"place_R={row.get('place_recall', row.get('recall', 0.0)):.3f} "
                f"per_pair_R={row['per_pair_recall']:.3f}"
            )

        pr_path = Path("results") / f"loop_closure_pr_curve_seq{seq}.csv"
        pr_fieldnames = [
            "threshold",
            "n_detected",
            "n_tp",
            "precision",
            "recall",
            "place_recall",
            "per_pair_recall",
            "place_tp",
            "n_events",
        ]
        with pr_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=pr_fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(pr_rows)
        print(f"  PR curve written to {pr_path}")

    # --- SUP-02 Round 2 K sweep (optional) ---
    if args.k_sweep is not None:
        print("\n--- SUP-02 Round 2 K sweep ---")
        try:
            k_values = [int(x.strip()) for x in args.k_sweep.split(",") if x.strip()]
        except ValueError as e:
            raise SystemExit(f"Invalid --k-sweep spec {args.k_sweep!r}: {e}")
        if not k_values:
            raise SystemExit("--k-sweep must specify at least one K value")

        # Reuse the canonical event set built earlier so all headline
        # numbers share the same clustering configuration.
        print(
            f"  GT events: {len(gt_events)} "
            f"(from {len(gt_pairs)} pairs, cluster_gap={args.cluster_gap} frames)"
        )

        round2_rows = []
        for k in k_values:
            print(f"\n  === K={k} ===")
            det_k = LoopClosureDetector(
                distance_threshold=lc_cfg.get("distance_threshold", 15.0),
                min_frame_gap=min_gap,
                icp_fitness_threshold=lc_cfg.get("icp_fitness_threshold", 0.9),
                mode="v2",
                sc_num_rings=sc_cfg.get("num_rings", 20),
                sc_num_sectors=sc_cfg.get("num_sectors", 60),
                sc_max_range=sc_cfg.get("max_range", 80.0),
                sc_distance_threshold=sc_cfg.get("distance_threshold", 0.4),
                sc_top_k=sc_cfg.get("top_k", SC_DEFAULT_TOP_K),
                sc_query_stride=sc_cfg.get("query_stride", 1),
                sc_max_matches_per_query=k,
                icp_downsample_voxel=lc_cfg.get("icp_downsample_voxel", 1.0),
            )
            r = _eval_detector(
                det_k, poses, dataset, gt_pairs,
                gt_events=gt_events, tolerance_frames=args.tolerance_frames,
            )
            round2_rows.append({
                "K": k,
                "pre_icp_tp": r["pre_icp_tp"],
                "post_icp_tp": r["post_icp_tp"],
                "per_pair_recall": round(r["per_pair_recall"], 4),
                "place_recall": round(r["place_recall"], 4),
                "icp_time_s": round(r["icp_time_s"], 1),
            })
            print(
                f"    pre_icp_tp={r['pre_icp_tp']} post_icp_tp={r['post_icp_tp']} "
                f"per_pair_R={r['per_pair_recall']:.4f} "
                f"place_R={r['place_recall']:.4f} "
                f"ICP={r['icp_time_s']:.0f}s"
            )

        r2_path = Path("results") / f"sup02_round2_seq{seq}.csv"
        with r2_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(round2_rows[0].keys()))
            writer.writeheader()
            writer.writerows(round2_rows)
        print(f"\nRound 2 table written to {r2_path}")

        print("\n| K | pre_icp_tp | post_icp_tp | per_pair_R | place_R | ICP time (s) |")
        print("|---|---|---|---|---|---|")
        for row in round2_rows:
            print(
                f"| {row['K']} | {row['pre_icp_tp']} | {row['post_icp_tp']} | "
                f"{row['per_pair_recall']:.4f} | {row['place_recall']:.4f} | "
                f"{row['icp_time_s']:.0f} |"
            )

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
