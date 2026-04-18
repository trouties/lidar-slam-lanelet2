#!/usr/bin/env python3
"""SUP-01: Run baseline comparison across systems and KITTI sequences.

Produces three CSV tables:
  - benchmarks/accuracy_table.csv   (APE / RPE)
  - benchmarks/latency_table.csv    (per-stage timing)
  - benchmarks/robustness_gnss_denied.csv  (GNSS-denied drift)

And appends a record to benchmarks/benchmark_manifest.json.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from scripts.run_pipeline import load_config, run_pipeline_cached
from src.benchmarks import (
    BenchmarkManifest,
    load_poses_kitti_format,
    make_denial_window,
    score_denial_drift,
)
from src.benchmarks.gnss_denial import make_prior_indices
from src.cache import LayeredCache
from src.data.kitti_loader import KITTIDataset
from src.odometry.kiss_icp_wrapper import (
    KissICPOdometry,
    evaluate_odometry,
)
from src.optimization.loop_closure import LoopClosureDetector
from src.optimization.pose_graph import PoseGraphOptimizer

BENCHMARKS_DIR = Path("benchmarks")
DIAGNOSTICS_DIR = Path("results") / "diagnostics"


def _dual_eval(est_poses, gt_poses) -> tuple[dict, dict]:
    """Compute metrics under both first-frame and SE(3) Umeyama alignment."""
    n = min(len(est_poses), len(gt_poses))
    est = list(est_poses[:n])
    gt = list(gt_poses[:n])
    first = evaluate_odometry(est, gt, align="first")
    se3 = evaluate_odometry(est, gt, align="se3")
    return first, se3


def _audit_baseline_frame(system: str, sequence: str, bl_poses, gt_velo) -> None:
    """Write a small JSON diagnostic comparing baseline T0 + step-10 motion vs GT."""
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    out = DIAGNOSTICS_DIR / f"frame_audit_{system}_{sequence}.json"
    n = min(len(bl_poses), len(gt_velo), 11)
    if n < 2:
        return
    T0_bl = np.asarray(bl_poses[0])
    T10_bl = np.asarray(bl_poses[min(10, n - 1)])
    T0_gt = np.asarray(gt_velo[0])
    T10_gt = np.asarray(gt_velo[min(10, n - 1)])
    rel_bl = np.linalg.inv(T0_bl) @ T10_bl
    rel_gt = np.linalg.inv(T0_gt) @ T10_gt
    diag = {
        "system": system,
        "sequence": sequence,
        "T0_baseline_translation_m": [float(x) for x in T0_bl[:3, 3]],
        "T0_baseline_translation_norm_m": float(np.linalg.norm(T0_bl[:3, 3])),
        "step10_baseline_dtrans_m": [float(x) for x in rel_bl[:3, 3]],
        "step10_baseline_dtrans_norm_m": float(np.linalg.norm(rel_bl[:3, 3])),
        "step10_gt_dtrans_m": [float(x) for x in rel_gt[:3, 3]],
        "step10_gt_dtrans_norm_m": float(np.linalg.norm(rel_gt[:3, 3])),
        "step10_direction_dot_product": float(
            np.dot(rel_bl[:3, 3], rel_gt[:3, 3])
            / max(np.linalg.norm(rel_bl[:3, 3]) * np.linalg.norm(rel_gt[:3, 3]), 1e-9)
        ),
    }
    with out.open("w") as f:
        json.dump(diag, f, indent=2)
    print(f"  Wrote frame audit: {out}")

# Expected structure for external baseline results:
#   external/baselines/{system}/results/poses_{seq}.txt
#   OR  external/baselines/{system}/results/{seq}.txt
# Each file uses KITTI pose format: 12 floats per line (3x4 row-major).
# Systems: lio_sam, fast_lio2, hdl_graph_slam
EXTERNAL_DIR = Path("external/baselines")

# KITTI Odometry sequences with ground truth (00-10)
SEQUENCES_WITH_GT = [f"{i:02d}" for i in range(11)]


def _run_own_pipeline(
    config: dict,
    sequence: str,
    output_dir: Path,
    gnss_denial: bool = False,
    cache: LayeredCache | None = None,
) -> dict:
    """Run own pipeline and return metrics + timing."""
    if gnss_denial:
        # Run without cache to inject GNSS-denial priors
        summary = _run_own_gnss_denied(config, sequence, output_dir)
    else:
        summary = run_pipeline_cached(
            config=config,
            sequence=sequence,
            cache=cache,
            output_dir=output_dir,
            verbose=True,
        )
    return summary


def _run_own_gnss_denied(
    config: dict,
    sequence: str,
    output_dir: Path,
    cache: LayeredCache | None = None,
) -> dict:
    """Run own pipeline with GNSS-denial window."""
    cfg = {**config, "data": {**config.get("data", {}), "sequence": sequence}}
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = KITTIDataset(
        root_path=cfg["data"]["kitti_root"],
        sequence=sequence,
    )
    Tr = dataset.calibration["Tr"] if dataset.calibration else np.eye(4)
    Tr_inv = np.linalg.inv(Tr)
    gt_cam = dataset.poses
    gt_velo = (
            [Tr_inv @ gt_cam[i] @ Tr for i in range(len(gt_cam))]
            if gt_cam is not None
            else None
        )

    # Stage 2: odometry (reuse cache passed from caller)
    odom_cached = cache.load_odometry(cfg) if cache else None
    if odom_cached is not None:
        poses_arr, _ = odom_cached
        poses = [poses_arr[i] for i in range(poses_arr.shape[0])]
        print(f"  [cache hit] {len(poses)} odometry poses loaded for GNSS denial")
    else:
        kiss_cfg = cfg.get("kiss_icp", {})
        odom = KissICPOdometry(
            max_range=kiss_cfg.get("max_range", 100.0),
            min_range=kiss_cfg.get("min_range", 5.0),
            voxel_size=kiss_cfg.get("voxel_size", 1.0),
        )
        poses = odom.run(dataset)

    # Find denial window
    start, end = make_denial_window(poses, target_distance=150.0)
    prior_indices = make_prior_indices(len(poses), start, end, prior_stride=50)
    print(f"  GNSS denial window: frames {start}-{end}")

    # Stage 3: pose graph with sparse priors
    gtsam_cfg = cfg.get("gtsam", {})
    lc_cfg = cfg.get("loop_closure", {})
    detector = LoopClosureDetector(
        distance_threshold=lc_cfg.get("distance_threshold", 15.0),
        min_frame_gap=lc_cfg.get("min_frame_gap", 100),
        icp_fitness_threshold=lc_cfg.get("icp_fitness_threshold", 0.9),
    )
    closures = detector.detect(poses, dataset=dataset)
    optimizer = PoseGraphOptimizer(
        odom_sigmas=gtsam_cfg.get("odom_sigmas"),
        prior_sigmas=gtsam_cfg.get("prior_sigmas"),
    )
    optimizer.build_graph(poses, prior_indices=prior_indices, gt_poses=gt_velo)
    for i, j, rel_pose in closures:
        optimizer.add_loop_closure(i, j, rel_pose)
    optimized_poses = optimizer.optimize()

    # Score denial drift
    drift = score_denial_drift(optimized_poses, gt_velo, start, end) if gt_velo else {}

    return {
        "sequence": sequence,
        "gnss_denial": {
            "start": start,
            "end": end,
            "distance_m": drift.get("window_length_m", 0),
            **drift,
        },
        "metrics": {
            "optimized": {
                "ape_rmse": drift.get("ape_rmse", float("nan")),
                "ape_mean": drift.get("ape_mean", float("nan")),
            },
        },
        "loop_closures": len(closures),
    }


def _load_baseline_poses(system: str, sequence: str) -> list[np.ndarray] | None:
    """Attempt to load pre-computed baseline poses."""
    candidates = [
        EXTERNAL_DIR / system / "results" / f"poses_{sequence}.txt",
        EXTERNAL_DIR / system / "results" / f"{sequence}.txt",
    ]
    for p in candidates:
        if p.exists():
            return load_poses_kitti_format(p)
    return None


def run_comparison(
    config: dict,
    sequences: list[str],
    baselines: list[str],
) -> None:
    """Run full baseline comparison."""
    BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)

    accuracy_rows: list[dict] = []
    latency_rows: list[dict] = []
    gnss_rows: list[dict] = []

    cache_cfg = config.get("cache", {})
    cache = None
    if cache_cfg.get("enabled", False):
        pass  # cache created per-sequence below

    for seq in sequences:
        print(f"\n{'='*60}")
        print(f"Sequence {seq}")
        print(f"{'='*60}")

        # Load GT
        dataset = KITTIDataset(
            root_path=config["data"]["kitti_root"], sequence=seq
        )
        Tr = dataset.calibration["Tr"] if dataset.calibration else np.eye(4)
        Tr_inv = np.linalg.inv(Tr)
        gt_cam = dataset.poses
        gt_velo = (
            [Tr_inv @ gt_cam[i] @ Tr for i in range(len(gt_cam))]
            if gt_cam is not None
            else None
        )

        # --- Own pipeline (normal) ---
        out_dir = Path("results") / "baseline" / "own"
        cache = None
        if cache_cfg.get("enabled", False):
            cache = LayeredCache(root=cache_cfg.get("root", "cache/kitti"), sequence=seq)

        summary = _run_own_pipeline(config, seq, out_dir, cache=cache)

        # Own fused: reload cam-frame poses and compute SE(3)-aligned APE against
        # cam-frame GT (alignment is frame-invariant, so using either frame gives
        # the same aligned RMSE as long as both trajectories share it).
        own_fused_se3: dict | None = None
        fused_pose_path = out_dir / f"poses_fused_{seq}.txt"
        if fused_pose_path.exists() and dataset.poses is not None:
            try:
                fused_cam = load_poses_kitti_format(fused_pose_path)
                _, se3_m = _dual_eval(fused_cam, list(dataset.poses))
                own_fused_se3 = {
                    "ape_rmse": float(se3_m["ape"]["rmse"]),
                    "ape_mean": float(se3_m["ape"]["mean"]),
                    "rpe_rmse": float(se3_m["rpe"]["rmse"]),
                }
            except Exception as e:
                print(f"  [warn] Own SE3 re-eval failed: {e}")

        for stage_label in ["odometry", "optimized", "fused"]:
            m = summary.get("metrics", {}).get(stage_label, {})
            if m:
                se3_row = own_fused_se3 if stage_label == "fused" else None
                accuracy_rows.append({
                    "system": "own",
                    "sequence": seq,
                    "stage": stage_label,
                    "ape_rmse": f"{m['ape_rmse']:.4f}" if "ape_rmse" in m else "",
                    "ape_mean": f"{m['ape_mean']:.4f}" if "ape_mean" in m else "",
                    "rpe_rmse": f"{m['rpe_rmse']:.4f}" if "rpe_rmse" in m else "",
                    "ape_rmse_se3": f"{se3_row['ape_rmse']:.4f}" if se3_row else "N/A",
                    "ape_mean_se3": f"{se3_row['ape_mean']:.4f}" if se3_row else "N/A",
                    "rpe_rmse_se3": f"{se3_row['rpe_rmse']:.4f}" if se3_row else "N/A",
                    "source": "own_run",
                })

        # Timing
        for stage_name, timing in summary.get("timing", {}).items():
            latency_rows.append({
                "system": "own",
                "sequence": seq,
                "stage": stage_name,
                "p50_ms": f"{timing['p50']:.1f}",
                "p95_ms": f"{timing['p95']:.1f}",
                "max_ms": f"{timing['max']:.1f}",
                "mean_ms": f"{timing['mean']:.1f}",
            })

        # --- Own pipeline (GNSS denied) ---
        print(f"\n--- [{seq}] Own pipeline (GNSS denied) ---")
        denied_summary = _run_own_gnss_denied(config, seq, out_dir, cache=cache)
        d = denied_summary.get("gnss_denial", {})
        gnss_rows.append({
            "system": "own",
            "sequence": seq,
            "denial_start_frame": d.get("start", ""),
            "denial_end_frame": d.get("end", ""),
            "denial_distance_m": f"{d.get('distance_m', 0):.4f}" if d.get("distance_m") else "",
            "ape_in_window_m": f"{d.get('ape_mean', float('nan')):.4f}",
            "drift_per_meter": f"{d.get('drift_per_meter', float('nan')):.6f}",
        })

        # --- External baselines ---
        for bl in baselines:
            if bl == "own":
                continue
            print(f"\n--- [{seq}] Baseline: {bl} ---")
            bl_poses = _load_baseline_poses(bl, seq)
            if bl_poses is None:
                print(f"  No poses found for {bl} seq {seq} — marking FAIL")
                accuracy_rows.append({
                    "system": bl,
                    "sequence": seq,
                    "stage": "optimized",
                    "ape_rmse": "FAIL",
                    "ape_mean": "FAIL",
                    "rpe_rmse": "FAIL",
                    "ape_rmse_se3": "FAIL",
                    "ape_mean_se3": "FAIL",
                    "rpe_rmse_se3": "FAIL",
                    "source": "FAIL",
                })
                gnss_rows.append({
                    "system": bl,
                    "sequence": seq,
                    "denial_start_frame": "",
                    "denial_end_frame": "",
                    "denial_distance_m": "",
                    "ape_in_window_m": "FAIL",
                    "drift_per_meter": "FAIL",
                })
                continue

            # Evaluate baseline under both alignments
            if gt_velo is not None:
                _audit_baseline_frame(bl, seq, bl_poses, gt_velo)
                first_m, se3_m = _dual_eval(bl_poses, gt_velo)
                accuracy_rows.append({
                    "system": bl,
                    "sequence": seq,
                    "stage": "optimized",
                    "ape_rmse": f"{first_m['ape']['rmse']:.4f}",
                    "ape_mean": f"{first_m['ape']['mean']:.4f}",
                    "rpe_rmse": f"{first_m['rpe']['rmse']:.4f}",
                    "ape_rmse_se3": f"{se3_m['ape']['rmse']:.4f}",
                    "ape_mean_se3": f"{se3_m['ape']['mean']:.4f}",
                    "rpe_rmse_se3": f"{se3_m['rpe']['rmse']:.4f}",
                    "source": "baseline_run",
                })
                print(
                    f"  APE_first={first_m['ape']['rmse']:.3f} m "
                    f"APE_se3={se3_m['ape']['rmse']:.3f} m "
                    f"(Δ={first_m['ape']['rmse'] - se3_m['ape']['rmse']:+.3f})"
                )

    # Write CSVs
    _write_csv(BENCHMARKS_DIR / "accuracy_table.csv", accuracy_rows)
    _write_csv(BENCHMARKS_DIR / "latency_table.csv", latency_rows)
    _write_csv(BENCHMARKS_DIR / "robustness_gnss_denied.csv", gnss_rows)

    # Validate acceptance criteria
    violations = _validate_tables()
    if violations:
        print(f"\n  WARNING: {len(violations)} acceptance violation(s):")
        for v in violations:
            print(f"    - {v}")
    else:
        print("\n  All acceptance checks passed.")

    # Write manifest
    manifest = BenchmarkManifest()
    manifest.append(
        task="SUP-01",
        config=config,
        sequences=sequences,
        artifacts=[
            "benchmarks/accuracy_table.csv",
            "benchmarks/latency_table.csv",
            "benchmarks/robustness_gnss_denied.csv",
        ],
        metrics={
            "n_accuracy_rows": len(accuracy_rows),
            "n_latency_rows": len(latency_rows),
            "n_gnss_rows": len(gnss_rows),
            "baselines_attempted": baselines,
            "per_sequence": {
                seq: next(
                    (r["ape_rmse"] for r in accuracy_rows
                     if r["system"] == "own" and r["sequence"] == seq
                     and r["stage"] == "fused"),
                    None,
                )
                for seq in sequences
            },
            "validation_pass": len(violations) == 0,
            "violations": violations[:10],
        },
    )

    print(f"\n{'='*60}")
    print("SUP-01 tables written to benchmarks/")
    print(f"  accuracy_table.csv: {len(accuracy_rows)} rows")
    print(f"  latency_table.csv: {len(latency_rows)} rows")
    print(f"  robustness_gnss_denied.csv: {len(gnss_rows)} rows")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _validate_tables() -> list[str]:
    """Check acceptance criteria on the three output CSVs.

    Returns a list of violation descriptions (empty = all pass).
    """
    violations: list[str] = []
    for csv_name in ["accuracy_table.csv", "latency_table.csv", "robustness_gnss_denied.csv"]:
        path = BENCHMARKS_DIR / csv_name
        if not path.exists():
            violations.append(f"MISSING: {csv_name}")
            continue
        with path.open() as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                violations.append(f"NO HEADERS: {csv_name}")
                continue
            for row_idx, row in enumerate(reader, 1):
                for col, val in row.items():
                    if val == "N/A":
                        # Intentional placeholder (e.g., SE(3)-aligned metrics
                        # not computed for non-fused Own stages). Not a violation.
                        continue
                    if val is None or val == "":
                        violations.append(f"EMPTY: {csv_name} row {row_idx} col '{col}'")
                    elif val == "nan":
                        violations.append(f"NAN: {csv_name} row {row_idx} col '{col}'")
    return violations


def main() -> None:
    parser = argparse.ArgumentParser(description="SUP-01: Baseline Comparison")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Pipeline config file",
    )
    parser.add_argument(
        "--sequences", type=str, default="00,05",
        help="Comma-separated KITTI sequence ids",
    )
    parser.add_argument(
        "--baselines", type=str, default="own,lio_sam,fast_lio2,hdl_graph_slam",
        help="Comma-separated system names",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    sequences = [s.strip() for s in args.sequences.split(",")]
    baselines = [b.strip() for b in args.baselines.split(",")]

    run_comparison(config, sequences, baselines)


if __name__ == "__main__":
    main()
