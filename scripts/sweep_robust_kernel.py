"""Sweep loop-closure robust kernels on a KITTI sequence.

Runs odometry + loop closure detection once (cache-hit on a cached sequence),
then re-optimizes the pose graph for each (kernel, scale) combination.
Reports APE RMSE and the number of effectively-rejected closures per kernel.

Usage::

    python scripts/sweep_robust_kernel.py --sequence 00
    python scripts/sweep_robust_kernel.py --sequence 00 \
        --kernels none,huber,cauchy,gm,dcs --scales 0.5,1.0,2.0,5.0
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import gtsam
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_pipeline import load_config  # noqa: E402
from src.benchmarks.timing import StageTimer  # noqa: E402
from src.cache import LayeredCache  # noqa: E402
from src.data.kitti_loader import KITTIDataset  # noqa: E402
from src.odometry.kiss_icp_wrapper import KissICPOdometry  # noqa: E402
from src.optimization.loop_closure import LoopClosureDetector  # noqa: E402
from src.optimization.pose_graph import PoseGraphOptimizer  # noqa: E402

# Reuse transform helper from run_pipeline to ensure the Velodyne ↔ camera
# frame convention matches production (§2 of refs/conventions.md).
from scripts.run_pipeline import transform_poses_to_camera_frame  # noqa: E402


def _ape_rmse(est_cam: list[np.ndarray], gt_cam: list[np.ndarray]) -> float:
    n = min(len(est_cam), len(gt_cam))
    sq = []
    for i in range(n):
        d = est_cam[i][:3, 3] - gt_cam[i][:3, 3]
        sq.append(float(d @ d))
    return float(np.sqrt(np.mean(sq))) if sq else float("nan")


def _weight_fn(kernel: str, scale: float, r: float) -> float:
    """Return the M-estimator weight w(r) for whitened residual magnitude r."""
    k = kernel.lower()
    if k in ("", "none") or r < 1e-12:
        return 1.0
    if k == "huber":
        return 1.0 if r <= scale else scale / r
    if k == "cauchy":
        return 1.0 / (1.0 + (r / scale) ** 2)
    if k in ("gm", "gemanmcclure"):
        # ρ(r) = (r²/2) / (1 + r²); w = (1 / (1+r²))² on unit scale.
        # Use GTSAM-style scaling: divide by scale first.
        u = r / scale
        return 1.0 / (1.0 + u * u) ** 2
    if k == "dcs":
        # DCS weight: w(r²) = min(1, 2φ / (φ + r²)). Residual in whitened space.
        r2 = r * r
        if r2 <= scale:
            return 1.0
        return (2.0 * scale) / (scale + r2)
    return 1.0


def _count_rejected_closures(
    graph: gtsam.NonlinearFactorGraph,
    result_values: gtsam.Values,
    lc_first_factor_idx: int,
    n_closures: int,
    kernel: str,
    scale: float,
    rejection_threshold: float = 0.1,
) -> int:
    """Count loop closure factors whose robust weight falls below a threshold.

    Iterates the last ``n_closures`` factors (the loop closure block) and
    evaluates their whitened residual magnitude at the optimized values.
    Weights computed analytically rather than pulled from GTSAM so the
    kernel + scale passed here must match what was baked into the graph.
    """
    rejected = 0
    for k in range(n_closures):
        factor_idx = lc_first_factor_idx + k
        try:
            factor = graph.at(factor_idx)
            e = factor.error(result_values)  # unwhitened squared-error measure
            r = float(np.sqrt(2.0 * max(e, 0.0)))
            w = _weight_fn(kernel, scale, r)
            if w < rejection_threshold:
                rejected += 1
        except Exception:
            continue
    return rejected


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep loop-closure robust kernels")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--sequence", default="00")
    parser.add_argument(
        "--kernels",
        default="none,huber,cauchy,gm,dcs",
        help="Comma-separated kernel names.",
    )
    parser.add_argument(
        "--scales",
        default="0.5,1.0,2.0,5.0",
        help="Comma-separated scale values. Ignored for kernel=none.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="CSV output path. Defaults to results/robust_sweep_seq{seq}.csv.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    seq = args.sequence

    dataset = KITTIDataset(root_path=cfg["data"]["kitti_root"], sequence=seq)
    if dataset.poses is None:
        raise SystemExit(f"No GT poses for seq {seq}")
    Tr = dataset.calibration["Tr"] if dataset.calibration else np.eye(4)
    Tr_inv = np.linalg.inv(Tr)
    gt_velo = [Tr_inv @ dataset.poses[i] @ Tr for i in range(len(dataset.poses))]

    cache_cfg = cfg.get("cache", {})
    cache = LayeredCache(cache_cfg.get("root", "cache/kitti"), sequence=seq) \
        if cache_cfg.get("enabled", False) else None

    # --- Stage 2: KISS-ICP poses (cache hit expected) ---
    poses: list[np.ndarray]
    odom_cached = cache.load_odometry(cfg) if cache else None
    if odom_cached is not None:
        poses_arr, _ = odom_cached
        poses = [poses_arr[i] for i in range(poses_arr.shape[0])]
        print(f"[cache] loaded {len(poses)} odometry poses")
    else:
        print("[miss] running KISS-ICP from scratch — this is slow (~12 min on Seq 00).")
        kiss_cfg = cfg.get("kiss_icp", {})
        odom = KissICPOdometry(
            max_range=kiss_cfg.get("max_range", 100.0),
            min_range=kiss_cfg.get("min_range", 5.0),
            voxel_size=kiss_cfg.get("voxel_size", 1.0),
        )
        poses = odom.run(dataset)
        if cache is not None:
            cache.save_odometry(np.asarray(poses), np.zeros(len(poses)), cfg)

    # --- Stage 3a: loop closure detection (ONE pass, shared across all kernels) ---
    print("\n[detect] running loop closure detection once...")
    lc_cfg = cfg.get("loop_closure", {})
    sc_cfg = lc_cfg.get("scan_context", {})
    detector = LoopClosureDetector(
        distance_threshold=lc_cfg.get("distance_threshold", 15.0),
        min_frame_gap=lc_cfg.get("min_frame_gap", 100),
        icp_fitness_threshold=lc_cfg.get("icp_fitness_threshold", 0.9),
        mode=lc_cfg.get("mode", "v2"),
        sc_num_rings=sc_cfg.get("num_rings", 20),
        sc_num_sectors=sc_cfg.get("num_sectors", 60),
        sc_max_range=sc_cfg.get("max_range", 80.0),
        sc_distance_threshold=sc_cfg.get("distance_threshold", 0.4),
        sc_top_k=sc_cfg.get("top_k", 25),
        sc_query_stride=sc_cfg.get("query_stride", 1),
        sc_max_matches_per_query=sc_cfg.get("max_matches_per_query", 0),
        icp_downsample_voxel=lc_cfg.get("icp_downsample_voxel", 1.0),
    )
    t0 = time.time()
    closures = detector.detect(poses, dataset=dataset)
    print(f"[detect] {len(closures)} loop closures in {time.time() - t0:.1f}s")

    # --- Stage 3b: sweep robust kernels, each re-optimizes from the shared closures ---
    gtsam_cfg = cfg.get("gtsam", {})
    kernels = [k.strip() for k in args.kernels.split(",") if k.strip()]
    scales = [float(s) for s in args.scales.split(",") if s.strip()]

    sweep_points: list[tuple[str, float]] = []
    for k in kernels:
        if k.lower() in ("none", ""):
            sweep_points.append(("none", 1.0))
        else:
            for s in scales:
                sweep_points.append((k, s))

    print(f"\n[sweep] {len(sweep_points)} (kernel, scale) points to run")
    rows = []
    baseline_ape = None
    for (kernel, scale) in sweep_points:
        tk = time.time()
        optimizer = PoseGraphOptimizer(
            odom_sigmas=gtsam_cfg.get("odom_sigmas"),
            prior_sigmas=gtsam_cfg.get("prior_sigmas"),
            robust_kernel=None if kernel == "none" else kernel,
            robust_scale=scale,
        )
        optimizer.build_graph(poses)
        lc_first_idx = optimizer.graph_size  # factor indices already inserted
        for i, j, rel in closures:
            optimizer.add_loop_closure(i, j, rel)
        opt_poses = optimizer.optimize()
        est_cam = transform_poses_to_camera_frame(opt_poses, Tr)
        gt_cam = [dataset.poses[i] for i in range(len(opt_poses))]
        ape = _ape_rmse(est_cam, gt_cam)

        rejected = 0
        if kernel != "none":
            rejected = _count_rejected_closures(
                optimizer.graph,
                optimizer.result_values,
                lc_first_idx,
                len(closures),
                kernel,
                scale,
            )

        delta = (ape - baseline_ape) if baseline_ape is not None else 0.0
        if kernel == "none":
            baseline_ape = ape
        pct = (100.0 * (ape - baseline_ape) / baseline_ape) if baseline_ape else 0.0
        print(
            f"  kernel={kernel:12s} scale={scale:4.1f} "
            f"APE={ape:7.4f} m  Δ={pct:+6.2f}%  rejected={rejected:4d}/{len(closures)}  "
            f"({time.time() - tk:.1f}s)"
        )
        rows.append({
            "kernel": kernel,
            "scale": scale,
            "n_closures": len(closures),
            "ape_rmse_m": round(ape, 6),
            "delta_pct_vs_baseline": round(pct, 3),
            "rejected_closures": rejected,
        })

    out_path = Path(args.output or f"results/robust_sweep_seq{seq}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[done] sweep written to {out_path}")


if __name__ == "__main__":
    main()
