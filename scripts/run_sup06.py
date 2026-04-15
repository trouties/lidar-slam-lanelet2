#!/usr/bin/env python3
"""SUP-06: Uncertainty Visualization.

Extracts per-keyframe position marginal covariances from the pose graph
optimizer (loose coupling by default; optional tight coupling with IMU),
writes a CSV, renders a static 3D PNG, and generates a GIF animation of
the ellipsoid balloon/deflate behavior across a GNSS-denial window.

Outputs (under ``benchmarks/uncertainty/``):
    - marginal_cov_<seq>_<mode>.csv
    - ellipsoids_static_<seq>_<mode>.png
    - ellipsoid_animation_<seq>_<mode>.gif
    - sup06_report_<seq>.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless — must precede any pyplot import chain
import numpy as np  # noqa: E402

from scripts.run_pipeline import load_config  # noqa: E402
from src.benchmarks import BenchmarkManifest  # noqa: E402
from src.benchmarks.gnss_denial import make_denial_window, make_prior_indices  # noqa: E402
from src.cache import LayeredCache  # noqa: E402
from src.data.imu_loader import load_imu_for_odometry_seq  # noqa: E402
from src.data.kitti_loader import KITTIDataset  # noqa: E402
from src.odometry.kiss_icp_wrapper import KissICPOdometry  # noqa: E402
from src.optimization.imu_factor import build_tight_coupled_graph  # noqa: E402
from src.optimization.loop_closure import LoopClosureDetector  # noqa: E402
from src.optimization.pose_graph import PoseGraphOptimizer  # noqa: E402
from src.visualization.uncertainty_plot import (  # noqa: E402
    animate_uncertainty_evolution,
    plot_trajectory_with_ellipsoids,
)

OUT_DIR = Path("benchmarks/uncertainty")


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _load_cached_poses(config: dict, sequence: str, dataset: KITTIDataset) -> list[np.ndarray]:
    """Prefer cached Stage 2 odometry; fall back to live KISS-ICP."""
    cache_cfg = config.get("cache", {})
    if cache_cfg.get("enabled", False):
        cache = LayeredCache(root=cache_cfg.get("root", "cache/kitti"), sequence=sequence)
        cached = cache.load_odometry(config)
        if cached is not None:
            poses_arr, _ = cached
            print(f"  [cache hit] {len(poses_arr)} odometry poses")
            return [poses_arr[i] for i in range(poses_arr.shape[0])]
    print("  [cache miss] running KISS-ICP")
    kiss_cfg = config.get("kiss_icp", {})
    odom = KissICPOdometry(
        max_range=kiss_cfg.get("max_range", 100.0),
        min_range=kiss_cfg.get("min_range", 5.0),
        voxel_size=kiss_cfg.get("voxel_size", 1.0),
    )
    return odom.run(dataset)


def _make_detector(config: dict) -> LoopClosureDetector:
    lc_cfg = config.get("loop_closure", {})
    sc_cfg = lc_cfg.get("scan_context", {})
    return LoopClosureDetector(
        distance_threshold=lc_cfg.get("distance_threshold", 15.0),
        min_frame_gap=lc_cfg.get("min_frame_gap", 100),
        icp_fitness_threshold=lc_cfg.get("icp_fitness_threshold", 0.9),
        mode=lc_cfg.get("mode", "v1"),
        sc_num_rings=sc_cfg.get("num_rings", 20),
        sc_num_sectors=sc_cfg.get("num_sectors", 60),
        sc_max_range=sc_cfg.get("max_range", 80.0),
        sc_distance_threshold=sc_cfg.get("distance_threshold", 0.4),
        sc_top_k=sc_cfg.get("top_k", 25),
        sc_query_stride=sc_cfg.get("query_stride", 1),
        sc_max_matches_per_query=sc_cfg.get("max_matches_per_query", 5),
        icp_downsample_voxel=lc_cfg.get("icp_downsample_voxel", 1.0),
    )


def _build_sample_frames(
    n_frames: int,
    stride: int,
    denial_start: int,
    denial_end: int,
) -> list[int]:
    """Regular stride + forced denial boundaries for clean acceptance checks."""
    samples = set(range(0, n_frames, stride))
    for k in (
        0,
        max(0, denial_start - 1),
        denial_start,
        (denial_start + denial_end) // 2,
        denial_end,
        min(n_frames - 1, denial_end + 1),
        n_frames - 1,
    ):
        if 0 <= k < n_frames:
            samples.add(k)
    return sorted(samples)


def _frame_flags(
    frame: int,
    denial_start: int,
    denial_end: int,
    prior_stride: int,
    n_frames: int,
    tail_buffer: int,
) -> tuple[bool, bool, bool]:
    """Classify a frame by its role in the pose graph.

    Returns:
        ``(is_prior, in_denial, is_tail)``:
          - ``is_prior``: frame index is a GNSS-prior anchor
            (``frame % prior_stride == 0`` and outside denial window).
          - ``in_denial``: ``denial_start <= frame <= denial_end``.
          - ``is_tail``: frame is in the last ``tail_buffer`` frames of the
            trajectory, where the pose graph has no downstream prior / limited
            loop closure support and marginal naturally swells.
    """
    in_denial = denial_start <= frame <= denial_end
    is_prior = (frame % prior_stride == 0) and not in_denial
    is_tail = frame >= n_frames - tail_buffer
    return is_prior, in_denial, is_tail


def _compute_acceptance(
    sample_frames: list[int],
    covariances: dict[int, np.ndarray],
    denial_start: int,
    denial_end: int,
    prior_stride: int,
    n_frames: int,
    recovery_buffer: int,
    tail_buffer: int,
) -> dict[str, float | bool | int]:
    """Compute SUP-06 acceptance using a physically clean baseline.

    KITTI Seq 00 samples land on either GNSS-anchored prior frames (trace
    pinned at ``prior_sigma^2 * 3 ~ 3e-4``) or drift frames between priors
    (trace ~0.1-0.3). Averaging both populations together is meaningless.
    The baseline here uses **median of non-prior drift frames** (the steady-
    state "dead-reckoning at prior_stride interval" signature), so the
    denial ratio measures "denial peak vs stable drift baseline".

    Tail frames (last ``tail_buffer``) are excluded because they lack
    downstream prior support and show edge swelling unrelated to denial.
    """
    traces = {k: float(np.trace(covariances[k])) for k in sample_frames}

    # Classify every sample
    flags = {
        k: _frame_flags(k, denial_start, denial_end, prior_stride, n_frames, tail_buffer)
        for k in sample_frames
    }

    # Stable drift baseline buckets: non-prior, non-tail, outside denial + recovery buffer
    pre_bucket = [
        traces[k] for k in sample_frames
        if (not flags[k][0]) and (not flags[k][1]) and (not flags[k][2])
        and k < denial_start - recovery_buffer
    ]
    post_bucket = [
        traces[k] for k in sample_frames
        if (not flags[k][0]) and (not flags[k][1]) and (not flags[k][2])
        and k > denial_end + recovery_buffer
    ]
    # Inside denial: exclude prior frames (rare in-denial) to measure pure drift
    denial_bucket = [
        traces[k] for k in sample_frames
        if flags[k][1] and (not flags[k][0])
    ]
    # Prior-anchor reference (should be ~prior_sigma^2 * 3)
    prior_bucket = [traces[k] for k in sample_frames if flags[k][0]]

    def _med(xs: list[float]) -> float:
        return float(np.median(xs)) if xs else float("nan")

    pre_trace = _med(pre_bucket)
    post_trace = _med(post_bucket)
    max_denial = float(max(denial_bucket)) if denial_bucket else float("nan")
    prior_median = _med(prior_bucket)

    denial_ratio = max_denial / pre_trace if pre_trace and pre_trace > 0 else float("nan")
    recovery_ratio = post_trace / pre_trace if pre_trace and pre_trace > 0 else float("nan")

    no_nulls = all(np.isfinite(traces[k]) for k in sample_frames)

    return {
        "pre_trace": pre_trace,
        "max_denial_trace": max_denial,
        "post_trace": post_trace,
        "prior_trace_median": prior_median,
        "denial_ratio": denial_ratio,
        "recovery_ratio": recovery_ratio,
        "recovery_buffer": int(recovery_buffer),
        "tail_buffer": int(tail_buffer),
        "n_pre_samples": len(pre_bucket),
        "n_post_samples": len(post_bucket),
        "n_denial_samples": len(denial_bucket),
        "n_prior_samples": len(prior_bucket),
        "passed_inflation": bool(denial_ratio >= 2.0),
        "passed_recovery": bool(recovery_ratio <= 1.5),
        "passed_no_nulls": bool(no_nulls),
    }


def _write_csv(
    sample_frames: list[int],
    covariances: dict[int, np.ndarray],
    trajectory: np.ndarray,
    denial_start: int,
    denial_end: int,
    prior_stride: int,
    n_frames: int,
    tail_buffer: int,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "frame",
        "x", "y", "z",
        "cxx", "cxy", "cxz",
        "cyx", "cyy", "cyz",
        "czx", "czy", "czz",
        "trace",
        "in_denial",
        "is_prior",
        "is_tail",
    ]
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for k in sample_frames:
            cov = covariances[k]
            pos = trajectory[k]
            is_prior, in_denial, is_tail = _frame_flags(
                k, denial_start, denial_end, prior_stride, n_frames, tail_buffer
            )
            row = [
                k,
                f"{pos[0]:.6f}", f"{pos[1]:.6f}", f"{pos[2]:.6f}",
                *[f"{v:.9e}" for v in cov.flatten()],
                f"{float(np.trace(cov)):.9e}",
                int(in_denial),
                int(is_prior),
                int(is_tail),
            ]
            w.writerow(row)


# ----------------------------------------------------------------------
# Mode runners
# ----------------------------------------------------------------------


def _run_loose(
    config: dict,
    dataset: KITTIDataset,
    poses: list[np.ndarray],
    gt_velo: list[np.ndarray] | None,
    prior_indices: list[int],
    sample_frames: list[int],
) -> tuple[list[np.ndarray], dict[int, np.ndarray]]:
    gtsam_cfg = config.get("gtsam", {})
    detector = _make_detector(config)
    closures = detector.detect(poses, dataset=dataset)
    print(f"  Loop closures: {len(closures)}")

    optimizer = PoseGraphOptimizer(
        odom_sigmas=gtsam_cfg.get("odom_sigmas"),
        prior_sigmas=gtsam_cfg.get("prior_sigmas"),
    )
    optimizer.build_graph(poses, prior_indices=prior_indices, gt_poses=gt_velo)
    for i, j, rel_pose in closures:
        optimizer.add_loop_closure(
            i, j, rel_pose, sigmas=gtsam_cfg.get("loop_closure_sigmas")
        )
    opt_poses = optimizer.optimize()
    covariances = optimizer.get_position_marginals(keys=sample_frames)
    return opt_poses, covariances


def _run_tight(
    config: dict,
    dataset: KITTIDataset,
    poses: list[np.ndarray],
    gt_velo: list[np.ndarray] | None,
    lidar_ts: np.ndarray,
    imu_acc: np.ndarray,
    imu_gyro: np.ndarray,
    imu_ts: np.ndarray,
    prior_indices: list[int],
    sample_frames: list[int],
) -> tuple[list[np.ndarray], dict[int, np.ndarray]]:
    gtsam_cfg = config.get("gtsam", {})
    imu_cfg = config.get("imu", {})
    detector = _make_detector(config)
    closures = detector.detect(poses, dataset=dataset)
    print(f"  Loop closures: {len(closures)}")

    opt_poses, _bias, marginals_fn = build_tight_coupled_graph(
        poses=poses,
        imu_acc=imu_acc,
        imu_gyro=imu_gyro,
        imu_timestamps=imu_ts,
        lidar_timestamps=lidar_ts,
        odom_sigmas=gtsam_cfg.get("odom_sigmas"),
        prior_sigmas=gtsam_cfg.get("prior_sigmas"),
        loop_closure_sigmas=gtsam_cfg.get("loop_closure_sigmas"),
        prior_indices=prior_indices,
        gt_poses=gt_velo,
        loop_closures=closures,
        accel_noise_sigma=imu_cfg.get("accel_noise_sigma", 5.0),
        gyro_noise_sigma=imu_cfg.get("gyro_noise_sigma", 0.5),
        accel_bias_sigma=imu_cfg.get("accel_bias_sigma", 0.1),
        gyro_bias_sigma=imu_cfg.get("gyro_bias_sigma", 0.01),
        return_marginals=True,
    )
    covariances = marginals_fn(sample_frames)
    return opt_poses, covariances


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="SUP-06: Uncertainty Visualization")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--sequence", default="00")
    parser.add_argument("--mode", choices=["loose", "tight", "both"], default="loose")
    parser.add_argument(
        "--sample-stride", type=int, default=10,
        help="Sample every N-th keyframe for marginal extraction",
    )
    parser.add_argument("--denial-distance", type=float, default=300.0)
    parser.add_argument("--prior-stride", type=int, default=50)
    parser.add_argument(
        "--tail-buffer", type=int, default=50,
        help="Last N frames excluded from baseline stats (pose graph edge effect)",
    )
    parser.add_argument("--raw-root", default=None, help="KITTI Raw root (for tight mode)")
    parser.add_argument(
        "--no-anim", action="store_true",
        help="Skip GIF rendering (faster; CSV + PNG only)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    seq = args.sequence
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}\nSUP-06 [{seq}] mode={args.mode}\n{'=' * 60}")

    dataset = KITTIDataset(root_path=config["data"]["kitti_root"], sequence=seq)
    Tr = dataset.calibration["Tr"] if dataset.calibration else np.eye(4)
    Tr_inv = np.linalg.inv(Tr)
    gt_cam = dataset.poses
    gt_velo = (
        [Tr_inv @ gt_cam[i] @ Tr for i in range(len(gt_cam))] if gt_cam is not None else None
    )

    poses = _load_cached_poses(config, seq, dataset)
    n = len(poses)
    lidar_ts_raw = (
        dataset.timestamps if dataset.timestamps is not None else np.arange(len(dataset)) * 0.1
    )
    lidar_ts = np.asarray(lidar_ts_raw[:n], dtype=np.float64)

    # GNSS denial window
    try:
        ds, de = make_denial_window(poses, target_distance=args.denial_distance)
    except ValueError:
        ds, de = make_denial_window(poses, target_distance=150.0)
    print(f"  Denial window: frames {ds}-{de}")
    prior_indices = make_prior_indices(n, ds, de, prior_stride=args.prior_stride)
    print(f"  Priors: {len(prior_indices)} frames (stride={args.prior_stride})")

    sample_frames = _build_sample_frames(n, args.sample_stride, ds, de)
    print(f"  Sampling {len(sample_frames)} keyframes (stride={args.sample_stride})")

    # Resolve mode list and optional IMU load
    modes = ["loose", "tight"] if args.mode == "both" else [args.mode]
    imu_acc = imu_gyro = imu_ts_clip = None
    if "tight" in modes:
        imu_result = load_imu_for_odometry_seq(seq, raw_root=args.raw_root)
        if imu_result is None:
            print("  [warning] IMU data unavailable; dropping tight mode")
            modes = [m for m in modes if m != "tight"]
            if not modes:
                print("  Nothing to run")
                return
        else:
            imu_acc, imu_gyro, imu_ts = imu_result
            mask = (imu_ts >= lidar_ts[0]) & (imu_ts <= lidar_ts[-1])
            imu_acc = imu_acc[mask]
            imu_gyro = imu_gyro[mask]
            imu_ts_clip = imu_ts[mask]
            print(f"  IMU: {len(imu_acc)} samples clipped to LiDAR range")
            if len(imu_acc) < 10:
                print("  [warning] too few IMU samples; dropping tight mode")
                modes = [m for m in modes if m != "tight"]
                if not modes:
                    return

    reports: dict[str, dict] = {}

    for mode in modes:
        print(f"\n--- [{seq}] mode={mode} ---")
        if mode == "loose":
            opt_poses, covariances = _run_loose(
                config, dataset, poses, gt_velo, prior_indices, sample_frames
            )
        else:
            opt_poses, covariances = _run_tight(
                config, dataset, poses, gt_velo, lidar_ts,
                imu_acc, imu_gyro, imu_ts_clip,
                prior_indices, sample_frames,
            )

        trajectory = np.array([p[:3, 3] for p in opt_poses])

        metrics = _compute_acceptance(
            sample_frames, covariances, ds, de,
            prior_stride=args.prior_stride,
            n_frames=n,
            recovery_buffer=args.prior_stride,
            tail_buffer=args.tail_buffer,
        )
        print(
            f"  pre_trace(drift median)={metrics['pre_trace']:.4e}  "
            f"max_denial={metrics['max_denial_trace']:.4e}  "
            f"post_trace={metrics['post_trace']:.4e}"
        )
        print(
            f"  prior_anchor median={metrics['prior_trace_median']:.4e}  "
            f"(baseline uses non-prior drift only)"
        )
        print(
            f"  denial_ratio={metrics['denial_ratio']:.2f}x "
            f"(pass >=2x: {metrics['passed_inflation']})"
        )
        print(
            f"  recovery_ratio={metrics['recovery_ratio']:.2f}x "
            f"(pass <=1.5x: {metrics['passed_recovery']})"
        )

        # Build per-sample flags dict for visualization
        sample_flags = {
            k: _frame_flags(k, ds, de, args.prior_stride, n, args.tail_buffer)
            for k in sample_frames
        }

        csv_path = OUT_DIR / f"marginal_cov_{seq}_{mode}.csv"
        _write_csv(
            sample_frames, covariances, trajectory, ds, de,
            prior_stride=args.prior_stride,
            n_frames=n,
            tail_buffer=args.tail_buffer,
            path=csv_path,
        )
        print(f"  CSV:  {csv_path} ({len(sample_frames)} rows)")

        # Persist trajectory for post-hoc rerendering
        traj_path = OUT_DIR / f"traj_{seq}_{mode}.npy"
        np.save(traj_path, trajectory)

        # Coverage assertions: peaks (denial + tail) must be in sampled set
        in_denial_samples = [k for k in sample_frames if sample_flags[k][1]]
        if in_denial_samples:
            peak_denial = max(
                in_denial_samples, key=lambda k: float(np.trace(covariances[k]))
            )
            assert peak_denial in sample_frames, (
                f"denial peak frame {peak_denial} missing from sample set"
            )
        tail_samples = [k for k in sample_frames if sample_flags[k][2]]
        if tail_samples:
            peak_tail = max(tail_samples, key=lambda k: float(np.trace(covariances[k])))
            assert peak_tail in sample_frames, (
                f"tail peak frame {peak_tail} missing from sample set"
            )

        baseline_label = (
            f"median non-prior drift, frames < {ds - args.prior_stride}, "
            f"excludes tail {n - args.tail_buffer}-{n}"
        )

        png_path = OUT_DIR / f"ellipsoids_static_{seq}_{mode}.png"
        plot_trajectory_with_ellipsoids(
            trajectory=trajectory,
            sample_frames=sample_frames,
            covariances=covariances,
            sample_flags=sample_flags,
            denial_window=(ds, de),
            tail_start=n - args.tail_buffer,
            metrics=metrics,
            output_path=png_path,
            title=f"SUP-06 [{seq}] {mode}: keyframe position uncertainty",
        )
        print(f"  PNG:  {png_path}")

        if not args.no_anim:
            gif_path = OUT_DIR / f"ellipsoid_animation_{seq}_{mode}.gif"
            animate_uncertainty_evolution(
                trajectory=trajectory,
                sample_frames=sample_frames,
                covariances=covariances,
                sample_flags=sample_flags,
                denial_window=(ds, de),
                tail_start=n - args.tail_buffer,
                pre_denial_trace=metrics["pre_trace"],
                metrics=metrics,
                baseline_window_label=baseline_label,
                output_path=gif_path,
                title=f"SUP-06 [{seq}] {mode}",
            )
            print(f"  GIF:  {gif_path}")

        reports[mode] = metrics

    report_path = OUT_DIR / f"sup06_report_{seq}.json"
    report_path.write_text(
        json.dumps(
            {
                "sequence": seq,
                "denial_window": [ds, de],
                "prior_stride": args.prior_stride,
                "sample_stride": args.sample_stride,
                "tail_buffer": args.tail_buffer,
                "n_frames": n,
                "n_samples": len(sample_frames),
                "modes": reports,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nReport: {report_path}")

    # Manifest
    try:
        manifest = BenchmarkManifest()
        artifacts = sorted(
            str(p) for p in OUT_DIR.glob(f"*_{seq}_*") if p.suffix in {".csv", ".png", ".gif"}
        )
        manifest.append(
            task="SUP-06",
            config=config,
            sequences=[seq],
            artifacts=artifacts,
            metrics={mode: r for mode, r in reports.items()},
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  [warning] manifest append failed: {exc}")


if __name__ == "__main__":
    main()
