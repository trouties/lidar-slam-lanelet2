#!/usr/bin/env python3
"""SUP-04: Compare tightly-coupled (GTSAM IMU preintegration) vs
loosely-coupled (ESKF constant-velocity) fusion.

Produces:
  - benchmarks/tight_vs_loose/ape_compare.csv
  - benchmarks/tight_vs_loose/bias_seq00.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from scripts.run_pipeline import load_config
from src.benchmarks import BenchmarkManifest, make_denial_window, score_denial_drift
from src.benchmarks.gnss_denial import make_prior_indices
from src.cache import LayeredCache
from src.data.imu_loader import load_imu_for_odometry_seq
from src.data.kitti_loader import KITTIDataset
from src.odometry.kiss_icp_wrapper import KissICPOdometry, evaluate_odometry
from src.optimization.imu_factor import build_tight_coupled_graph
from src.optimization.loop_closure import LoopClosureDetector
from src.optimization.pose_graph import PoseGraphOptimizer

OUT_DIR = Path("benchmarks/tight_vs_loose")


def _run_loose(config, seq, dataset, poses, gt_velo, timestamps, prior_indices=None):
    """Run loose coupling: pose graph + ESKF."""
    from src.fusion.eskf import ESKF

    gtsam_cfg = config.get("gtsam", {})
    lc_cfg = config.get("loop_closure", {})
    detector = LoopClosureDetector(
        distance_threshold=lc_cfg.get("distance_threshold", 15.0),
        min_frame_gap=lc_cfg.get("min_frame_gap", 100),
        icp_fitness_threshold=lc_cfg.get("icp_fitness_threshold", 0.9),
        mode=lc_cfg.get("mode", "v1"),
    )
    closures = detector.detect(poses, dataset=dataset)
    optimizer = PoseGraphOptimizer(
        odom_sigmas=gtsam_cfg.get("odom_sigmas"),
        prior_sigmas=gtsam_cfg.get("prior_sigmas"),
    )
    optimizer.build_graph(poses, prior_indices=prior_indices, gt_poses=gt_velo)
    for i, j, rel_pose in closures:
        optimizer.add_loop_closure(i, j, rel_pose)
    opt_poses = optimizer.optimize()

    # ESKF smoothing
    ekf_cfg = config.get("ekf", {})
    eskf = ESKF(
        process_noise_pos=ekf_cfg.get("process_noise_pos", 0.1),
        process_noise_vel=ekf_cfg.get("process_noise_vel", 0.5),
        process_noise_rot=ekf_cfg.get("process_noise_rot", 0.01),
        measurement_noise_pos=ekf_cfg.get("measurement_noise_pos", 0.05),
        measurement_noise_rot=ekf_cfg.get("measurement_noise_rot", 0.01),
    )
    fused = eskf.run(opt_poses, timestamps[:len(opt_poses)])
    return fused


def _run_tight(config, seq, dataset, poses, gt_velo, lidar_ts, imu_acc, imu_gyro,
               imu_ts, prior_indices=None):
    """Run tight coupling: pose graph + GTSAM IMU preintegration + loop closure."""
    gtsam_cfg = config.get("gtsam", {})
    lc_cfg = config.get("loop_closure", {})

    # Detect loop closures (same as loose path)
    detector = LoopClosureDetector(
        distance_threshold=lc_cfg.get("distance_threshold", 15.0),
        min_frame_gap=lc_cfg.get("min_frame_gap", 100),
        icp_fitness_threshold=lc_cfg.get("icp_fitness_threshold", 0.9),
        mode=lc_cfg.get("mode", "v1"),
    )
    closures = detector.detect(poses, dataset=dataset)

    opt_poses, bias_history = build_tight_coupled_graph(
        poses=poses,
        imu_acc=imu_acc,
        imu_gyro=imu_gyro,
        imu_timestamps=imu_ts,
        lidar_timestamps=lidar_ts,
        odom_sigmas=gtsam_cfg.get("odom_sigmas"),
        prior_sigmas=gtsam_cfg.get("prior_sigmas"),
        prior_indices=prior_indices,
        gt_poses=gt_velo,
        loop_closures=closures,
    )
    return opt_poses, bias_history


def main():
    parser = argparse.ArgumentParser(description="SUP-04: Tight vs Loose Comparison")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--sequences", default="00,05")
    parser.add_argument("--check-bias", action="store_true", help="Only check bias bounds")
    parser.add_argument("--raw-root", default=None, help="KITTI Raw root dir")
    args = parser.parse_args()

    config = load_config(args.config)
    sequences = [s.strip() for s in args.sequences.split(",")]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ape_rows = []

    for seq in sequences:
        print(f"\n{'='*60}")
        print(f"Sequence {seq}")
        print(f"{'='*60}")

        # Check IMU availability
        imu_result = load_imu_for_odometry_seq(seq, raw_root=args.raw_root)
        if imu_result is None:
            print(f"  No IMU data for seq {seq} — skipping")
            ape_rows.append({
                "sequence": seq, "mode": "tight", "scenario": "normal",
                "ape_rmse": "FAIL:NO_IMU", "ape_mean": "FAIL:NO_IMU",
            })
            continue

        imu_acc, imu_gyro, imu_ts = imu_result
        print(f"  IMU: {len(imu_acc)} samples, {imu_ts[-1]:.1f}s")

        # Load dataset
        dataset = KITTIDataset(root_path=config["data"]["kitti_root"], sequence=seq)
        Tr = dataset.calibration["Tr"] if dataset.calibration else np.eye(4)
        Tr_inv = np.linalg.inv(Tr)
        gt_cam = dataset.poses
        gt_velo = (
            [Tr_inv @ gt_cam[i] @ Tr for i in range(len(gt_cam))]
            if gt_cam is not None
            else None
        )
        lidar_ts = (
            dataset.timestamps
            if dataset.timestamps is not None
            else np.arange(len(dataset)) * 0.1
        )

        # Get cached LiDAR odometry
        cache_cfg = config.get("cache", {})
        cache = None
        if cache_cfg.get("enabled", False):
            cache = LayeredCache(root=cache_cfg.get("root", "cache/kitti"), sequence=seq)
        odom_cached = cache.load_odometry(config) if cache else None
        if odom_cached is not None:
            poses_arr, _ = odom_cached
            poses = [poses_arr[i] for i in range(poses_arr.shape[0])]
            print(f"  Loaded {len(poses)} cached odometry poses")
        else:
            kiss_cfg = config.get("kiss_icp", {})
            odom = KissICPOdometry(
                max_range=kiss_cfg.get("max_range", 100.0),
                min_range=kiss_cfg.get("min_range", 5.0),
                voxel_size=kiss_cfg.get("voxel_size", 1.0),
            )
            poses = odom.run(dataset)

        # Align IMU timestamps to LiDAR timestamps
        # KITTI Odometry timestamps start at 0; IMU timestamps may have different base
        # Assume both start at frame 0
        n_frames = min(len(poses), len(lidar_ts))
        poses = poses[:n_frames]
        lidar_ts_arr = np.asarray(lidar_ts[:n_frames], dtype=np.float64)

        # Clip IMU to LiDAR time range
        imu_mask = (imu_ts >= lidar_ts_arr[0]) & (imu_ts <= lidar_ts_arr[-1])
        imu_acc_clip = imu_acc[imu_mask]
        imu_gyro_clip = imu_gyro[imu_mask]
        imu_ts_clip = imu_ts[imu_mask]
        print(f"  IMU clipped to LiDAR range: {len(imu_acc_clip)} samples")

        if len(imu_acc_clip) < 10:
            print(f"  Too few IMU samples — skipping tight coupling for seq {seq}")
            continue

        # === Normal scenario ===
        print(f"\n--- [{seq}] Normal scenario ---")

        # Loose
        print("  Running loose coupling...")
        loose_poses = _run_loose(config, seq, dataset, poses, gt_velo, lidar_ts_arr)
        if gt_velo is not None:
            n_eval = min(len(loose_poses), len(gt_velo))
            loose_metrics = evaluate_odometry(loose_poses[:n_eval], gt_velo[:n_eval])
            print(f"  Loose APE: {loose_metrics['ape']['rmse']:.4f} m (RMSE)")
            ape_rows.append({
                "sequence": seq, "mode": "loose", "scenario": "normal",
                "ape_rmse": f"{loose_metrics['ape']['rmse']:.4f}",
                "ape_mean": f"{loose_metrics['ape']['mean']:.4f}",
            })

        # Tight
        print("  Running tight coupling...")
        tight_poses, bias_hist = _run_tight(
            config, seq, dataset, poses, gt_velo, lidar_ts_arr,
            imu_acc_clip, imu_gyro_clip, imu_ts_clip,
        )
        if gt_velo is not None:
            n_eval = min(len(tight_poses), len(gt_velo))
            tight_metrics = evaluate_odometry(tight_poses[:n_eval], gt_velo[:n_eval])
            print(f"  Tight APE: {tight_metrics['ape']['rmse']:.4f} m (RMSE)")
            ape_rows.append({
                "sequence": seq, "mode": "tight", "scenario": "normal",
                "ape_rmse": f"{tight_metrics['ape']['rmse']:.4f}",
                "ape_mean": f"{tight_metrics['ape']['mean']:.4f}",
            })

        # Save bias
        bias_arr = np.array(bias_hist)
        bias_path = OUT_DIR / f"bias_{seq}.csv"
        np.savetxt(
            bias_path, bias_arr,
            header="ax_bias,ay_bias,az_bias,wx_bias,wy_bias,wz_bias",
            delimiter=",", comments="",
        )
        print(f"  Bias saved to {bias_path}")

        # === GNSS denied scenario ===
        print(f"\n--- [{seq}] GNSS denied scenario ---")
        try:
            start, end = make_denial_window(poses, target_distance=300.0)
        except ValueError:
            # Try 150m if 300m is too much
            try:
                start, end = make_denial_window(poses, target_distance=150.0)
            except ValueError:
                print("  Trajectory too short for denial window")
                continue

        prior_indices = make_prior_indices(n_frames, start, end, prior_stride=50)
        denial_dist = 0.0
        src_poses = gt_velo[start : end + 1] if gt_velo else poses[start : end + 1]
        trans = np.array([p[:3, 3] for p in src_poses])
        if len(trans) > 1:
            denial_dist = float(np.sum(np.linalg.norm(np.diff(trans, axis=0), axis=1)))
        print(f"  Denial window: frames {start}-{end}, distance {denial_dist:.1f}m")

        # Loose denied
        loose_denied = _run_loose(config, seq, dataset, poses, gt_velo, lidar_ts_arr,
                                   prior_indices=prior_indices)
        if gt_velo is not None:
            drift_loose = score_denial_drift(loose_denied, gt_velo, start, end)
            print(f"  Loose denied APE: {drift_loose['ape_mean']:.4f} m")
            ape_rows.append({
                "sequence": seq, "mode": "loose", "scenario": "gnss_denied",
                "ape_rmse": f"{drift_loose['ape_rmse']:.4f}",
                "ape_mean": f"{drift_loose['ape_mean']:.4f}",
            })

        # Tight denied
        tight_denied, _ = _run_tight(
            config, seq, dataset, poses, gt_velo, lidar_ts_arr,
            imu_acc_clip, imu_gyro_clip, imu_ts_clip,
            prior_indices=prior_indices,
        )
        if gt_velo is not None:
            drift_tight = score_denial_drift(tight_denied, gt_velo, start, end)
            print(f"  Tight denied APE: {drift_tight['ape_mean']:.4f} m")
            ape_rows.append({
                "sequence": seq, "mode": "tight", "scenario": "gnss_denied",
                "ape_rmse": f"{drift_tight['ape_rmse']:.4f}",
                "ape_mean": f"{drift_tight['ape_mean']:.4f}",
            })

    # Write APE comparison
    if ape_rows:
        csv_path = OUT_DIR / "ape_compare.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(ape_rows[0].keys()))
            writer.writeheader()
            writer.writerows(ape_rows)
        print(f"\nAPE comparison: {csv_path} ({len(ape_rows)} rows)")

    # Manifest
    manifest = BenchmarkManifest()
    manifest.append(
        task="SUP-04",
        config=config,
        sequences=sequences,
        artifacts=[str(OUT_DIR / "ape_compare.csv")],
        metrics={"n_rows": len(ape_rows)},
    )


if __name__ == "__main__":
    main()
