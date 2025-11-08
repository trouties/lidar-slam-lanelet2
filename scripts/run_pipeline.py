"""Main entry point for the LiDAR SLAM HD Map pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from src.data.kitti_loader import KITTIDataset
from src.fusion.eskf import ESKF
from src.odometry.kiss_icp_wrapper import KissICPOdometry, evaluate_odometry
from src.optimization.loop_closure import LoopClosureDetector
from src.optimization.pose_graph import PoseGraphOptimizer


def load_config(config_path: str | Path) -> dict:
    """Load YAML configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="LiDAR SLAM HD Map Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(config.get("output", {}).get("dir", "results"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Stage 1: Data Loading ---
    print("=== Stage 1: Loading KITTI dataset ===")
    dataset = KITTIDataset(
        root_path=config["data"]["kitti_root"],
        sequence=config["data"]["sequence"],
    )
    print(f"  Sequence: {dataset.sequence}")
    print(f"  Frames: {len(dataset)}")
    if len(dataset) == 0:
        print("  No scans found. Check data path and run scripts/verify_kitti.py")
        return

    # --- Stage 2: LiDAR Odometry ---
    print("\n=== Stage 2: KISS-ICP Odometry ===")
    kiss_cfg = config.get("kiss_icp", {})
    odom = KissICPOdometry(
        max_range=kiss_cfg.get("max_range", 100.0),
        min_range=kiss_cfg.get("min_range", 5.0),
        voxel_size=kiss_cfg.get("voxel_size", 1.0),
    )
    poses = odom.run(dataset)

    # Save poses
    poses_path = output_dir / f"poses_{dataset.sequence}.txt"
    KissICPOdometry.save_poses_kitti_format(poses, poses_path)
    print(f"  Saved {len(poses)} poses to {poses_path}")

    # Evaluate odometry against ground truth if available
    if dataset.poses is not None:
        print("\n=== Odometry Evaluation ===")
        n = min(len(poses), len(dataset.poses))
        result = evaluate_odometry(poses[:n], dataset.poses[:n])
        print(f"  APE RMSE: {result['ape']['rmse']:.4f} m")
        print(f"  APE Mean: {result['ape']['mean']:.4f} m")
        print(f"  RPE RMSE: {result['rpe']['rmse']:.4f} m")
    else:
        print(f"\n  No ground truth for sequence {dataset.sequence} (only 00-10 have GT)")

    # --- Stage 3: Pose Graph Optimization ---
    print("\n=== Stage 3: Pose Graph Optimization ===")
    gtsam_cfg = config.get("gtsam", {})
    lc_cfg = config.get("loop_closure", {})

    # Detect loop closures
    detector = LoopClosureDetector(
        distance_threshold=lc_cfg.get("distance_threshold", 15.0),
        min_frame_gap=lc_cfg.get("min_frame_gap", 100),
        icp_fitness_threshold=lc_cfg.get("icp_fitness_threshold", 0.3),
    )
    closures = detector.detect(poses, dataset=dataset)
    print(f"  Detected {len(closures)} loop closure(s)")

    # Build and optimize pose graph
    optimizer = PoseGraphOptimizer(
        odom_sigmas=gtsam_cfg.get("odom_sigmas"),
        prior_sigmas=gtsam_cfg.get("prior_sigmas"),
    )
    optimizer.build_graph(poses)
    for i, j, rel_pose in closures:
        optimizer.add_loop_closure(i, j, rel_pose)

    optimized_poses = optimizer.optimize()

    # Save optimized poses
    opt_path = output_dir / f"poses_optimized_{dataset.sequence}.txt"
    KissICPOdometry.save_poses_kitti_format(optimized_poses, opt_path)
    print(f"  Saved {len(optimized_poses)} optimized poses to {opt_path}")

    # Evaluate optimized trajectory
    if dataset.poses is not None:
        n = min(len(optimized_poses), len(dataset.poses))
        result_opt = evaluate_odometry(optimized_poses[:n], dataset.poses[:n])
        print(f"  Optimized APE RMSE: {result_opt['ape']['rmse']:.4f} m")
        print(f"  Optimized APE Mean: {result_opt['ape']['mean']:.4f} m")

    # --- Stage 4: ESKF Sensor Fusion ---
    print("\n=== Stage 4: ESKF Sensor Fusion ===")
    ekf_cfg = config.get("ekf", {})
    eskf = ESKF(
        process_noise_pos=ekf_cfg.get("process_noise_pos", 0.1),
        process_noise_vel=ekf_cfg.get("process_noise_vel", 0.5),
        process_noise_rot=ekf_cfg.get("process_noise_rot", 0.01),
        measurement_noise_pos=ekf_cfg.get("measurement_noise_pos", 0.05),
        measurement_noise_rot=ekf_cfg.get("measurement_noise_rot", 0.01),
    )

    if dataset.timestamps is not None:
        fused_poses = eskf.run(optimized_poses, dataset.timestamps)
    else:
        # Fallback: assume 10 Hz
        timestamps = np.arange(len(optimized_poses)) * 0.1
        fused_poses = eskf.run(optimized_poses, timestamps)

    fused_path = output_dir / f"poses_fused_{dataset.sequence}.txt"
    KissICPOdometry.save_poses_kitti_format(fused_poses, fused_path)
    print(f"  Saved {len(fused_poses)} fused poses to {fused_path}")

    if dataset.poses is not None:
        n = min(len(fused_poses), len(dataset.poses))
        result_fused = evaluate_odometry(fused_poses[:n], dataset.poses[:n])
        print(f"  Fused APE RMSE: {result_fused['ape']['rmse']:.4f} m")
        print(f"  Fused APE Mean: {result_fused['ape']['mean']:.4f} m")

    print("\nDone.")


if __name__ == "__main__":
    main()
