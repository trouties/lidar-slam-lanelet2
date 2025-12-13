"""Main entry point for the LiDAR SLAM HD Map pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from src.data.kitti_loader import KITTIDataset
from src.fusion.eskf import ESKF
from src.mapping import (
    MapBuilder,
    cluster_points,
    extract_lane_markings,
    extract_road_surface,
    save_features_geojson,
)
from src.odometry.kiss_icp_wrapper import (
    KissICPOdometry,
    evaluate_odometry,
    transform_poses_to_camera_frame,
)
from src.optimization.loop_closure import LoopClosureDetector
from src.optimization.pose_graph import PoseGraphOptimizer


def load_config(config_path: str | Path) -> dict:
    """Load YAML configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def _eval(label: str, est_poses, gt_poses):
    """Evaluate and print APE/RPE if ground truth is available."""
    if gt_poses is None:
        return
    n = min(len(est_poses), len(gt_poses))
    result = evaluate_odometry(est_poses[:n], gt_poses[:n])
    print(f"  {label} APE RMSE: {result['ape']['rmse']:.4f} m")
    print(f"  {label} APE Mean: {result['ape']['mean']:.4f} m")
    if "rpe" in result:
        print(f"  {label} RPE RMSE: {result['rpe']['rmse']:.4f} m")


def main() -> None:
    parser = argparse.ArgumentParser(description="LiDAR SLAM HD Map Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Limit number of frames to process (for quick testing)",
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
    if args.max_frames is not None and args.max_frames < len(dataset):
        dataset.scan_files = dataset.scan_files[: args.max_frames]
        print(f"  Limited to {args.max_frames} frames")
    print(f"  Sequence: {dataset.sequence}")
    print(f"  Frames: {len(dataset)}")
    print(f"  Calibration: {'loaded' if dataset.calibration else 'missing'}")
    print(f"  Ground truth: {'yes' if dataset.poses is not None else 'no'}")
    if len(dataset) == 0:
        print("  No scans found. Check data path and run scripts/verify_kitti.py")
        return

    # Calibration for frame conversion (Velodyne → camera)
    Tr = dataset.calibration["Tr"] if dataset.calibration else np.eye(4)
    # GT poses in camera frame; convert to Velodyne frame for consistent processing
    gt_poses_cam = dataset.poses  # (M, 4, 4) or None
    Tr_inv = np.linalg.inv(Tr)
    if gt_poses_cam is not None:
        gt_velo = [Tr_inv @ gt_poses_cam[i] @ Tr for i in range(len(gt_poses_cam))]
    else:
        gt_velo = None

    # --- Stage 2: LiDAR Odometry ---
    print("\n=== Stage 2: KISS-ICP Odometry ===")
    kiss_cfg = config.get("kiss_icp", {})
    odom = KissICPOdometry(
        max_range=kiss_cfg.get("max_range", 100.0),
        min_range=kiss_cfg.get("min_range", 5.0),
        voxel_size=kiss_cfg.get("voxel_size", 1.0),
    )
    poses = odom.run(dataset)  # Velodyne frame

    # Save in KITTI camera-frame convention
    poses_cam = transform_poses_to_camera_frame(poses, Tr)
    poses_path = output_dir / f"poses_{dataset.sequence}.txt"
    KissICPOdometry.save_poses_kitti_format(poses_cam, poses_path)
    print(f"  Saved {len(poses)} poses to {poses_path}")

    _eval("Odometry", poses, gt_velo)

    # --- Stage 3: Pose Graph Optimization ---
    print("\n=== Stage 3: Pose Graph Optimization ===")
    gtsam_cfg = config.get("gtsam", {})
    lc_cfg = config.get("loop_closure", {})

    # Loop closure: all in Velodyne frame (consistent with point clouds)
    detector = LoopClosureDetector(
        distance_threshold=lc_cfg.get("distance_threshold", 15.0),
        min_frame_gap=lc_cfg.get("min_frame_gap", 100),
        icp_fitness_threshold=lc_cfg.get("icp_fitness_threshold", 0.3),
    )
    closures = detector.detect(poses, dataset=dataset)
    print(f"  Detected {len(closures)} loop closure(s)")

    # Build and optimize pose graph (Velodyne frame)
    optimizer = PoseGraphOptimizer(
        odom_sigmas=gtsam_cfg.get("odom_sigmas"),
        prior_sigmas=gtsam_cfg.get("prior_sigmas"),
    )
    optimizer.build_graph(poses)
    for i, j, rel_pose in closures:
        optimizer.add_loop_closure(i, j, rel_pose)

    optimized_poses = optimizer.optimize()

    opt_cam = transform_poses_to_camera_frame(optimized_poses, Tr)
    opt_path = output_dir / f"poses_optimized_{dataset.sequence}.txt"
    KissICPOdometry.save_poses_kitti_format(opt_cam, opt_path)
    print(f"  Saved {len(optimized_poses)} optimized poses to {opt_path}")

    _eval("Optimized", optimized_poses, gt_velo)

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
        ts = dataset.timestamps[: len(optimized_poses)]
    else:
        ts = np.arange(len(optimized_poses)) * 0.1

    fused_poses = eskf.run(optimized_poses, ts)

    fused_cam = transform_poses_to_camera_frame(fused_poses, Tr)
    fused_path = output_dir / f"poses_fused_{dataset.sequence}.txt"
    KissICPOdometry.save_poses_kitti_format(fused_cam, fused_path)
    print(f"  Saved {len(fused_poses)} fused poses to {fused_path}")

    _eval("Fused", fused_poses, gt_velo)

    # --- Stage 5: Semantic Map Assembly & Feature Extraction ---
    print("\n=== Stage 5: Semantic Map Assembly ===")
    mapping_cfg = config.get("mapping", {})
    builder = MapBuilder(
        voxel_size=mapping_cfg.get("voxel_size", 0.1),
        max_range=mapping_cfg.get("max_range", 50.0),
        downsample_every=mapping_cfg.get("downsample_every", 50),
    )
    global_map = builder.build(dataset, fused_poses)
    print(f"  Global map points: {len(global_map.points):,}")

    map_path = output_dir / f"global_map_{dataset.sequence}.pcd"
    MapBuilder.save(global_map, map_path)
    print(f"  Saved map to {map_path}")

    # Feature extraction from the global map.
    xyz = np.asarray(global_map.points)
    intensities = np.asarray(global_map.colors)[:, 0]  # colors encode reflectance

    road_pts, road_int = extract_road_surface(
        xyz,
        intensities,
        z_min=mapping_cfg.get("road_z_min", -1.95),
        z_max=mapping_cfg.get("road_z_max", -1.45),
    )
    print(f"  Road surface points: {len(road_pts):,}")

    lane_pts = extract_lane_markings(
        road_pts,
        road_int,
        intensity_threshold=mapping_cfg.get("intensity_threshold", 0.35),
    )
    print(f"  Lane marking candidates: {len(lane_pts):,}")

    clusters = cluster_points(
        lane_pts,
        eps=mapping_cfg.get("dbscan_eps", 0.5),
        min_points=mapping_cfg.get("dbscan_min_points", 10),
    )
    print(f"  Lane marking clusters: {len(clusters)}")

    geojson_path = output_dir / f"features_{dataset.sequence}.geojson"
    save_features_geojson(clusters, geojson_path, feature_type="lane_marking")
    print(f"  Saved features to {geojson_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
