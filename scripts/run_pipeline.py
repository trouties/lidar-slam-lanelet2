"""Main entry point for the LiDAR SLAM HD Map pipeline.

Supports a layered cache (``src.cache.LayeredCache``) so that Stage 5
parameter iteration or Stage 6 tuning does not need to re-run the upstream
stages. See :func:`run_pipeline_cached` for the programmatic entry point
used by the benchmark script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.benchmarks.timing import StageTimer
from src.cache import STAGE_ORDER, LayeredCache
from src.data.kitti_loader import KITTIDataset
from src.export import export_lanelet2_osm
from src.fusion.eskf import ESKF
from src.mapping import (
    MapBuilder,
    cluster_points,
    extract_curbs,
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

# Force-rebuild accepts cache stage names plus two aliases.
_FORCE_REBUILD_CHOICES = ["none", "all", *STAGE_ORDER]


def load_config(config_path: str | Path) -> dict:
    """Load YAML configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def _eval_metrics(est_poses, gt_poses) -> dict[str, float]:
    """Return APE/RPE metrics dict (empty if no GT)."""
    if gt_poses is None:
        return {}
    n = min(len(est_poses), len(gt_poses))
    result = evaluate_odometry(est_poses[:n], gt_poses[:n])
    out: dict[str, float] = {
        "ape_rmse": float(result["ape"]["rmse"]),
        "ape_mean": float(result["ape"]["mean"]),
    }
    if "rpe" in result:
        out["rpe_rmse"] = float(result["rpe"]["rmse"])
    return out


def _log_metrics(label: str, metrics: dict[str, float], verbose: bool) -> None:
    if not verbose or not metrics:
        return
    print(f"  {label} APE RMSE: {metrics.get('ape_rmse', float('nan')):.4f} m")
    print(f"  {label} APE Mean: {metrics.get('ape_mean', float('nan')):.4f} m")
    if "rpe_rmse" in metrics:
        print(f"  {label} RPE RMSE: {metrics['rpe_rmse']:.4f} m")


def _cluster_size_stats(clusters: list[np.ndarray]) -> dict[str, float]:
    if not clusters:
        return {"count": 0, "p05": 0, "p50": 0, "p95": 0, "min": 0, "max": 0}
    sizes = np.array([c.shape[0] for c in clusters])
    return {
        "count": int(sizes.size),
        "min": int(sizes.min()),
        "p05": int(np.percentile(sizes, 5)),
        "p50": int(np.percentile(sizes, 50)),
        "p95": int(np.percentile(sizes, 95)),
        "max": int(sizes.max()),
    }


def run_pipeline_cached(
    config: dict,
    sequence: str,
    cache: LayeredCache | None = None,
    force_rebuild: str = "none",
    max_frames: int | None = None,
    output_dir: Path | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run the pipeline on a single KITTI sequence, honoring the cache.

    Args:
        config: Parsed YAML config dict.
        sequence: KITTI Odometry sequence id, e.g. ``"00"``.
        cache: Optional :class:`LayeredCache`. If ``None``, caching is off.
        force_rebuild: One of ``_FORCE_REBUILD_CHOICES``. ``"none"`` (default)
            uses cache wherever valid. ``"all"`` clears the whole chain.
            Any stage name clears that stage and all downstream stages.
        max_frames: Optional frame cap for quick sanity testing.
        output_dir: Where the ``results/`` artifacts go. If ``None``, uses
            ``config["output"]["dir"]``.
        verbose: Print per-stage progress lines.

    Returns:
        A metrics dict summarizing this run (consumed by the benchmark script).
    """
    cfg = config  # alias for brevity below
    cfg = {**cfg, "data": {**cfg.get("data", {}), "sequence": sequence}}

    if output_dir is None:
        output_dir = Path(cfg.get("output", {}).get("dir", "results"))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ``max_frames`` is strictly a sanity-test / debug knob. Partial runs
    # would poison the cache because frame-count is not part of the config
    # hash — silently disable caching when a frame cap is in effect.
    if max_frames is not None and cache is not None:
        if verbose:
            print("  [cache disabled] max_frames set — not persisting to cache")
        cache = None

    if cache is not None and force_rebuild != "none":
        cache.invalidate(force_rebuild)

    # --- Stage 1: Data Loading (always runs; file I/O is cheap) ---
    if verbose:
        print(f"\n=== [{sequence}] Stage 1: Loading KITTI dataset ===")
    dataset = KITTIDataset(
        root_path=cfg["data"]["kitti_root"],
        sequence=sequence,
    )
    if max_frames is not None and max_frames < len(dataset):
        dataset.scan_files = dataset.scan_files[:max_frames]
        if verbose:
            print(f"  Limited to {max_frames} frames")
    if verbose:
        print(f"  Sequence: {dataset.sequence}  Frames: {len(dataset)}")
    if len(dataset) == 0:
        raise RuntimeError(f"No scans found for sequence {sequence}")

    Tr = dataset.calibration["Tr"] if dataset.calibration else np.eye(4)
    Tr_inv = np.linalg.inv(Tr)
    gt_poses_cam = dataset.poses
    if gt_poses_cam is not None:
        gt_velo = [Tr_inv @ gt_poses_cam[i] @ Tr for i in range(len(gt_poses_cam))]
    else:
        gt_velo = None

    summary: dict[str, Any] = {
        "sequence": sequence,
        "frame_count": len(dataset),
        "has_gt": gt_velo is not None,
        "cache_hits": {},
        "metrics": {},
        "timing": {},
    }

    # --- Stage 2: LiDAR Odometry ---
    if verbose:
        print(f"=== [{sequence}] Stage 2: KISS-ICP Odometry ===")
    odom_cached = cache.load_odometry(cfg) if cache else None
    if odom_cached is not None:
        poses_arr, timestamps = odom_cached
        poses = [poses_arr[i] for i in range(poses_arr.shape[0])]
        if verbose:
            print(f"  [cache hit] {len(poses)} poses loaded")
        summary["cache_hits"]["odometry"] = True
        cached_m = cache.get_stage_metrics("odometry", cfg) if cache else None
        if cached_m and "timing" in cached_m:
            summary["timing"]["stage2"] = cached_m["timing"]
    else:
        kiss_cfg = cfg.get("kiss_icp", {})
        odom = KissICPOdometry(
            max_range=kiss_cfg.get("max_range", 100.0),
            min_range=kiss_cfg.get("min_range", 5.0),
            voxel_size=kiss_cfg.get("voxel_size", 1.0),
        )
        timer_s2 = StageTimer("stage2_odometry")
        with timer_s2:
            poses = odom.run(dataset)
        summary["timing"]["stage2"] = timer_s2.summary()
        if dataset.timestamps is not None:
            timestamps = np.asarray(dataset.timestamps[: len(poses)], dtype=np.float64)
        else:
            timestamps = np.arange(len(poses), dtype=np.float64) * 0.1
        summary["cache_hits"]["odometry"] = False

    odom_metrics = _eval_metrics(poses, gt_velo)
    summary["metrics"]["odometry"] = odom_metrics
    _log_metrics("Odometry", odom_metrics, verbose)

    if cache is not None and not summary["cache_hits"]["odometry"]:
        cache.save_odometry(
            np.asarray(poses),
            timestamps,
            cfg,
            metrics={
                "frame_count": len(poses),
                **odom_metrics,
                "timing": summary["timing"].get("stage2", {}),
            },
        )

    # Write human-readable kitti-format poses for observability.
    poses_cam = transform_poses_to_camera_frame(poses, Tr)
    KissICPOdometry.save_poses_kitti_format(poses_cam, output_dir / f"poses_{sequence}.txt")

    # --- Stage 3: Pose Graph Optimization ---
    if verbose:
        print(f"=== [{sequence}] Stage 3: Pose Graph Optimization ===")
    opt_cached = cache.load_optimized(cfg) if cache else None
    if opt_cached is not None:
        opt_arr = opt_cached
        optimized_poses = [opt_arr[i] for i in range(opt_arr.shape[0])]
        if verbose:
            print(f"  [cache hit] {len(optimized_poses)} optimized poses loaded")
        summary["cache_hits"]["optimized"] = True
        cached_m = cache.get_stage_metrics("optimized", cfg) if cache else None
        if cached_m and "timing" in cached_m:
            summary["timing"]["stage3"] = cached_m["timing"]
    else:
        gtsam_cfg = cfg.get("gtsam", {})
        lc_cfg = cfg.get("loop_closure", {})
        sc_cfg = lc_cfg.get("scan_context", {})
        detector = LoopClosureDetector(
            distance_threshold=lc_cfg.get("distance_threshold", 15.0),
            min_frame_gap=lc_cfg.get("min_frame_gap", 100),
            icp_fitness_threshold=lc_cfg.get("icp_fitness_threshold", 0.9),
            mode=lc_cfg.get("mode", "v1"),
            sc_num_rings=sc_cfg.get("num_rings", 20),
            sc_num_sectors=sc_cfg.get("num_sectors", 60),
            sc_max_range=sc_cfg.get("max_range", 80.0),
            sc_distance_threshold=sc_cfg.get("distance_threshold", 0.4),
            sc_top_k=sc_cfg.get("top_k", 10),
            sc_query_stride=sc_cfg.get("query_stride", 1),
            sc_max_matches_per_query=sc_cfg.get("max_matches_per_query", 0),
        )
        timer_s3 = StageTimer("stage3_optimization")
        with timer_s3:
            closures = detector.detect(poses, dataset=dataset)
            if verbose:
                print(f"  Detected {len(closures)} loop closure(s)")
            optimizer = PoseGraphOptimizer(
                odom_sigmas=gtsam_cfg.get("odom_sigmas"),
                prior_sigmas=gtsam_cfg.get("prior_sigmas"),
            )
            optimizer.build_graph(poses)
            for i, j, rel_pose in closures:
                optimizer.add_loop_closure(i, j, rel_pose)
            optimized_poses = optimizer.optimize()
        summary["timing"]["stage3"] = timer_s3.summary()
        summary["cache_hits"]["optimized"] = False
        summary["loop_closures"] = len(closures)

    opt_metrics = _eval_metrics(optimized_poses, gt_velo)
    summary["metrics"]["optimized"] = opt_metrics
    _log_metrics("Optimized", opt_metrics, verbose)

    if cache is not None and not summary["cache_hits"]["optimized"]:
        cache.save_optimized(
            np.asarray(optimized_poses),
            cfg,
            metrics={
                "frame_count": len(optimized_poses),
                "loop_closures": summary.get("loop_closures", 0),
                **opt_metrics,
                "timing": summary["timing"].get("stage3", {}),
            },
        )

    opt_cam = transform_poses_to_camera_frame(optimized_poses, Tr)
    KissICPOdometry.save_poses_kitti_format(opt_cam, output_dir / f"poses_optimized_{sequence}.txt")

    # --- Stage 4: ESKF Sensor Fusion ---
    if verbose:
        print(f"=== [{sequence}] Stage 4: ESKF Sensor Fusion ===")
    fused_cached = cache.load_fused(cfg) if cache else None
    if fused_cached is not None:
        fused_arr = fused_cached
        fused_poses = [fused_arr[i] for i in range(fused_arr.shape[0])]
        if verbose:
            print(f"  [cache hit] {len(fused_poses)} fused poses loaded")
        summary["cache_hits"]["fused"] = True
        cached_m = cache.get_stage_metrics("fused", cfg) if cache else None
        if cached_m and "timing" in cached_m:
            summary["timing"]["stage4"] = cached_m["timing"]
    else:
        ekf_cfg = cfg.get("ekf", {})
        eskf = ESKF(
            process_noise_pos=ekf_cfg.get("process_noise_pos", 0.1),
            process_noise_vel=ekf_cfg.get("process_noise_vel", 0.5),
            process_noise_rot=ekf_cfg.get("process_noise_rot", 0.01),
            measurement_noise_pos=ekf_cfg.get("measurement_noise_pos", 0.05),
            measurement_noise_rot=ekf_cfg.get("measurement_noise_rot", 0.01),
        )
        if dataset.timestamps is not None:
            ts_fuse = dataset.timestamps[: len(optimized_poses)]
        else:
            ts_fuse = np.arange(len(optimized_poses)) * 0.1
        timer_s4 = StageTimer("stage4_fusion")
        with timer_s4:
            fused_poses = eskf.run(optimized_poses, ts_fuse)
        summary["timing"]["stage4"] = timer_s4.summary()
        summary["cache_hits"]["fused"] = False

    fused_metrics = _eval_metrics(fused_poses, gt_velo)
    summary["metrics"]["fused"] = fused_metrics
    _log_metrics("Fused", fused_metrics, verbose)

    if cache is not None and not summary["cache_hits"]["fused"]:
        cache.save_fused(
            np.asarray(fused_poses),
            cfg,
            metrics={
                "frame_count": len(fused_poses),
                **fused_metrics,
                "timing": summary["timing"].get("stage4", {}),
            },
        )

    fused_cam = transform_poses_to_camera_frame(fused_poses, Tr)
    KissICPOdometry.save_poses_kitti_format(fused_cam, output_dir / f"poses_fused_{sequence}.txt")

    # --- Stage 4b: Global map master (voxel = master_voxel_size) ---
    if verbose:
        print(f"=== [{sequence}] Stage 4b: Global Map Master (master voxel) ===")
    mapping_cfg = cfg.get("mapping", {})
    master_voxel = float(mapping_cfg.get("master_voxel_size", 0.05))
    working_voxel = float(mapping_cfg.get("voxel_size", 0.15))

    master_cached = cache.load_global_map_master(cfg) if cache else None
    if master_cached is not None:
        master_pcd = master_cached
        if verbose:
            print(f"  [cache hit] master map loaded: {len(master_pcd.points):,} points")
        summary["cache_hits"]["map_master"] = True
        cached_m = cache.get_stage_metrics("map_master", cfg) if cache else None
        if cached_m and "timing" in cached_m:
            summary["timing"]["stage4b"] = cached_m["timing"]
    else:
        builder = MapBuilder(
            voxel_size=master_voxel,
            max_range=float(mapping_cfg.get("max_range", 30.0)),
            downsample_every=int(mapping_cfg.get("downsample_every", 500)),
        )
        timer_s4b = StageTimer("stage4b_map_master")
        with timer_s4b:
            master_pcd = builder.build(dataset, fused_poses)
        summary["timing"]["stage4b"] = timer_s4b.summary()
        if verbose:
            print(f"  Built master map: {len(master_pcd.points):,} points")
        summary["cache_hits"]["map_master"] = False

    summary["metrics"]["map_master"] = {"point_count": len(master_pcd.points)}

    if cache is not None and not summary["cache_hits"]["map_master"]:
        cache.save_global_map_master(
            master_pcd,
            cfg,
            metrics={
                "point_count": len(master_pcd.points),
                "timing": summary["timing"].get("stage4b", {}),
            },
        )

    # --- Stage 5: Working map + feature extraction ---
    if verbose:
        print(f"=== [{sequence}] Stage 5: Semantic Map Assembly ===")

    stage5_cached = cache.load_stage5(cfg) if cache else None
    if stage5_cached is not None:
        working_pcd, clusters, curb_clusters = stage5_cached
        if verbose:
            print(
                f"  [cache hit] Stage 5: "
                f"{len(working_pcd.points):,} points, {len(clusters)} lane clusters, "
                f"{len(curb_clusters)} curb clusters"
            )
        summary["cache_hits"]["stage5"] = True
        # Rehydrate diagnostic counts from the cache metadata (they were
        # computed and stored when stage5 was first built).
        meta_snapshot = cache.metadata_snapshot() if cache else {}
        stage5_meta = (meta_snapshot.get("stage5") or {}).get("metrics", {})
        road_pts_count = stage5_meta.get("road_point_count")
        lane_pts_count = stage5_meta.get("lane_candidate_count")
        curb_pts_count = stage5_meta.get("curb_point_count")
        if stage5_meta.get("timing"):
            summary["timing"]["stage5"] = stage5_meta["timing"]
    else:
        timer_s5 = StageTimer("stage5_features")
        with timer_s5:
            working_pcd = MapBuilder.downsample_existing(master_pcd, working_voxel)
            if verbose:
                print(
                    f"  Downsampled to working voxel={working_voxel}: "
                    f"{len(working_pcd.points):,} points"
                )

            xyz = np.asarray(working_pcd.points)
            intensities = np.asarray(working_pcd.colors)[:, 0]

            road_pts, road_int = extract_road_surface(
                xyz,
                intensities,
                z_min=float(mapping_cfg.get("road_z_min", -2.0)),
                z_max=float(mapping_cfg.get("road_z_max", -1.5)),
            )
            road_pts_count = len(road_pts)
            if verbose:
                print(f"  Road surface points: {road_pts_count:,}")

            lane_pts = extract_lane_markings(
                road_pts,
                road_int,
                intensity_threshold=float(mapping_cfg.get("intensity_threshold", 0.40)),
            )
            lane_pts_count = len(lane_pts)
            if verbose:
                print(f"  Lane marking candidates: {lane_pts_count:,}")

            clusters = cluster_points(
                lane_pts,
                eps=float(mapping_cfg.get("dbscan_eps", 0.7)),
                min_points=int(mapping_cfg.get("dbscan_min_points", 40)),
            )
            if verbose:
                print(f"  Lane marking clusters: {len(clusters)}")

            curb_pts = extract_curbs(
                xyz,
                grid_size=float(mapping_cfg.get("curb_grid_size", 0.30)),
                z_min=float(mapping_cfg.get("curb_z_min", -2.0)),
                z_max=float(mapping_cfg.get("curb_z_max", -1.2)),
                height_min=float(mapping_cfg.get("curb_height_min", 0.10)),
                height_max=float(mapping_cfg.get("curb_height_max", 0.25)),
                road_z_top=float(mapping_cfg.get("curb_road_z_top", -1.55)),
            )
            curb_pts_count = int(curb_pts.shape[0])
            if verbose:
                print(f"  Curb candidate points: {curb_pts_count:,}")

            curb_clusters = cluster_points(
                curb_pts,
                eps=float(mapping_cfg.get("curb_dbscan_eps", 0.5)),
                min_points=int(mapping_cfg.get("curb_dbscan_min_points", 10)),
                trim_k=float(mapping_cfg.get("curb_trim_k", 1.2)),
            )
            if verbose:
                print(f"  Curb clusters: {len(curb_clusters)}")
        summary["timing"]["stage5"] = timer_s5.summary()
        summary["cache_hits"]["stage5"] = False

    cluster_stats = _cluster_size_stats(clusters)
    stage5_metrics = {
        "working_point_count": len(working_pcd.points),
        "road_point_count": road_pts_count,
        "lane_candidate_count": lane_pts_count,
        "curb_point_count": curb_pts_count,
        "curb_cluster_count": len(curb_clusters),
        **cluster_stats,
    }
    summary["metrics"]["stage5"] = stage5_metrics

    if cache is not None and not summary["cache_hits"]["stage5"]:
        cache.save_stage5(
            working_pcd,
            clusters,
            cfg,
            curb_clusters=curb_clusters,
            metrics={
                **stage5_metrics,
                "timing": summary["timing"].get("stage5", {}),
            },
        )

    # Human-readable results dir outputs (overwrite each run).
    MapBuilder.save(working_pcd, output_dir / f"global_map_{sequence}.pcd")
    save_features_geojson(
        clusters, output_dir / f"features_{sequence}.geojson", feature_type="lane_marking"
    )
    save_features_geojson(
        curb_clusters, output_dir / f"curbs_{sequence}.geojson", feature_type="curb"
    )

    # --- Stage 6: Lanelet2 HD Map Export (always runs, cheap) ---
    if verbose:
        print(f"=== [{sequence}] Stage 6: Lanelet2 HD Map Export ===")
    export_cfg = cfg.get("export", {})
    osm_path = output_dir / f"map_{sequence}.osm"
    counts = export_lanelet2_osm(clusters, curb_clusters, osm_path, **export_cfg)
    lane_c = counts["lane"]
    curb_c = counts["curb"]
    if verbose:
        print(
            f"  Lane: thin={lane_c['line_thin']} thick={lane_c['line_thick']} "
            f"area={lane_c['area']} dropped={lane_c['dropped']} "
            f"(of {lane_c['total_input']}) total={lane_c['total_length_m']:.1f}m"
        )
        print(
            f"  Curb: kept={curb_c['kept']} (rescued={curb_c['rescued']}) "
            f"dropped={curb_c['dropped']} (of {curb_c['total_input']}) "
            f"total={curb_c['total_length_m']:.1f}m"
        )

    def _coerce(k: str, v) -> float | int:
        return float(v) if k == "total_length_m" else int(v)

    summary["metrics"]["stage6"] = {
        "lane": {k: _coerce(k, v) for k, v in lane_c.items()},
        "curb": {k: _coerce(k, v) for k, v in curb_c.items()},
    }

    # Dump timing data if any stages were actually computed (not cached)
    if summary["timing"]:
        timing_path = output_dir / f"latency_{sequence}.json"
        timing_path.write_text(json.dumps(summary["timing"], indent=2) + "\n")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="LiDAR SLAM HD Map Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--sequence",
        type=str,
        default=None,
        help="KITTI Odometry sequence id; overrides config",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Limit number of frames to process (for quick testing)",
    )
    parser.add_argument(
        "--force-rebuild",
        type=str,
        default="none",
        choices=_FORCE_REBUILD_CHOICES,
        help=(
            "Invalidate cache entries before running. "
            "'none' (default) uses cache; 'all' clears everything; "
            "a stage name clears that stage and all downstream stages."
        ),
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable cache entirely (bypass cache.enabled=true in config).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    sequence = args.sequence or config["data"]["sequence"]

    cache: LayeredCache | None = None
    cache_cfg = config.get("cache", {})
    if cache_cfg.get("enabled", False) and not args.no_cache:
        cache = LayeredCache(
            root=cache_cfg.get("root", "cache/kitti"),
            sequence=sequence,
        )

    summary = run_pipeline_cached(
        config=config,
        sequence=sequence,
        cache=cache,
        force_rebuild=args.force_rebuild,
        max_frames=args.max_frames,
        output_dir=Path(config.get("output", {}).get("dir", "results")),
        verbose=True,
    )

    print("\nDone.")
    print(f"  Cache hits: {summary['cache_hits']}")


if __name__ == "__main__":
    main()
