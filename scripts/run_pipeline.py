"""Run the pipeline on one KITTI Odometry sequence.

  Stage 1  scans, with the KITTI scan correction (src/data/kitti_loader.py)
  Stage 2  KISS-ICP odometry, and the per-frame degeneracy diagnostic
  Stage 3  pose graph with Scan Context loop closures
  Stage 4  point-cloud map accumulated along the Stage 3 trajectory
  Stage 5  curb extraction from the map
  Stage 6  Lanelet2 OSM export, georeferenced with the GNSS/INS log of the
           KITTI raw drive when it is available (src/export/georef.py)

Stages 2-5 are cached per sequence (src/cache/layered_cache.py); a cache
entry is reused only while the config of the stage and of every stage before
it is unchanged. Outputs go to ``output.dir`` (default ``results/``).

Usage:
  python scripts/run_pipeline.py --sequence 00
  python scripts/run_pipeline.py --sequence 00 --until optimized
  python scripts/run_pipeline.py --sequence 00 --force-rebuild map
  python scripts/run_pipeline.py --kitti-root /data --output-dir out --max-frames 200
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.benchmarks.timing import StageTimer
from src.cache import STAGE_ORDER, LayeredCache
from src.data.kitti_loader import KITTIDataset
from src.export import export_lanelet2_osm
from src.export.georef import georeference_kitti
from src.mapping import (
    MapBuilder,
    cluster_points,
    extract_curbs,
    heights_in_nearest_pose_frame,
    save_features_geojson,
)
from src.odometry.degeneracy import analyzer_from_config, scores_to_array
from src.odometry.kiss_icp_wrapper import (
    KissICPOdometry,
    evaluate_odometry,
    transform_poses_to_camera_frame,
)
from src.optimization.loop_closure import LoopClosureDetector
from src.optimization.pose_graph import PoseGraphOptimizer

# Force-rebuild accepts cache stage names plus two aliases.
_FORCE_REBUILD_CHOICES = ["none", "all", *STAGE_ORDER]
# Stages after which a run may stop early (skipping mapping and export).
_UNTIL_CHOICES = ["odometry", "optimized"]
# Sub-stage timers of Stage 3, stored with its cache entry.
_STAGE3_TIMERS = ("sc_query", "icp_verify", "icp_refine", "icp_retry", "graph_optimize")


def load_config(config_path: str | Path) -> dict:
    """Load YAML configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


@dataclass
class _Run:
    """State shared by the stages of one run."""

    cfg: dict
    sequence: str
    dataset: KITTIDataset
    gt_velo: list[np.ndarray] | None
    cache: LayeredCache | None
    output_dir: Path
    verbose: bool
    summary: dict[str, Any] = field(default_factory=dict)

    def log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def header(self, title: str) -> None:
        self.log(f"=== [{self.sequence}] {title} ===")

    def cache_hit(self, stage: str, timing_keys: tuple[str, ...] = ()) -> None:
        """Record a cache hit and restore the timing stored with the entry."""
        self.summary["cache_hits"][stage] = True
        stored = self.cache.get_stage_metrics(stage, self.cfg) or {}
        for k, key in enumerate(timing_keys):
            # Entries written before the per-stage keys keep the stage's own
            # timer under "timing".
            value = stored.get(key, stored.get("timing") if k == 0 else None)
            if value:
                self.summary["timing"][key] = value

    def evaluate(self, label: str, poses: list[np.ndarray]) -> dict[str, float]:
        """APE (first-frame aligned) and RPE against KITTI ground truth, if any."""
        if self.gt_velo is None:
            return {}
        n = min(len(poses), len(self.gt_velo))
        result = evaluate_odometry(poses[:n], self.gt_velo[:n])
        m = {"ape_rmse": float(result["ape"]["rmse"]), "ape_mean": float(result["ape"]["mean"])}
        if "rpe" in result:
            m["rpe_rmse"] = float(result["rpe"]["rmse"])
        self.log(f"  {label} APE RMSE {m['ape_rmse']:.4f} m, mean {m['ape_mean']:.4f} m")
        if "rpe_rmse" in m:
            self.log(f"  {label} RPE RMSE {m['rpe_rmse']:.4f} m")
        return m

    def save_poses(self, poses: list[np.ndarray], name: str) -> None:
        """KITTI-format poses (camera frame, as the devkit expects)."""
        Tr = self.dataset.calibration["Tr"] if self.dataset.calibration else np.eye(4)
        cam = transform_poses_to_camera_frame(poses, Tr)
        KissICPOdometry.save_poses_kitti_format(cam, self.output_dir / name)


# ---------------------------------------------------------------------------
# Stage 2: odometry
# ---------------------------------------------------------------------------


def _write_degeneracy_csv(scores: np.ndarray, path: Path, threshold: float) -> None:
    """Per-frame degeneracy scores, with a flag for ``cond_number > threshold``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(
            "frame_idx,cond_number,lambda_min,lambda_max,n_corr,eig_dx,eig_dy,eig_dz,degenerate\n"
        )
        for i in range(scores.shape[0]):
            cond = scores[i, 0]
            flag = 1 if (np.isfinite(cond) and cond > threshold) else 0
            f.write(
                f"{i},{cond:.6e},{scores[i, 1]:.6e},{scores[i, 2]:.6e},{int(scores[i, 3])},"
                f"{scores[i, 4]:.6f},{scores[i, 5]:.6f},{scores[i, 6]:.6f},{flag}\n"
            )


def _degeneracy_stats(scores: np.ndarray, threshold: float) -> dict[str, float]:
    cond = scores[:, 0]
    finite = cond[np.isfinite(cond)]
    if finite.size == 0:
        return {"count": 0}
    return {
        "count": int(finite.size),
        "p50": float(np.percentile(finite, 50)),
        "p95": float(np.percentile(finite, 95)),
        "p99": float(np.percentile(finite, 99)),
        "max": float(finite.max()),
        "n_above_threshold": int((finite > threshold).sum()),
        "threshold": float(threshold),
    }


def stage_odometry(run: _Run) -> list[np.ndarray]:
    """KISS-ICP poses in the Velodyne frame of the first scan."""
    run.header("Stage 2: KISS-ICP odometry")
    cfg, cache = run.cfg, run.cache
    deg_cfg = cfg.get("degeneracy", {}) or {}
    with_degeneracy = bool(deg_cfg.get("enabled", False))

    cached = cache.load_odometry(cfg) if cache else None
    scores = None
    if cached is not None and with_degeneracy:
        # The scores need the same scan stream; recompute Stage 2 if they are stale.
        scores = cache.load_degeneracy(cfg)
        if scores is None:
            run.log("  degeneracy scores missing or stale: rerunning Stage 2")
            cached = None

    if cached is not None:
        poses = list(cached[0])
        run.log(f"  [cache hit] {len(poses)} poses")
        run.cache_hit("odometry", ("stage2",))
    else:
        k = cfg.get("kiss_icp", {})
        odom = KissICPOdometry(
            max_range=k.get("max_range", 100.0),
            min_range=k.get("min_range", 5.0),
            voxel_size=k.get("voxel_size", 1.0),
        )
        analyzer = analyzer_from_config(deg_cfg) if with_degeneracy else None
        timer = StageTimer("stage2_odometry")
        result = odom.run(run.dataset, timer=timer, degeneracy_analyzer=analyzer)
        if analyzer is None:
            poses = result
        else:
            poses, score_list = result
            scores = scores_to_array(score_list)
        run.summary["timing"]["stage2"] = timer.summary()
        run.summary["cache_hits"]["odometry"] = False

    metrics = run.evaluate("Odometry", poses)
    run.summary["metrics"]["odometry"] = metrics
    if cache is not None and not run.summary["cache_hits"]["odometry"]:
        ts = run.dataset.timestamps
        timestamps = (
            np.asarray(ts[: len(poses)], dtype=np.float64)
            if ts is not None
            else np.arange(len(poses), dtype=np.float64) * 0.1
        )
        stored = {"frame_count": len(poses), **metrics}
        stored["stage2"] = run.summary["timing"].get("stage2", {})
        cache.save_odometry(np.asarray(poses), timestamps, cfg, metrics=stored)
        if scores is not None:
            cache.save_degeneracy(scores, cfg)

    if scores is not None:
        threshold = float(deg_cfg.get("cond_threshold", 100.0))
        _write_degeneracy_csv(scores, run.output_dir / f"degeneracy_{run.sequence}.csv", threshold)
        st = _degeneracy_stats(scores, threshold)
        run.summary["metrics"]["degeneracy"] = st
        run.log(
            f"  degeneracy: cond p50 {st.get('p50', float('nan')):.2f}, "
            f"p95 {st.get('p95', float('nan')):.2f}, "
            f"{st.get('n_above_threshold', 0)}/{st.get('count', 0)} frames above {threshold}"
        )

    run.save_poses(poses, f"poses_{run.sequence}.txt")
    return poses


# ---------------------------------------------------------------------------
# Stage 3: pose graph
# ---------------------------------------------------------------------------


def stage_optimize(run: _Run, poses: list[np.ndarray]) -> list[np.ndarray]:
    """Stage 2 poses with Scan Context loop closures, optimized with GTSAM."""
    run.header("Stage 3: pose graph")
    cfg, cache = run.cfg, run.cache
    timing_keys = ("stage3", *(f"stage3_{k}" for k in _STAGE3_TIMERS))

    cached = cache.load_optimized(cfg) if cache else None
    if cached is not None:
        optimized = list(cached)
        run.log(f"  [cache hit] {len(optimized)} poses")
        run.cache_hit("optimized", timing_keys)
        stored = cache.get_stage_metrics("optimized", cfg) or {}
        run.summary["loop_closures"] = stored.get("loop_closures")
    else:
        lc_cfg = cfg.get("loop_closure", {})
        detector = LoopClosureDetector.from_config(lc_cfg)
        timer = StageTimer("stage3_optimization")
        timer_graph = StageTimer("stage3_graph_optimize")
        with timer:
            closures = detector.detect(poses, dataset=run.dataset)
            run.log(f"  {len(closures)} loop closures")
            with timer_graph:
                optimizer = PoseGraphOptimizer.from_config(cfg.get("gtsam", {}), lc_cfg)
                optimizer.build_graph(poses)
                for i, j, rel_pose in closures:
                    optimizer.add_loop_closure(i, j, rel_pose)
                optimized = optimizer.optimize()
        t = run.summary["timing"]
        t["stage3"] = timer.summary()
        t["stage3_sc_query"] = detector.sc_query_timer.summary()
        t["stage3_icp_verify"] = detector.icp_verify_timer.summary()
        t["stage3_icp_refine"] = detector.icp_refine_timer.summary()
        t["stage3_icp_retry"] = detector.icp_retry_timer.summary()
        t["stage3_graph_optimize"] = timer_graph.summary()
        run.summary["cache_hits"]["optimized"] = False
        run.summary["loop_closures"] = len(closures)

    metrics = run.evaluate("Optimized", optimized)
    run.summary["metrics"]["optimized"] = metrics
    if cache is not None and not run.summary["cache_hits"]["optimized"]:
        stored = {
            "frame_count": len(optimized),
            "loop_closures": run.summary["loop_closures"],
            **metrics,
            **{k: run.summary["timing"].get(k, {}) for k in timing_keys},
        }
        cache.save_optimized(np.asarray(optimized), cfg, metrics=stored)

    run.save_poses(optimized, f"poses_optimized_{run.sequence}.txt")
    return optimized


# ---------------------------------------------------------------------------
# Stage 4: map
# ---------------------------------------------------------------------------


def stage_map(run: _Run, optimized: list[np.ndarray]):
    """Global map at ``mapping.master_voxel_size`` from the Stage 3 poses."""
    run.header("Stage 4: map")
    cfg, cache = run.cfg, run.cache
    m = cfg.get("mapping", {})

    pcd = cache.load_map(cfg) if cache else None
    if pcd is not None:
        run.log(f"  [cache hit] {len(pcd.points):,} points")
        run.cache_hit("map", ("map",))
    else:
        builder = MapBuilder(
            voxel_size=float(m.get("master_voxel_size", 0.10)),
            max_range=float(m.get("max_range", 30.0)),
            downsample_every=int(m.get("downsample_every", 500)),
        )
        timer = StageTimer("map")
        with timer:
            pcd = builder.build(run.dataset, optimized)
        run.log(f"  {len(pcd.points):,} points")
        run.summary["timing"]["map"] = timer.summary()
        run.summary["cache_hits"]["map"] = False
        if cache is not None:
            cache.save_map(
                pcd,
                cfg,
                metrics={"point_count": len(pcd.points), "map": run.summary["timing"]["map"]},
            )
    run.summary["metrics"]["map"] = {"point_count": len(pcd.points)}
    return pcd


# ---------------------------------------------------------------------------
# Stage 5: curbs
# ---------------------------------------------------------------------------


def stage_curbs(run: _Run, master_pcd, optimized: list[np.ndarray]):
    """Working-resolution map and the curb clusters extracted from it."""
    run.header("Stage 5: curbs")
    cfg, cache = run.cfg, run.cache
    m = cfg.get("mapping", {})

    cached = cache.load_curbs(cfg) if cache else None
    if cached is not None:
        working, curbs = cached
        run.log(f"  [cache hit] {len(working.points):,} points, {len(curbs)} curb clusters")
        run.cache_hit("curbs", ("curbs",))
        stored = cache.get_stage_metrics("curbs", cfg) or {}
        candidates = stored.get("curb_point_count")
    else:
        timer = StageTimer("curbs")
        with timer:
            working = MapBuilder.downsample_existing(master_pcd, float(m.get("voxel_size", 0.15)))
            xyz = np.asarray(working.points)
            # Heights above the sensor of the nearest pose, not z in the map
            # frame: the road under the car is at about -1.73 m only there
            # (docs/map-quality.md).
            heights = heights_in_nearest_pose_frame(xyz, np.asarray(optimized))
            points = extract_curbs(
                xyz,
                grid_size=float(m.get("curb_grid_size", 0.30)),
                z_min=float(m.get("curb_z_min", -2.0)),
                z_max=float(m.get("curb_z_max", -1.2)),
                height_min=float(m.get("curb_height_min", 0.10)),
                height_max=float(m.get("curb_height_max", 0.25)),
                road_z_top=float(m.get("curb_road_z_top", -1.55)),
                heights=heights,
            )
            candidates = int(points.shape[0])
            curbs = cluster_points(
                points,
                eps=float(m.get("curb_dbscan_eps", 0.6)),
                min_points=int(m.get("curb_dbscan_min_points", 10)),
                trim_k=float(m.get("curb_trim_k", 1.2)),
            )
        run.log(
            f"  {len(working.points):,} points at the working voxel, "
            f"{candidates:,} curb candidates, {len(curbs)} clusters"
        )
        run.summary["timing"]["curbs"] = timer.summary()
        run.summary["cache_hits"]["curbs"] = False

    metrics = {
        "working_point_count": len(working.points),
        "curb_point_count": candidates,
        "curb_cluster_count": len(curbs),
    }
    run.summary["metrics"]["curbs"] = metrics
    if cache is not None and not run.summary["cache_hits"]["curbs"]:
        cache.save_curbs(
            working, curbs, cfg, metrics={**metrics, "curbs": run.summary["timing"]["curbs"]}
        )

    MapBuilder.save(working, run.output_dir / f"global_map_{run.sequence}.pcd")
    save_features_geojson(curbs, run.output_dir / f"curbs_{run.sequence}.geojson")
    return working, curbs


# ---------------------------------------------------------------------------
# Stage 6: Lanelet2 export
# ---------------------------------------------------------------------------


def stage_export(run: _Run, curbs: list[np.ndarray], optimized: list[np.ndarray]) -> None:
    run.header("Stage 6: Lanelet2 export")
    export_cfg = dict(run.cfg.get("export", {}))
    raw_root = export_cfg.pop("kitti_raw_root", None)
    georef = None
    if raw_root and run.dataset.timestamps is not None:
        georef = georeference_kitti(
            raw_root,
            run.sequence,
            np.asarray(optimized),
            run.dataset.timestamps[: len(optimized)],
        )
    if georef is None:
        run.log("  No GNSS log for this sequence: map placed at the nominal origin")
    else:
        run.log(
            f"  Georeferenced to EPSG:{georef.epsg} with {georef.n_control} GNSS epochs "
            f"(time offset {georef.time_offset_s:.3f} s): residual horizontal RMS "
            f"{georef.residual_horizontal_rms_m:.2f} m, vertical RMS "
            f"{georef.residual_vertical_rms_m:.2f} m"
        )
    osm_path = run.output_dir / f"map_{run.sequence}.osm"
    counts = export_lanelet2_osm(curbs, osm_path, georef=georef, **export_cfg)["curb"]
    run.log(
        f"  Curbs: kept {counts['kept']} (rescued {counts['rescued']}), dropped "
        f"{counts['dropped']} of {counts['total_input']}, {counts['total_length_m']:.1f} m"
    )
    run.summary["metrics"]["export"] = {
        k: float(v) if k == "total_length_m" else int(v) for k, v in counts.items()
    }


# ---------------------------------------------------------------------------


def run_pipeline_cached(
    config: dict,
    sequence: str,
    cache: LayeredCache | None = None,
    force_rebuild: str = "none",
    max_frames: int | None = None,
    output_dir: Path | None = None,
    verbose: bool = True,
    until: str | None = None,
) -> dict[str, Any]:
    """Run the pipeline on one KITTI sequence, reusing fresh cache entries.

    Args:
        config: Parsed YAML config dict.
        sequence: KITTI Odometry sequence id, e.g. ``"00"``.
        cache: Optional :class:`LayeredCache`. If ``None``, caching is off.
        force_rebuild: ``"none"`` uses the cache wherever valid, ``"all"``
            clears it; a stage name clears that stage and all later ones.
        max_frames: Optional frame cap for quick checks. Disables the cache,
            because the frame count is not part of the cache key.
        output_dir: Where the outputs go; defaults to ``config["output"]["dir"]``.
        verbose: Print per-stage progress lines.
        until: ``"odometry"`` or ``"optimized"``: return after that stage.

    Returns:
        Summary dict: cache hits, metrics and timing per stage.
    """
    cfg = {**config, "data": {**config.get("data", {}), "sequence": sequence}}
    if output_dir is None:
        output_dir = Path(cfg.get("output", {}).get("dir", "results"))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if max_frames is not None and cache is not None:
        if verbose:
            print("  [cache disabled] max_frames set: not reading or writing the cache")
        cache = None
    if cache is not None and force_rebuild != "none":
        cache.invalidate(force_rebuild)

    if verbose:
        print(f"\n=== [{sequence}] Stage 1: KITTI scans ===")
    dataset = KITTIDataset.from_config(cfg["data"], sequence)
    if max_frames is not None and max_frames < len(dataset):
        dataset.scan_files = dataset.scan_files[:max_frames]
    if len(dataset) == 0:
        raise RuntimeError(f"No scans found for sequence {sequence}")
    if verbose:
        print(f"  {len(dataset)} frames")

    gt_velo = None
    if dataset.poses is not None:
        Tr = dataset.calibration["Tr"] if dataset.calibration else np.eye(4)
        Tr_inv = np.linalg.inv(Tr)
        gt_velo = [Tr_inv @ g @ Tr for g in dataset.poses]

    run = _Run(cfg, sequence, dataset, gt_velo, cache, output_dir, verbose)
    run.summary = {
        "sequence": sequence,
        "frame_count": len(dataset),
        "has_gt": gt_velo is not None,
        "cache_hits": {},
        "metrics": {},
        "timing": {},
    }

    poses = stage_odometry(run)
    if until != "odometry":
        optimized = stage_optimize(run, poses)
        if until != "optimized":
            master = stage_map(run, optimized)
            _, curbs = stage_curbs(run, master, optimized)
            stage_export(run, curbs, optimized)

    if run.summary["timing"]:
        path = output_dir / f"latency_{sequence}.json"
        path.write_text(json.dumps(run.summary["timing"], indent=2) + "\n")
    return run.summary


def main() -> None:
    parser = argparse.ArgumentParser(description="LiDAR SLAM to Lanelet2 pipeline")
    parser.add_argument("--config", default="configs/default.yaml", help="configuration file")
    parser.add_argument("--sequence", default=None, help="KITTI sequence; overrides config")
    parser.add_argument("--max-frames", type=int, default=None, help="process the first N frames")
    parser.add_argument(
        "--force-rebuild",
        default="none",
        choices=_FORCE_REBUILD_CHOICES,
        help="'all' clears the cache; a stage name clears that stage and all later ones",
    )
    parser.add_argument(
        "--until", default=None, choices=_UNTIL_CHOICES, help="stop after this stage"
    )
    parser.add_argument("--no-cache", action="store_true", help="neither read nor write the cache")
    parser.add_argument("--kitti-root", default=None, help="KITTI Odometry root; overrides config")
    parser.add_argument("--output-dir", default=None, help="output directory; overrides config")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.kitti_root:
        config["data"]["kitti_root"] = args.kitti_root
    sequence = args.sequence or config["data"]["sequence"]
    cache = None
    cache_cfg = config.get("cache", {})
    if cache_cfg.get("enabled", False) and not args.no_cache:
        cache = LayeredCache(root=cache_cfg.get("root", "cache/kitti"), sequence=sequence)

    summary = run_pipeline_cached(
        config=config,
        sequence=sequence,
        cache=cache,
        force_rebuild=args.force_rebuild,
        max_frames=args.max_frames,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        verbose=True,
        until=args.until,
    )
    print("\nDone.")
    print(f"  Cache hits: {summary['cache_hits']}")


if __name__ == "__main__":
    main()
