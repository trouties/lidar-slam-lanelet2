#!/usr/bin/env python3
"""How good is the map? The measurements behind docs/map-quality.md.

  crispness  Road-surface thickness on the driven lane and occupied-voxel
             count of local maps built from different poses. Measures how
             much pose noise blurs the map, without ground truth.
  features   Where the curb clusters, and the clusters of the lane-marking
             channel that was removed from the pipeline, lie relative to the
             driven path; the road's cross-section.
  lanesweep  Whether a painted line can pass the lane-marking clustering, and
             what a smaller DBSCAN min_points gives.
  revisit    Where the drive passes the same street twice, how far apart the
             two copies of it are in the map.
  georef     The rigid fit of a trajectory to the GNSS/INS positions in UTM
             (the georeferencing of the map), and how the KITTI ground truth
             relates to the same positions.
  lanelet2   Load the exported map with Lanelet2 and check that it carries
             the intended geometry.
  worldband  How many frames a road band applied to world z would cover.
  figure     Map figures for the documentation.

Ground truth is used only to find revisits and to start the revisit
registration from a pose inside its capture range; the offsets themselves
are measured by registering the scans.

Usage:
  python scripts/run_pipeline.py --sequence 00
  python -m scripts.map_study crispness --sequences 00 --out-dir benchmarks/map
  python -m scripts.map_study revisit --sequences 00 02 05 06 07 08 09 --out-dir benchmarks/map
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from scripts.run_pipeline import load_config
from src.cache import LayeredCache
from src.data.kitti_loader import KITTIDataset
from src.data.kitti_raw import ODOM_TO_RAW
from src.export.georef import (
    georeference_kitti,
    gnss_at,
    load_imu_lever_arm,
    load_oxts_monotonic,
    oxts_dir_for_sequence,
)
from src.export.lanelet2_export import _pca_stats
from src.mapping import MapBuilder, cluster_points, heights_in_nearest_pose_frame

SEQUENCES_WITH_LOOPS = ("00", "02", "05", "06", "07", "08", "09")
POSE_SOURCES = ("odometry", "optimized", "gt")
SOURCE_LABEL = {
    "odometry": "Stage 2 (odometry)",
    "optimized": "Stage 3 (pose graph)",
    "gt": "KITTI ground truth",
}
# Height of the KITTI Velodyne above the road (KITTI setup description).
SENSOR_HEIGHT_M = 1.73

# The lane-marking channel that this study took out of the pipeline
# (docs/map-quality.md, section 4): map points in the road band (height in the
# frame of the nearest pose) with a reflectance of at least 0.40, clustered
# with DBSCAN. A cluster counts as a line when it passes the line test of the
# export (``_is_line``).
ROAD_BAND_M = (-2.0, -1.5)
LANE_REFLECTANCE_MIN = 0.40
LANE_DBSCAN_EPS_M = 0.7
LANE_DBSCAN_MIN_POINTS = 40


def _sequence_config(cfg: dict, sequence: str) -> dict:
    return {**cfg, "data": {**cfg.get("data", {}), "sequence": sequence}}


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


class Sequence:
    """Scans, timestamps and every pose source of one KITTI sequence."""

    def __init__(self, cfg: dict, sequence: str) -> None:
        self.cfg = _sequence_config(cfg, sequence)
        self.seq = sequence
        self.ds = KITTIDataset.from_config(self.cfg["data"], sequence)
        Tr = self.ds.calibration["Tr"]
        Tr_inv = np.linalg.inv(Tr)
        self.gt = np.array([Tr_inv @ g @ Tr for g in self.ds.poses])
        self.timestamps = np.asarray(self.ds.timestamps)
        self.cache = LayeredCache(self.cfg.get("cache", {}).get("root", "cache/kitti"), sequence)
        self._poses: dict[str, np.ndarray] = {"gt": self.gt}

    def poses(self, source: str) -> np.ndarray:
        if source not in self._poses:
            if source == "odometry":
                cached = self.cache.load_odometry(self.cfg)
                cached = None if cached is None else cached[0]
            elif source == "optimized":
                cached = self.cache.load_optimized(self.cfg)
            else:
                raise ValueError(source)
            if cached is None:
                raise SystemExit(
                    f"{self.seq}: no fresh '{source}' cache for this config; run "
                    f"`python scripts/run_pipeline.py --sequence {self.seq}` first"
                )
            self._poses[source] = np.asarray(cached)
        return self._poses[source]

    def scan(self, k: int) -> np.ndarray:
        return self.ds[k][0]


# ---------------------------------------------------------------------------
# crispness
# ---------------------------------------------------------------------------


def _road_profile(points: np.ndarray, poses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Height in the nearest pose frame and signed lateral offset from it."""
    t = poses[:, :3, 3]
    _, idx = cKDTree(t[:, :2]).query(points[:, :2], workers=-1)
    d = points - t[idx]
    h = np.einsum("ij,ij->i", d, poses[idx, :3, 2])
    lat = np.einsum("ij,ij->i", d, poses[idx, :3, 1])
    return h, lat


def cmd_crispness(args: argparse.Namespace) -> None:
    """Thickness of the road under the car in local maps from different poses.

    Each window is a stretch of consecutive scans, so it contains a single
    pass and no loop closure: the spread of the road height directly under
    the driven path then comes from sensor noise, road texture and how
    consistently the poses place the scans, not from revisits.
    """
    cfg = load_config(args.config)
    mcfg = cfg.get("mapping", {})
    rows = []
    for seq in args.sequences:
        sq = Sequence(cfg, seq)
        n = len(sq.ds)
        starts = np.linspace(0, n - args.window, args.n_windows).astype(int)
        for s in starts:
            frames = range(s, s + args.window)
            scans = [sq.scan(k) for k in frames]
            for source in args.sources:
                P = sq.poses(source)
                b = MapBuilder(
                    voxel_size=float(mcfg.get("master_voxel_size", 0.10)),
                    max_range=float(mcfg.get("max_range", 30.0)),
                    downsample_every=500,
                )
                for scan, k in zip(scans, frames):
                    b.add_frame(scan, P[k])
                pts = np.asarray(b.finalize().points)
                h, lat = _road_profile(pts, P[s : s + args.window])
                road = (np.abs(lat) < args.lane_halfwidth) & (np.abs(h + SENSOR_HEIGHT_M) < 0.5)
                q = np.percentile(h[road], [5, 25, 50, 75, 95])
                rows.append(
                    {
                        "sequence": seq,
                        "first_frame": int(s),
                        "frames": args.window,
                        "source": source,
                        "occupied_voxels": int(pts.shape[0]),
                        "road_points": int(road.sum()),
                        "road_height_median_m": round(float(q[2]), 4),
                        "road_height_iqr_m": round(float(q[3] - q[1]), 4),
                        "road_height_p5_m": round(float(q[0]), 4),
                        "road_height_p95_m": round(float(q[4]), 4),
                    }
                )
                print("  ".join(f"{k}={v}" for k, v in rows[-1].items()))
    if args.out_dir:
        _write_csv(Path(args.out_dir) / "crispness.csv", rows)


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------


def _load_geojson_clusters(path: Path) -> list[np.ndarray]:
    data = json.loads(Path(path).read_text())
    return [np.asarray(f["geometry"]["coordinates"], dtype=float) for f in data["features"]]


def cmd_features(args: argparse.Namespace) -> None:
    """Lateral position of the Stage 5 clusters, and the road's cross-section.

    Heights are above the road under the car (nearest pose frame plus the
    sensor height).
    """
    cfg = load_config(args.config)
    results = Path(cfg.get("output", {}).get("dir", "results"))
    rows = []
    edges = [0.0, 2.0, 4.0, 6.0, 8.0, np.inf]
    for seq in args.sequences:
        sq = Sequence(cfg, seq)
        P = sq.poses("optimized")
        kinds = {
            "lane_marking": _lane_clusters(sq, LANE_DBSCAN_MIN_POINTS),
            "curb": _load_geojson_clusters(results / f"curbs_{seq}.geojson"),
        }
        for kind, clusters in kinds.items():
            if not clusters:
                continue
            cent = np.array([c.mean(axis=0) for c in clusters])
            npts = np.array([len(c) for c in clusters])
            h, lat = _road_profile(cent, P)
            alat = np.abs(lat)
            row = {
                "sequence": seq,
                "feature": kind,
                "clusters": len(clusters),
                "points": int(npts.sum()),
                "lateral_offset_median_m": round(float(np.median(alat)), 2),
            }
            for lo, hi in zip(edges[:-1], edges[1:]):
                m = (alat >= lo) & (alat < hi)
                label = f"{int(lo)}-{int(hi)}m" if np.isfinite(hi) else f">{int(lo)}m"
                row[f"clusters_{label}"] = int(m.sum())
            rows.append(row)
            print("  ".join(f"{k}={v}" for k, v in row.items()))
    if args.out_dir:
        _write_csv(Path(args.out_dir) / "features.csv", rows)

    # Road cross-section: heights inside the road band by distance from the path.
    # The band also holds sidewalk tops, so the lower quantiles follow the road.
    prof = []
    for seq in args.sequences:
        sq = Sequence(cfg, seq)
        pts, _ = _working_map(sq)
        h, lat = _road_profile(pts, sq.poses("optimized"))
        band = (h >= ROAD_BAND_M[0]) & (h <= ROAD_BAND_M[1])
        for lo in range(0, 8):
            m = band & (np.abs(lat) >= lo) & (np.abs(lat) < lo + 1)
            q = np.percentile(h[m], [10, 25, 50]) + SENSOR_HEIGHT_M
            prof.append(
                {
                    "sequence": seq,
                    "lateral_from_m": lo,
                    "lateral_to_m": lo + 1,
                    "points": int(m.sum()),
                    "height_p10_m": round(float(q[0]), 3),
                    "height_p25_m": round(float(q[1]), 3),
                    "height_median_m": round(float(q[2]), 3),
                }
            )
            print("  ".join(f"{k}={v}" for k, v in prof[-1].items()))
    if args.out_dir:
        _write_csv(Path(args.out_dir) / "road_profile.csv", prof)


# ---------------------------------------------------------------------------
# revisit
# ---------------------------------------------------------------------------


def _revisit_epochs(gt: np.ndarray, min_gap: int, radius: float, stride: int) -> np.ndarray:
    """(i, j) pairs where frame j passes within ``radius`` of an earlier frame i.

    Frames closer than ``min_gap`` in index belong to the same pass. For each
    later frame (every ``stride``-th) the nearest earlier frame is taken.
    """
    xy = gt[:, :2, 3]
    tree = cKDTree(xy)
    pairs = []
    for j in range(min_gap, len(xy), stride):
        cand = [i for i in tree.query_ball_point(xy[j], radius) if i <= j - min_gap]
        if cand:
            i = min(cand, key=lambda k: np.linalg.norm(xy[k] - xy[j]))
            pairs.append((i, j))
    return np.array(pairs, dtype=int).reshape(-1, 2)


def _window_scans(sq: Sequence, center: int, half: int) -> dict[int, np.ndarray]:
    lo, hi = max(0, center - half), min(len(sq.ds) - 1, center + half)
    scans = {}
    for k in range(lo, hi + 1):
        pts = sq.scan(k)[:, :3]
        r = np.linalg.norm(pts, axis=1)
        scans[k] = pts[(r > 3.0) & (r < 40.0)]
    return scans


def _local_map(scans: dict[int, np.ndarray], P: np.ndarray, center: int, voxel: float):
    """The scans merged with poses ``P``, in the frame of scan ``center``."""
    inv = np.linalg.inv(P[center])
    clouds = []
    for k, pts in scans.items():
        T = inv @ P[k]
        clouds.append(pts @ T[:3, :3].T + T[:3, 3])
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.concatenate(clouds)))
    return pcd.voxel_down_sample(voxel)


def _register(src, tgt, init: np.ndarray, stages) -> tuple[np.ndarray, float, float]:
    """Coarse-to-fine point-to-plane ICP, the Stage 3 loop-closure refinement."""
    reg = o3d.pipelines.registration
    T = init
    res = None
    for voxel, corr in stages:
        s = src.voxel_down_sample(voxel)
        t = tgt.voxel_down_sample(voxel)
        for c in (s, t):
            c.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=3 * voxel, max_nn=30))
        res = reg.registration_icp(
            s,
            t,
            corr,
            T,
            reg.TransformationEstimationPointToPlane(),
            reg.ICPConvergenceCriteria(max_iteration=50),
        )
        T = res.transformation
    return np.asarray(T), float(res.fitness), float(res.inlier_rmse)


def cmd_revisit(args: argparse.Namespace) -> None:
    """Offset between the two copies of a street in maps built from each pose source.

    For each revisit epoch (i, j), the scans around i and around j are merged
    into two local maps with the poses under test, and the second is
    registered onto the first. The registration gives the relative pose the
    scans actually have; the poses claim ``inv(P_i) P_j``. The difference,
    evaluated at scan j's origin, is how far the second copy of the street
    is displaced from the first in that map.
    """
    cfg = load_config(args.config)
    stages = [tuple(s) for s in cfg["loop_closure"]["refine"]["stages"]]
    rows, epochs_out = [], []
    for seq in args.sequences:
        sq = Sequence(cfg, seq)
        pairs = _revisit_epochs(sq.gt, args.min_gap, args.radius, args.stride)
        print(f"{seq}: {len(pairs)} revisit epochs")
        offsets = {s: [] for s in args.sources}
        for i, j in pairs:
            scans_i = _window_scans(sq, i, args.half_window)
            scans_j = _window_scans(sq, j, args.half_window)
            gt_rel = np.linalg.inv(sq.gt[i]) @ sq.gt[j]
            for source in args.sources:
                P = sq.poses(source)
                A = _local_map(scans_i, P, i, args.voxel)
                B = _local_map(scans_j, P, j, args.voxel)
                claimed = np.linalg.inv(P[i]) @ P[j]
                best = None
                # Start from the poses under test and from the ground truth; keep the
                # better fit. The measurement is the converged registration.
                inits = (claimed,) if source == "gt" else (claimed, gt_rel)
                for init in inits:
                    T, fit, rmse = _register(B, A, init, stages)
                    if best is None or (fit, -rmse) > (best[1], -best[2]):
                        best = (T, fit, rmse)
                T, fit, rmse = best
                d = T[:3, 3] - claimed[:3, 3]  # in scan i's frame (x forward, z up)
                dR = T[:3, :3] @ claimed[:3, :3].T
                ang = float(np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))))
                ok = fit >= args.min_fitness
                offsets[source].append((np.hypot(d[0], d[1]), abs(d[2]), ang, ok))
                epochs_out.append(
                    {
                        "sequence": seq,
                        "i": int(i),
                        "j": int(j),
                        "source": source,
                        "horizontal_m": round(float(np.hypot(d[0], d[1])), 4),
                        "vertical_m": round(float(d[2]), 4),
                        "rotation_deg": round(ang, 4),
                        "fitness": round(fit, 3),
                        "inlier_rmse_m": round(rmse, 4),
                    }
                )
        for source in args.sources:
            o = np.array(offsets[source], dtype=float).reshape(-1, 4)
            ok = o[:, 3] > 0
            row = {
                "sequence": seq,
                "source": source,
                "epochs": len(o),
                "registered": int(ok.sum()),
                "horizontal_median_m": round(float(np.median(o[ok, 0])), 3) if ok.any() else "",
                "horizontal_p95_m": round(float(np.percentile(o[ok, 0], 95)), 3)
                if ok.any()
                else "",
                "vertical_median_m": round(float(np.median(o[ok, 1])), 3) if ok.any() else "",
                "vertical_p95_m": round(float(np.percentile(o[ok, 1], 95)), 3) if ok.any() else "",
                "rotation_median_deg": round(float(np.median(o[ok, 2])), 3) if ok.any() else "",
            }
            rows.append(row)
            print("  " + "  ".join(f"{k}={v}" for k, v in row.items() if k != "sequence"))
    if args.out_dir:
        _write_csv(Path(args.out_dir) / "revisit.csv", rows)
        _write_csv(Path(args.out_dir) / "revisit_epochs.csv", epochs_out)


# ---------------------------------------------------------------------------
# georef
# ---------------------------------------------------------------------------


def _kitti_mercator(lat: np.ndarray, lon: np.ndarray, lat0: float) -> tuple[np.ndarray, ...]:
    """The projection the KITTI devkit uses to turn OxTS positions into poses.

    A spherical Mercator (radius 6378137 m) scaled by cos(lat0), applied to
    WGS84 geodetic coordinates (``latlonToMercator`` in the devkit).
    """
    er, s = 6378137.0, np.cos(np.radians(lat0))
    return s * er * np.radians(lon), s * er * np.log(np.tan(np.radians(90.0 + lat) / 2.0))


def _rigid_residuals(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    from evo.core.geometry import umeyama_alignment

    r, t, _ = umeyama_alignment(src.T, dst.T, with_scale=False)
    return src @ r.T + t - dst


def cmd_georef(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    raw_root = Path(cfg["export"]["kitti_raw_root"]).expanduser()
    rows = []
    for seq in args.sequences:
        sq = Sequence(cfg, seq)
        if oxts_dir_for_sequence(raw_root, seq) is None:
            print(f"{seq}: no OxTS log under {raw_root}, skipped")
            continue
        for source in args.sources:
            g = georeference_kitti(raw_root, seq, sq.poses(source), sq.timestamps)
            rows.append(
                {
                    "sequence": seq,
                    "source": source,
                    "projection": "UTM",
                    "time_offset_s": g.time_offset_s,
                    "control_points": g.n_control,
                    "horizontal_rms_m": g.residual_horizontal_rms_m,
                    "horizontal_max_m": g.residual_horizontal_max_m,
                    "vertical_rms_m": g.residual_vertical_rms_m,
                }
            )
            print("  ".join(f"{k}={v}" for k, v in rows[-1].items()))

        # The ground truth against the same GNSS positions, in UTM and in the
        # devkit's Mercator, at the time offset fitted for the ground truth.
        g = georeference_kitti(raw_root, seq, sq.gt, sq.timestamps)
        data, t_ox = load_oxts_monotonic(oxts_dir_for_sequence(raw_root, seq))
        lat, lon, alt, rtk = gnss_at(data, t_ox, sq.timestamps, g.time_offset_s)
        lever = load_imu_lever_arm(raw_root, ODOM_TO_RAW[seq][0])
        p_unit = np.einsum("nij,j->ni", sq.gt[:, :3, :3], lever) + sq.gt[:, :3, 3]
        mx, my = _kitti_mercator(lat, lon, float(data[0, 0]))
        r = _rigid_residuals(p_unit[rtk], np.column_stack([mx, my, alt])[rtk])
        h = np.hypot(r[:, 0], r[:, 1])
        rows.append(
            {
                "sequence": seq,
                "source": "gt",
                "projection": "KITTI devkit Mercator",
                "time_offset_s": g.time_offset_s,
                "control_points": len(h),
                "horizontal_rms_m": round(float(np.sqrt(np.mean(h**2))), 3),
                "horizontal_max_m": round(float(h.max()), 3),
                "vertical_rms_m": round(float(np.sqrt(np.mean(r[:, 2] ** 2))), 3),
            }
        )
        print("  ".join(f"{k}={v}" for k, v in rows[-1].items()))
    if args.out_dir:
        _write_csv(Path(args.out_dir) / "georef.csv", rows)


# ---------------------------------------------------------------------------
# lanelet2
# ---------------------------------------------------------------------------


def cmd_lanelet2(args: argparse.Namespace) -> None:
    import xml.etree.ElementTree as ET

    import lanelet2
    from lanelet2.io import Origin
    from lanelet2.projection import UtmProjector

    cfg = load_config(args.config)
    results = Path(cfg.get("output", {}).get("dir", "results"))
    rows = []
    for seq in args.sequences:
        osm = results / f"map_{seq}.osm"
        side = json.loads(osm.with_suffix(".georef.json").read_text())
        o = side["lanelet2_origin"]
        lmap, errors = lanelet2.io.loadRobust(str(osm), UtmProjector(Origin(o["lat"], o["lon"])))
        T = np.asarray(side["map_to_utm"])
        dev = []
        for p in lmap.pointLayer:
            x, y = float(p.attributes["local_x"]), float(p.attributes["local_y"])
            # Lanelet2 stores ``ele`` as z; recover the map z through the third row.
            z = (p.z - T[2, 3] - T[2, 0] * x - T[2, 1] * y) / T[2, 2]
            # Lanelet2 local coordinates are UTM minus the origin's UTM, and the
            # origin is the map origin, so they should equal the rotated map coordinates.
            e = T[:3, :3] @ np.array([x, y, z])
            dev.append(np.hypot(p.x - e[0], p.y - e[1]))
        rows.append(
            {
                "sequence": seq,
                "georeference": side["source"],
                "load_errors": len(errors),
                "ways_in_file": len(ET.parse(osm).getroot().findall("way")),
                "linestrings": len(lmap.lineStringLayer),
                "polygons": len(lmap.polygonLayer),
                "lanelets": len(lmap.laneletLayer),
                "max_position_deviation_m": float(f"{max(dev):.2e}"),
            }
        )
        print("  ".join(f"{k}={v}" for k, v in rows[-1].items()))
    if args.out_dir:
        _write_csv(Path(args.out_dir) / "lanelet2_load.csv", rows)


# ---------------------------------------------------------------------------
# lanesweep
# ---------------------------------------------------------------------------


def cmd_lanesweep(args: argparse.Namespace) -> None:
    """Lane-marking candidates, their DBSCAN neighbourhoods, and other min_points.

    Answers two questions: can a painted line pass the configured DBSCAN at
    all, and does a smaller ``min_points`` turn the candidates into lines?
    A line cluster here is one that passes ``_is_line``; its lateral offset
    is measured from the driven path.
    """
    cfg = load_config(args.config)
    eps = LANE_DBSCAN_EPS_M
    rows = []
    for seq in args.sequences:
        sq = Sequence(cfg, seq)
        P = sq.poses("optimized")
        Q = _lane_candidate_points(sq)
        nbrs = np.array([len(x) for x in cKDTree(Q).query_ball_point(Q, eps)])
        for mp in args.min_points:
            clusters = cluster_points(Q, eps=eps, min_points=mp)
            lines = [c for c in clusters if _is_line(c)]
            if lines:
                cent = np.array([c.mean(axis=0) for c in lines])
                _, lat = _road_profile(cent, P)
                off_road = float(np.mean(np.abs(lat) > args.off_road_m))
            else:
                off_road = float("nan")
            rows.append(
                {
                    "sequence": seq,
                    "candidates": len(Q),
                    "eps_m": eps,
                    "min_points": mp,
                    "share_with_min_points_neighbours": round(float(np.mean(nbrs >= mp)), 4),
                    "max_neighbours": int(nbrs.max()),
                    "clusters": len(clusters),
                    "line_clusters": len(lines),
                    f"line_clusters_beyond_{args.off_road_m:g}m": round(off_road, 2),
                }
            )
            print("  ".join(f"{k}={v}" for k, v in rows[-1].items()))
    if args.out_dir:
        _write_csv(Path(args.out_dir) / "lanesweep.csv", rows)


# ---------------------------------------------------------------------------
# worldband
# ---------------------------------------------------------------------------


def cmd_worldband(args: argparse.Namespace) -> None:
    """How often the road under the car falls inside the road band in world z.

    Before the band was applied in the nearest pose's frame, it was applied
    to world z, i.e. in the frame of the first scan. This counts the frames
    whose road surface (the point ``SENSOR_HEIGHT_M`` below the sensor) lies
    inside that band, for a trajectory.
    """
    cfg = load_config(args.config)
    lo, hi = ROAD_BAND_M
    rows = []
    for seq in args.sequences:
        sq = Sequence(cfg, seq)
        for source in ("optimized", "gt"):
            P = sq.poses(source)
            road_z = (P[:, :3, :3] @ np.array([0.0, 0.0, -SENSOR_HEIGHT_M]) + P[:, :3, 3])[:, 2]
            rows.append(
                {
                    "sequence": seq,
                    "source": source,
                    "road_z_min_m": round(float(road_z.min()), 2),
                    "road_z_max_m": round(float(road_z.max()), 2),
                    "frames_in_band": round(float(np.mean((road_z >= lo) & (road_z <= hi))), 3),
                }
            )
            print("  ".join(f"{k}={v}" for k, v in rows[-1].items()))
    if args.out_dir:
        _write_csv(Path(args.out_dir) / "worldband.csv", rows)


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------

# Reference data-viz palette: light surface, ink, sequential blue, categorical slot 2.
_SURFACE, _INK, _INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
_BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
_ORANGE = "#eb6834"


def _raster_min(xy: np.ndarray, v: np.ndarray, bounds, res: float) -> np.ndarray:
    """Per-pixel minimum of ``v``; NaN where empty. Rows run south to north."""
    x0, x1, y0, y1 = bounds
    nx, ny = int(np.ceil((x1 - x0) / res)), int(np.ceil((y1 - y0) / res))
    ix = ((xy[:, 0] - x0) / res).astype(int)
    iy = ((xy[:, 1] - y0) / res).astype(int)
    m = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    img = np.full(ny * nx, np.inf)
    np.minimum.at(img, iy[m] * nx + ix[m], v[m])
    img[~np.isfinite(img)] = np.nan
    return img.reshape(ny, nx)


def _raster_mean(xy: np.ndarray, v: np.ndarray, bounds, res: float) -> np.ndarray:
    x0, x1, y0, y1 = bounds
    nx, ny = int(np.ceil((x1 - x0) / res)), int(np.ceil((y1 - y0) / res))
    ix = ((xy[:, 0] - x0) / res).astype(int)
    iy = ((xy[:, 1] - y0) / res).astype(int)
    m = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    k = iy[m] * nx + ix[m]
    s = np.bincount(k, weights=v[m], minlength=nx * ny)
    c = np.bincount(k, minlength=nx * ny)
    with np.errstate(invalid="ignore", divide="ignore"):
        img = s / c
    img[c == 0] = np.nan
    return img.reshape(ny, nx)


def _osm_ways(path: Path, way_type: str) -> list[np.ndarray]:
    """Map-frame polylines of the ways with ``type=way_type`` in an exported map."""
    import xml.etree.ElementTree as ET

    root = ET.parse(path).getroot()
    nodes = {}
    for n in root.findall("node"):
        tags = {t.get("k"): t.get("v") for t in n.findall("tag")}
        nodes[n.get("id")] = (float(tags["local_x"]), float(tags["local_y"]))
    out = []
    for w in root.findall("way"):
        tags = {t.get("k"): t.get("v") for t in w.findall("tag")}
        if tags.get("type") == way_type:
            out.append(np.array([nodes[nd.get("ref")] for nd in w.findall("nd")]))
    return out


def _style(ax) -> None:
    ax.set_facecolor(_SURFACE)
    ax.tick_params(colors=_INK2, labelsize=8, length=2)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlabel("x [m]", color=_INK2, fontsize=8)
    ax.set_ylabel("y [m]", color=_INK2, fontsize=8)


def _figure_map(cfg: dict, seq: str, out: Path, zoom: tuple[float, float, float, float]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    sq = Sequence(cfg, seq)
    P = sq.poses("optimized")
    pts, _ = _working_map(sq)
    h = heights_in_nearest_pose_frame(pts, P) + SENSOR_HEIGHT_M
    osm = Path(cfg.get("output", {}).get("dir", "results")) / f"map_{seq}.osm"
    curbs = _osm_ways(osm, "curbstone")
    t = P[:, :3, 3]

    cmap = ListedColormap(_BLUE_RAMP)
    cmap.set_bad(_SURFACE)
    vmin, vmax = -0.25, 0.45
    pad = 5.0
    lo, hi = pts[:, :2].min(axis=0) - pad, pts[:, :2].max(axis=0) + pad
    full = (lo[0], hi[0], lo[1], hi[1])

    fig, axes = plt.subplots(
        1, 2, figsize=(13, 7.2), gridspec_kw={"width_ratios": [1.0, 0.62]}, facecolor=_SURFACE
    )
    for ax, bounds, res, lw in ((axes[0], full, 0.4, 0.8), (axes[1], zoom, 0.2, 1.6)):
        img = _raster_min(pts[:, :2], h, bounds, res)
        im = ax.imshow(
            img,
            origin="lower",
            extent=bounds,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.plot(t[:, 0], t[:, 1], color=_INK2, lw=0.6, ls=(0, (4, 3)))
        for c in curbs:
            ax.plot(c[:, 0], c[:, 1], color=_ORANGE, lw=lw, solid_capstyle="round")
        ax.set_xlim(bounds[0], bounds[1])
        ax.set_ylim(bounds[2], bounds[3])
        ax.set_aspect("equal")
        _style(ax)
    x0, x1, y0, y1 = zoom
    axes[0].add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=_INK, lw=0.8))
    axes[0].set_title(
        f"KITTI {seq}: map from the pose-graph trajectory", color=_INK, fontsize=10, loc="left"
    )
    axes[1].set_title("Detail (box on the left)", color=_INK, fontsize=10, loc="left")
    cb = fig.colorbar(im, ax=axes, shrink=0.55, pad=0.02, aspect=30)
    cb.set_label("height above the road under the car [m]", color=_INK2)
    cb.ax.tick_params(colors=_INK2, labelsize=8)
    cb.outline.set_visible(False)
    fig.legend(
        handles=[
            Line2D([], [], color=_ORANGE, lw=2, label="curb (exported)"),
            Line2D([], [], color=_INK2, lw=0.8, ls=(0, (4, 3)), label="driven path"),
        ],
        loc="lower left",
        bbox_to_anchor=(0.06, 0.03),
        ncol=2,
        frameon=False,
        fontsize=9,
        labelcolor=_INK,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=_SURFACE, bbox_inches="tight")
    print(f"wrote {out}")


def _working_map(sq: Sequence) -> tuple[np.ndarray, np.ndarray]:
    """Working-resolution map (points, reflectance) from the cache, for the current config."""
    cached = sq.cache.load_curbs(sq.cfg)
    if cached is None:
        raise SystemExit(f"{sq.seq}: no fresh map cache; run run_pipeline.py first")
    pcd = cached[0]
    return np.asarray(pcd.points), np.asarray(pcd.colors)[:, 0]


def _lane_candidates(cfg: dict, seq: str):
    """Working map, Stage 3 poses and the lane-marking candidate mask of a sequence."""
    sq = Sequence(cfg, seq)
    P = sq.poses("optimized")
    pts, refl = _working_map(sq)
    h = heights_in_nearest_pose_frame(pts, P)
    road = (h >= ROAD_BAND_M[0]) & (h <= ROAD_BAND_M[1])
    return pts, refl, P, road, road & (refl >= LANE_REFLECTANCE_MIN)


def _lane_candidate_points(sq: Sequence) -> np.ndarray:
    """Map points of the removed lane-marking channel: bright points in the road band."""
    pts, refl = _working_map(sq)
    h = heights_in_nearest_pose_frame(pts, sq.poses("optimized"))
    road = (h >= ROAD_BAND_M[0]) & (h <= ROAD_BAND_M[1])
    return pts[road & (refl >= LANE_REFLECTANCE_MIN)]


def _lane_clusters(sq: Sequence, min_points: int) -> list[np.ndarray]:
    return cluster_points(_lane_candidate_points(sq), eps=LANE_DBSCAN_EPS_M, min_points=min_points)


def _is_line(cluster: np.ndarray) -> bool:
    """The export's line test: at least 1 m long, linearity 0.75, at most 2 m thick."""
    if cluster.shape[0] < 3:
        return False
    st = _pca_stats(cluster[:, :2])
    return (
        not st.get("degenerate")
        and st["length"] >= 1.0
        and st["linearity"] >= 0.75
        and st["thickness"] <= 2.0
    )


def _reflectance_panel(ax, pts, refl, P, road, cand, frame: int, title: str) -> None:
    """Road reflectance around one scan, in its driving frame, with candidates."""
    import matplotlib.pyplot as plt

    d = pts - P[frame, :3, 3]
    u, v = d @ P[frame, :3, 0], d @ P[frame, :3, 1]  # forward, left
    box = (np.abs(u) < 60.0) & (np.abs(v) < 14.0)
    uv = np.column_stack([u[box & road], v[box & road]])
    img = _raster_mean(uv, refl[box & road], (-60, 60, -14, 14), 0.25)
    cmap = plt.get_cmap("gray").copy()
    cmap.set_bad(_SURFACE)
    ax.imshow(img, origin="lower", extent=(-60, 60, -14, 14), cmap=cmap, vmin=0.1, vmax=0.55)
    m = box & cand
    ax.scatter(u[m], v[m], s=3, color=_ORANGE, lw=0, label="lane-marking candidate")
    ax.set_aspect("equal")
    _style(ax)
    ax.set_xlabel("along the road [m]", color=_INK2, fontsize=8)
    ax.set_ylabel("across [m]", color=_INK2, fontsize=8)
    ax.set_title(title, color=_INK, fontsize=10, loc="left")


def _figure_lane_markings(cfg: dict, out: Path) -> None:
    """Paint on a highway (01) and on a residential street (00), and the DBSCAN gate."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    thr = LANE_REFLECTANCE_MIN
    eps, min_pts = LANE_DBSCAN_EPS_M, LANE_DBSCAN_MIN_POINTS
    fig, axes = plt.subplots(
        3, 1, figsize=(12, 9.6), gridspec_kw={"height_ratios": [1.2, 1.2, 1.0]}, facecolor=_SURFACE
    )
    pts, refl, P, road, cand = _lane_candidates(cfg, "01")
    _reflectance_panel(
        axes[0],
        pts,
        refl,
        P,
        road,
        cand,
        300,
        f"KITTI 01 (highway): road reflectance, and candidates with reflectance \u2265 {thr}",
    )
    axes[0].legend(loc="upper right", frameon=False, fontsize=8, labelcolor=_INK, markerscale=3)
    Q = pts[cand]
    nbrs = np.array([len(x) for x in cKDTree(Q).query_ball_point(Q, eps)])

    pts0, refl0, P0, road0, cand0 = _lane_candidates(cfg, "00")
    _reflectance_panel(
        axes[1],
        pts0,
        refl0,
        P0,
        road0,
        cand0,
        4090,
        "KITTI 00 (residential street), same scale and threshold",
    )
    del pts0, refl0, road0, cand0

    ax = axes[2]
    bins = np.arange(0, max(nbrs.max(), min_pts) + 3)
    ax.hist(nbrs, bins=bins, color=_BLUE_RAMP[3], edgecolor=_SURFACE, lw=0.5)
    ax.axvline(min_pts, color=_INK, lw=1.0)
    ax.text(
        min_pts,
        ax.get_ylim()[1] * 0.9,
        f"  DBSCAN min_points = {min_pts}",
        color=_INK,
        fontsize=8,
        va="top",
    )
    _style(ax)
    ax.set_xlabel(f"candidates within eps = {eps} m of a candidate", color=_INK2, fontsize=8)
    ax.set_ylabel("candidates", color=_INK2, fontsize=8)
    share = float(np.mean(nbrs >= min_pts))
    ax.set_title(
        f"KITTI 01, all {len(nbrs):,} candidates: {100 * share:.2f} % have at least "
        f"{min_pts} candidates within {eps} m",
        color=_INK,
        fontsize=10,
        loc="left",
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=_SURFACE, bbox_inches="tight")
    print(f"wrote {out}")


def cmd_figure(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    out = Path(args.out_dir)
    if "map" in args.which:
        _figure_map(cfg, "00", out / "map_00.png", zoom=(200.0, 280.0, -100.0, 40.0))
    if "lane-markings" in args.which:
        _figure_lane_markings(cfg, out / "lane_markings.png")


# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--config", default="configs/default.yaml")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("crispness", help="road thickness of local maps per pose source")
    p.add_argument("--sequences", nargs="+", default=["00"])
    p.add_argument("--sources", nargs="+", default=list(POSE_SOURCES), choices=POSE_SOURCES)
    p.add_argument("--window", type=int, default=200)
    p.add_argument("--n-windows", type=int, default=5)
    p.add_argument("--lane-halfwidth", type=float, default=2.0)
    p.add_argument("--out-dir", default=None)
    p.set_defaults(func=cmd_crispness)

    p = sub.add_parser("features", help="lateral position of the extracted clusters")
    p.add_argument("--sequences", nargs="+", default=["00"])
    p.add_argument("--out-dir", default=None)
    p.set_defaults(func=cmd_features)

    p = sub.add_parser("revisit", help="offset between two passes over the same street")
    p.add_argument("--sequences", nargs="+", default=list(SEQUENCES_WITH_LOOPS))
    p.add_argument(
        "--sources", nargs="+", default=["odometry", "optimized", "gt"], choices=POSE_SOURCES
    )
    p.add_argument("--min-gap", type=int, default=300, help="frames between the two passes")
    p.add_argument("--radius", type=float, default=3.0, help="m, horizontal, on the ground truth")
    p.add_argument("--stride", type=int, default=10)
    p.add_argument("--half-window", type=int, default=5, help="scans on each side of an epoch")
    p.add_argument("--voxel", type=float, default=0.2)
    p.add_argument("--min-fitness", type=float, default=0.5)
    p.add_argument("--out-dir", default=None)
    p.set_defaults(func=cmd_revisit)

    p = sub.add_parser("georef", help="rigid fit of trajectories to the GNSS positions")
    p.add_argument("--sequences", nargs="+", default=["00", "05"])
    p.add_argument(
        "--sources", nargs="+", default=["odometry", "optimized", "gt"], choices=POSE_SOURCES
    )
    p.add_argument("--out-dir", default=None)
    p.set_defaults(func=cmd_georef)

    p = sub.add_parser("lanelet2", help="load the exported maps with Lanelet2")
    p.add_argument("--sequences", nargs="+", default=["00"])
    p.add_argument("--out-dir", default=None)
    p.set_defaults(func=cmd_lanelet2)

    p = sub.add_parser("lanesweep", help="lane-marking candidates under several min_points")
    p.add_argument("--sequences", nargs="+", default=["00", "01"])
    p.add_argument("--min-points", nargs="+", type=int, default=[40, 10, 5])
    p.add_argument("--off-road-m", type=float, default=6.0)
    p.add_argument("--out-dir", default=None)
    p.set_defaults(func=cmd_lanesweep)

    p = sub.add_parser("worldband", help="frames whose road lies in the band in world z")
    p.add_argument("--sequences", nargs="+", default=["00"])
    p.add_argument("--out-dir", default=None)
    p.set_defaults(func=cmd_worldband)

    p = sub.add_parser("figure", help="figures for docs/map-quality.md")
    p.add_argument("--which", nargs="+", default=["map", "lane-markings"])
    p.add_argument("--out-dir", default="docs/figures")
    p.set_defaults(func=cmd_figure)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
