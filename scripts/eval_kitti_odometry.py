#!/usr/bin/env python3
"""KITTI 00-10 trajectory evaluation with the devkit relative error and APE.

Evaluates the poses that ``run_pipeline.py`` writes (Stage 2 odometry and
Stage 3 optimized), and can run two odometry-only comparison variants so that
the Stage 2 numbers can be checked against published KISS-ICP results:

  uncorrected         this repository's loader and KISS-ICP wrapper with the
                      KITTI scan correction switched off (config ``kiss_icp``
                      otherwise unchanged).
  kiss_icp_reference  KISS-ICP's own KITTI dataloader with its default
                      configuration; equivalent to
                      ``kiss_icp_pipeline --dataloader kitti --sequence XX``.

``run`` stores Velodyne-frame poses of a comparison variant as float64
``.npy``. ``evaluate`` writes one CSV row per (source, sequence) plus a
pooled 00-10 row per source when all eleven sequences are present.
``references`` evaluates the same trajectories against the KITTI ground
truth and against its metric version (``src/benchmarks/metric_reference.py``),
on the sequences whose KITTI raw OxTS log is available.

Metrics:
  t_rel, r_rel   KITTI devkit relative error, camera frame
                 (see ``src/benchmarks/kitti_metrics.py``), with the
                 translation error split into lateral / vertical / forward.
  ape_first      APE RMSE, translation, Velodyne frame, no alignment
                 (frames share the origin) -- as in ``run_pipeline.py``.
  ape_first_vertical  vertical (Velodyne z) component of ape_first.
  ape_rot_start  APE RMSE after the best rotation about the start position
                 (start position fixed, orientation free).
  ape_se3        APE RMSE after Umeyama SE(3) alignment, Velodyne frame.

Usage:
  python scripts/run_pipeline.py --sequence 04 --until optimized
  python -m scripts.eval_kitti_odometry run --variant kiss_icp_reference
  python -m scripts.eval_kitti_odometry evaluate --csv benchmarks/kitti_odometry.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import yaml

from src.benchmarks.evaluator import ape_rotation_about_start
from src.benchmarks.kitti_metrics import KittiSegmentErrors, kitti_segment_errors
from src.data.kitti_loader import KITTIDataset, load_poses
from src.odometry.kiss_icp_wrapper import KissICPOdometry, evaluate_odometry

VARIANTS = ("uncorrected", "kiss_icp_reference")
# run_pipeline.py output name -> source label in the CSV
PIPELINE_OUTPUTS = {"poses_{seq}.txt": "stage2", "poses_optimized_{seq}.txt": "stage3"}
SEQUENCES_WITH_GT = tuple(f"{i:02d}" for i in range(11))


def _run_reference(root: Path, sequence: str) -> list[np.ndarray]:
    """KISS-ICP's own KITTI loop (mirrors ``OdometryPipeline._run_pipeline``)."""
    from kiss_icp.config.parser import load_config
    from kiss_icp.datasets.kitti import KITTIOdometryDataset
    from kiss_icp.kiss_icp import KissICP

    dataset = KITTIOdometryDataset(data_dir=str(root), sequence=sequence)
    icp = KissICP(load_config(None, max_range=None))
    poses = []
    for idx in range(len(dataset)):
        frame, timestamps = dataset[idx]
        icp.register_frame(frame, timestamps)
        poses.append(icp.last_pose.copy())
        if (idx + 1) % 500 == 0:
            print(f"  Frame {idx + 1}/{len(dataset)}", flush=True)
    return poses


def _run_one(variant: str, root: Path, sequence: str, kiss_cfg: dict) -> list[np.ndarray]:
    if variant == "kiss_icp_reference":
        return _run_reference(root, sequence)
    dataset = KITTIDataset(root_path=root, sequence=sequence, correct_scan=False)
    odom = KissICPOdometry(
        max_range=kiss_cfg.get("max_range", 100.0),
        min_range=kiss_cfg.get("min_range", 5.0),
        voxel_size=kiss_cfg.get("voxel_size", 1.0),
    )
    return odom.run(dataset)


def cmd_run(args: argparse.Namespace) -> None:
    cfg = yaml.safe_load(Path(args.config).read_text())
    root = Path(args.kitti_root or cfg["data"]["kitti_root"]).expanduser()
    out = Path(args.out_dir) / args.variant
    out.mkdir(parents=True, exist_ok=True)
    for seq in args.sequences:
        pose_file = out / f"{seq}.npy"
        if pose_file.exists() and not args.force:
            print(f"[{args.variant}/{seq}] exists, skipping")
            continue
        print(f"[{args.variant}/{seq}] running", flush=True)
        t0 = time.perf_counter()
        poses = _run_one(args.variant, root, seq, cfg.get("kiss_icp", {}))
        wall = time.perf_counter() - t0
        np.save(pose_file, np.asarray(poses, dtype=np.float64))
        uses_repo_cfg = args.variant != "kiss_icp_reference"
        meta = {
            "variant": args.variant,
            "sequence": seq,
            "frames": len(poses),
            "wall_s": round(wall, 1),
            "kiss_icp_cfg": cfg.get("kiss_icp", {}) if uses_repo_cfg else None,
        }
        (out / f"{seq}.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(f"[{args.variant}/{seq}] done: {len(poses)} poses, {wall:.0f} s", flush=True)


def _metric_columns(seg: KittiSegmentErrors) -> dict:
    axes = seg.axis_breakdown_percent()
    return {
        "n_segments": seg.count,
        "t_rel_pct": round(seg.t_rel_percent(), 4),
        "r_rel_deg_per_100m": round(seg.r_rel_deg_per_100m(), 4),
        "t_lateral_pct": round(axes["lateral"], 4),
        "t_vertical_pct": round(axes["vertical"], 4),
        "t_forward_pct": round(axes["forward"], 4),
    }


def _evaluate_sequence(
    root: Path, sequence: str, est_velo: np.ndarray, gt_velo: np.ndarray | None = None
) -> tuple[dict, KittiSegmentErrors]:
    """Metrics of one trajectory; ``gt_velo`` replaces the KITTI ground truth if given."""
    ds = KITTIDataset(root_path=root, sequence=sequence)
    Tr = ds.calibration["Tr"]
    Tr_inv = np.linalg.inv(Tr)
    if gt_velo is None:
        gt_cam = ds.poses
        gt_velo = Tr_inv @ gt_cam @ Tr
    else:
        gt_cam = Tr @ gt_velo @ Tr_inv
    if gt_cam is None or len(gt_cam) != len(est_velo):
        raise ValueError(f"{sequence}: GT/estimate length mismatch")
    est_cam = Tr @ est_velo @ Tr_inv

    seg = kitti_segment_errors(gt_cam, est_cam)
    ape_first = evaluate_odometry(list(est_velo), list(gt_velo), align="first")["ape"]["rmse"]
    ape_se3 = evaluate_odometry(list(est_velo), list(gt_velo), align="se3")["ape"]["rmse"]
    dz = est_velo[:, 2, 3] - gt_velo[:, 2, 3]
    ape_rot_start = ape_rotation_about_start(est_velo[:, :3, 3], gt_velo[:, :3, 3])
    path_len = float(np.sum(np.linalg.norm(np.diff(gt_cam[:, :3, 3], axis=0), axis=1)))
    row = {
        "sequence": sequence,
        "frames": len(est_velo),
        "path_length_m": round(path_len, 1),
        **_metric_columns(seg),
        "ape_first_m": round(float(ape_first), 4),
        "ape_first_vertical_m": round(float(np.sqrt(np.mean(dz**2))), 4),
        "ape_rot_start_m": round(ape_rot_start, 4),
        "ape_se3_m": round(float(ape_se3), 4),
    }
    return row, seg


def _sources(
    args: argparse.Namespace, root: Path, pipeline_dir: Path
) -> dict[str, dict[str, np.ndarray]]:
    """``{source label: {sequence: Velodyne-frame poses}}`` for what exists on disk."""
    sources: dict[str, dict[str, np.ndarray]] = {}
    if pipeline_dir:
        for pattern, label in PIPELINE_OUTPUTS.items():
            found = {}
            for seq in args.sequences:
                path = pipeline_dir / pattern.format(seq=seq)
                if path.exists():
                    Tr = KITTIDataset(root_path=root, sequence=seq).calibration["Tr"]
                    found[seq] = np.linalg.inv(Tr) @ load_poses(path) @ Tr
            if found:
                sources[label] = found
    for variant in args.variants:
        found = {}
        for seq in args.sequences:
            path = Path(args.out_dir) / variant / f"{seq}.npy"
            if path.exists():
                found[seq] = np.load(path)
        if found:
            sources[variant] = found
    return sources


def cmd_evaluate(args: argparse.Namespace) -> None:
    cfg = yaml.safe_load(Path(args.config).read_text())
    root = Path(args.kitti_root or cfg["data"]["kitti_root"]).expanduser()
    pipeline_dir = Path(args.pipeline_dir or cfg.get("output", {}).get("dir", "results"))
    rows = []
    for label, by_seq in _sources(args, root, pipeline_dir).items():
        parts: list[KittiSegmentErrors] = []
        for seq, est_velo in by_seq.items():
            row, seg = _evaluate_sequence(root, seq, est_velo)
            rows.append({"source": label, **row})
            parts.append(seg)
            print(
                f"{label:>20s} {seq}  t_rel={row['t_rel_pct']:.3f}%  "
                f"r_rel={row['r_rel_deg_per_100m']:.3f}deg/100m  "
                f"(lat {row['t_lateral_pct']:.2f} / vert {row['t_vertical_pct']:.2f} / "
                f"fwd {row['t_forward_pct']:.2f} %)  "
                f"APE first={row['ape_first_m']:.2f}m (vert {row['ape_first_vertical_m']:.2f}) "
                f"rot_start={row['ape_rot_start_m']:.2f}m se3={row['ape_se3_m']:.2f}m"
            )
        if tuple(sorted(by_seq)) == SEQUENCES_WITH_GT:
            pooled = KittiSegmentErrors.pool(parts)
            rows.append(
                {
                    "source": label,
                    "sequence": "00-10",
                    "frames": sum(len(p) for p in by_seq.values()),
                    "path_length_m": "",
                    **_metric_columns(pooled),
                    "ape_first_m": "",
                    "ape_first_vertical_m": "",
                    "ape_rot_start_m": "",
                    "ape_se3_m": "",
                }
            )
            print(
                f"{label:>20s} 00-10 pooled  t_rel={pooled.t_rel_percent():.3f}%  "
                f"r_rel={pooled.r_rel_deg_per_100m():.3f}deg/100m"
            )
    if args.csv and rows:
        path = Path(args.csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {path}")


def cmd_references(args: argparse.Namespace) -> None:
    """Every trajectory against the KITTI ground truth and against its metric version.

    Only sequences whose KITTI raw OxTS log is available (see
    ``scripts/download_kitti_oxts.py``) are evaluated, and both references
    are pooled over the same sequences. The row with source ``kitti_gt`` is
    the KITTI ground truth evaluated against its metric version: the error
    the devkit projection alone contributes.
    """
    from src.benchmarks.metric_reference import metric_ground_truth
    from src.data.kitti_raw import ODOM_TO_RAW
    from src.export.georef import load_imu_lever_arm, oxts_dir_for_sequence

    cfg = yaml.safe_load(Path(args.config).read_text())
    root = Path(args.kitti_root or cfg["data"]["kitti_root"]).expanduser()
    raw_root = Path(args.kitti_raw).expanduser()
    pipeline_dir = Path(args.pipeline_dir or cfg.get("output", {}).get("dir", "results"))

    metric, corr_rows = {}, []
    for seq in args.sequences:
        oxts = oxts_dir_for_sequence(raw_root, seq)
        if oxts is None:
            print(f"{seq}: no OxTS log under {raw_root}, skipped")
            continue
        ds = KITTIDataset(root_path=root, sequence=seq)
        Tr = ds.calibration["Tr"]
        gt_velo = np.linalg.inv(Tr) @ ds.poses @ Tr
        lever = load_imu_lever_arm(raw_root, ODOM_TO_RAW[seq][0])
        m = metric_ground_truth(gt_velo, np.asarray(ds.timestamps), oxts, lever)
        metric[seq] = (gt_velo, m.poses)
        corr_rows.append(
            {
                "sequence": seq,
                "time_offset_s": round(m.time_offset_s, 3),
                "devkit_fit_rms_m": round(m.devkit_fit_rms_m, 4),
                "correction_rms_m": round(m.correction_rms_m, 4),
                "correction_max_m": round(m.correction_max_m, 4),
            }
        )
        print("  ".join(f"{k}={v}" for k, v in corr_rows[-1].items()))

    args.sequences = sorted(metric)
    sources = _sources(args, root, pipeline_dir)
    sources["kitti_gt"] = {seq: g for seq, (g, _) in metric.items()}
    pooled_label = ",".join(sorted(metric))
    rows = []
    for label, by_seq in sources.items():
        if sorted(by_seq) != sorted(metric):
            print(f"{label}: missing sequences, skipped")
            continue
        for ref in ("kitti", "metric"):
            if label == "kitti_gt" and ref == "kitti":
                continue
            parts = []
            for seq in sorted(by_seq):
                gt = None if ref == "kitti" else metric[seq][1]
                row, seg = _evaluate_sequence(root, seq, by_seq[seq], gt_velo=gt)
                rows.append({"source": label, "reference": ref, **row})
                parts.append(seg)
            pooled = KittiSegmentErrors.pool(parts)
            rows.append(
                {
                    "source": label,
                    "reference": ref,
                    "sequence": pooled_label,
                    **_metric_columns(pooled),
                }
            )
            print(
                f"{label:>20s} vs {ref:6s} pooled  t_rel={pooled.t_rel_percent():.4f}%  "
                f"r_rel={pooled.r_rel_deg_per_100m():.4f}deg/100m"
            )
    out = Path(args.out_csv_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, data in (("kitti_metric_reference.csv", corr_rows), ("kitti_references.csv", rows)):
        keys = list(dict.fromkeys(k for r in data for k in r))
        with (out / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(data)
        print(f"wrote {out / name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--kitti-root", default=None, help="overrides data.kitti_root")
    parser.add_argument("--out-dir", default="results/kitti_odometry")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--sequences", nargs="+", default=list(SEQUENCES_WITH_GT))
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", parents=[common], help="run a comparison variant")
    p_run.add_argument("--variant", choices=VARIANTS, required=True)
    p_run.add_argument("--force", action="store_true", help="recompute existing sequences")
    p_run.set_defaults(func=cmd_run)

    p_eval = sub.add_parser("evaluate", parents=[common], help="compute metrics")
    p_eval.add_argument(
        "--pipeline-dir",
        default=None,
        help="run_pipeline.py output dir to evaluate (default: output.dir of --config)",
    )
    p_eval.add_argument("--variants", nargs="*", choices=VARIANTS, default=list(VARIANTS))
    p_eval.add_argument("--csv", default=None, help="write results to this CSV")
    p_eval.set_defaults(func=cmd_evaluate)

    p_ref = sub.add_parser(
        "references", parents=[common], help="KITTI ground truth vs. its metric version"
    )
    p_ref.add_argument("--pipeline-dir", default=None)
    p_ref.add_argument("--variants", nargs="*", choices=VARIANTS, default=list(VARIANTS))
    p_ref.add_argument("--kitti-raw", default="~/data/kitti_raw", help="KITTI raw root (OxTS)")
    p_ref.add_argument("--out-csv-dir", default="benchmarks")
    p_ref.set_defaults(func=cmd_references)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
