"""Stage 2 (KISS-ICP odometry) on the ten nuScenes mini scenes.

Each scene is a single 20 s pass without a revisit, so Stage 3 has no loop
closure to add and its optimum is the odometry itself; only Stage 2 is run.
Writes one row per scene: path length and translation APE RMSE with the
first-frame and the SE(3) Umeyama datum (docs/evaluation.md).

Usage:
    python scripts/eval_nuscenes.py --dataroot ~/data/nuscenes
    python scripts/eval_nuscenes.py --dataroot ~/data/nuscenes --max-frames 20
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import yaml

# Ensure project root is on path when run as a script.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.nuscenes_loader import NuScenesDataset
from src.odometry.kiss_icp_wrapper import KissICPOdometry, evaluate_odometry

# nuScenes' 32-beam Velodyne HDL-32E returns far fewer points per sweep than
# KITTI's 64-beam HDL-64E; a finer voxel keeps enough of them for the
# registration, and the closer minimum range keeps the near ground.
NUSCENES_VOXEL_SIZE_M = 0.5
NUSCENES_MIN_RANGE_M = 3.0


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_scene(nusc, scene: dict, kiss_cfg: dict, max_frames: int | None = None) -> dict:
    """KISS-ICP on one scene's lidar sweeps, evaluated against the nuScenes ego poses."""
    # Sweeps (about 20 Hz), not keyframes: keyframes are at 2 Hz, about 4 m
    # apart at city speed, too far for scan-to-map registration.
    dataset = NuScenesDataset(nusc, scene["token"])
    n_frames = len(dataset)
    if max_frames is not None:
        n_frames = min(n_frames, max_frames)
        dataset._filepaths = dataset._filepaths[:n_frames]
        dataset.poses = dataset.poses[:n_frames]
        dataset.timestamps = dataset.timestamps[:n_frames]
    print(f"  Scene {scene['name']}: {n_frames} sweeps")

    odom = KissICPOdometry(
        max_range=kiss_cfg.get("max_range", 100.0),
        min_range=NUSCENES_MIN_RANGE_M,
        voxel_size=NUSCENES_VOXEL_SIZE_M,
    )
    poses = odom.run(dataset)
    gt = [dataset.poses[i] for i in range(n_frames)]
    n = min(len(poses), len(gt))
    t = np.array([g[:3, 3] for g in gt[:n]])
    return {
        "scene_token": scene["token"],
        "scene_name": scene["name"],
        "n_frames": n,
        "path_length_m": round(float(np.sum(np.linalg.norm(np.diff(t, axis=0), axis=1))), 1),
        "ape_first_m": round(float(evaluate_odometry(poses[:n], gt[:n])["ape"]["rmse"]), 4),
        "ape_se3_m": round(
            float(evaluate_odometry(poses[:n], gt[:n], align="se3")["ape"]["rmse"]), 4
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="KISS-ICP on the nuScenes mini scenes")
    parser.add_argument(
        "--dataroot",
        default="~/data/nuscenes",
        help="Path to nuScenes mini dataset root",
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Pipeline config YAML",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Cap frames per scene (for quick smoke tests)",
    )
    parser.add_argument(
        "--output",
        default="benchmarks/nuscenes_ape.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    try:
        from nuscenes.nuscenes import NuScenes
    except ImportError:
        print("ERROR: nuscenes-devkit not installed. Run: pip install nuscenes-devkit")
        sys.exit(1)

    cfg = load_config(args.config)
    kiss_cfg = cfg.get("kiss_icp", {})
    dataroot = str(Path(args.dataroot).expanduser())
    print(f"Loading nuScenes mini from {dataroot} ...")
    nusc = NuScenes(version="v1.0-mini", dataroot=dataroot, verbose=False)
    print(f"Found {len(nusc.scene)} scenes\n")

    rows: list[dict] = []
    for i, scene in enumerate(nusc.scene):
        print(f"[{i + 1}/{len(nusc.scene)}] {scene['name']}")
        row = run_scene(nusc, scene, kiss_cfg, args.max_frames)
        rows.append(row)
        print(
            f"  {row['path_length_m']} m, APE first-frame {row['ape_first_m']:.3f} m, "
            f"SE(3) {row['ape_se3_m']:.3f} m"
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
