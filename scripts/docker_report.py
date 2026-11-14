"""APE of the Docker run (results/ape.txt) and a bird's-eye plot (results/trajectory.png).

Runs after ``scripts/run_pipeline.py`` in ``scripts/docker_entrypoint.sh``.
Reads ``poses_<SEQ>.txt`` (Stage 2) and ``poses_optimized_<SEQ>.txt``
(Stage 3) from ``--results-dir`` and the ground truth from
``<dataset-root>/poses/<SEQ>.txt``; only its first rows, as many as there are
estimated poses, are used. APE is the translation RMSE in the KITTI camera
frame, with the first-frame and the SE(3) Umeyama datum (docs/evaluation.md).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.kitti_loader import load_poses  # noqa: E402
from src.odometry.kiss_icp_wrapper import evaluate_odometry  # noqa: E402

STAGES = (("stage2", "poses"), ("stage3", "poses_optimized"))


def _ape(est: np.ndarray, gt: np.ndarray, align: str) -> float:
    n = min(len(est), len(gt))
    return float(evaluate_odometry(list(est[:n]), list(gt[:n]), align=align)["ape"]["rmse"])


def _plot(est: np.ndarray, gt: np.ndarray, out: Path, title: str) -> None:
    """Bird's-eye view in the KITTI camera frame (x right, z forward)."""
    n = min(len(est), len(gt))
    fig, ax = plt.subplots(figsize=(7, 7), dpi=120)
    ax.plot(gt[:n, 0, 3], gt[:n, 2, 3], "k-", lw=2.0, label="KITTI ground truth")
    ax.plot(est[:n, 0, 3], est[:n, 2, 3], "r--", lw=1.5, label="Stage 3")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="best")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--sequence", default="00")
    ap.add_argument("--dataset-root", type=Path, required=True)
    args = ap.parse_args(argv)

    gt = load_poses(args.dataset_root / "poses" / f"{args.sequence}.txt")
    lines = []
    est = None
    for stage, stem in STAGES:
        est = load_poses(args.results_dir / f"{stem}_{args.sequence}.txt")
        lines.append(f"{stage}_frames={len(est)}")
        lines.append(f"{stage}_ape_first_m={_ape(est, gt, 'first'):.4f}")
        lines.append(f"{stage}_ape_se3_m={_ape(est, gt, 'se3'):.4f}")
    (args.results_dir / "ape.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    _plot(est, gt, args.results_dir / "trajectory.png", f"KITTI {args.sequence}, {len(est)} frames")
    return 0


if __name__ == "__main__":
    sys.exit(main())
