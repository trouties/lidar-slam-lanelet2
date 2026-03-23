"""Pose file evaluation utilities.

Wraps the evo toolkit to evaluate KITTI-format pose files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.odometry.kiss_icp_wrapper import evaluate_odometry


def load_poses_kitti_format(path: str | Path) -> list[np.ndarray]:
    """Load poses from a KITTI-format file (12 floats per line, 3x4 row-major).

    Returns:
        List of 4x4 SE(3) matrices.
    """
    path = Path(path)
    poses = []
    with path.open() as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            if len(vals) != 12:
                continue
            T = np.eye(4)
            T[:3, :] = np.array(vals).reshape(3, 4)
            poses.append(T)
    return poses


def evaluate_pose_file(
    est_path: str | Path,
    gt_path: str | Path,
) -> dict[str, dict]:
    """Evaluate estimated poses against ground truth.

    Both files must be in KITTI pose format (12 floats per line).

    Returns:
        Dictionary with ``'ape'`` and ``'rpe'`` sub-dicts, each containing
        ``rmse``, ``mean``, ``median``, ``std``, ``min``, ``max``.
    """
    est = load_poses_kitti_format(est_path)
    gt = load_poses_kitti_format(gt_path)
    n = min(len(est), len(gt))
    return evaluate_odometry(est[:n], gt[:n])
