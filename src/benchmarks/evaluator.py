"""Pose file loading and the rotation-about-start APE datum."""

from __future__ import annotations

from pathlib import Path

import numpy as np


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


def ape_rotation_about_start(est_xyz: np.ndarray, gt_xyz: np.ndarray) -> float:
    """Translation APE RMSE with only the start position held fixed.

    The estimate is rotated about its first position by the rotation that best
    fits it to the reference, then compared point by point. This datum has
    three free parameters and sits between first-frame comparison (none free)
    and SE(3) alignment (six free). The drop from first-frame APE to this value
    is the part of the error that a single orientation offset explains; the
    further drop to SE(3) APE is what a free origin absorbs on top.

    Args:
        est_xyz: ``(N, 3)`` estimated positions.
        gt_xyz: ``(N, 3)`` reference positions in the same frame.

    Returns:
        RMSE of the position error in metres.
    """
    a = np.asarray(est_xyz, dtype=float) - est_xyz[0]
    b = np.asarray(gt_xyz, dtype=float) - gt_xyz[0]
    # Kabsch without centring: R maximises trace(R^T b^T a).
    U, _, Vt = np.linalg.svd(b.T @ a)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    err = a @ R.T - b
    return float(np.sqrt(np.mean(np.sum(err**2, axis=1))))
