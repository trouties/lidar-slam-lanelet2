"""GNSS denial window utilities.

KITTI Odometry has no GNSS, so we simulate "GNSS denial" by choosing a
contiguous frame window where no absolute pose priors are added to the
pose graph.  Outside that window, priors are added every ``prior_stride``
frames to emulate normal GNSS reception.
"""

from __future__ import annotations

import numpy as np

from src.odometry.kiss_icp_wrapper import evaluate_odometry


def make_denial_window(
    poses: list[np.ndarray],
    target_distance: float = 150.0,
) -> tuple[int, int]:
    """Find a contiguous frame window whose arc-length ≥ *target_distance*.

    Starts from the trajectory midpoint and extends forward.

    Args:
        poses: List of 4x4 SE(3) pose matrices.
        target_distance: Minimum arc-length in metres.

    Returns:
        ``(start_frame, end_frame)`` inclusive indices.

    Raises:
        ValueError: If the trajectory is too short.
    """
    translations = np.array([p[:3, 3] for p in poses])
    midpoint = len(translations) // 2

    cumulative = 0.0
    for end in range(midpoint, len(translations) - 1):
        cumulative += float(np.linalg.norm(translations[end + 1] - translations[end]))
        if cumulative >= target_distance:
            return midpoint, end + 1

    raise ValueError(
        f"Trajectory arc-length from midpoint ({cumulative:.1f} m) "
        f"is shorter than target {target_distance} m"
    )


def make_prior_indices(
    n_frames: int,
    denial_start: int,
    denial_end: int,
    prior_stride: int = 50,
) -> list[int]:
    """Build frame indices where absolute priors should be added.

    Frame 0 always gets a prior.  Outside the denial window, priors are
    added every *prior_stride* frames.  Inside the window, no priors.

    Args:
        n_frames: Total number of frames.
        denial_start: First frame of denial window (inclusive).
        denial_end: Last frame of denial window (inclusive).
        prior_stride: Add a prior every N frames outside the window.

    Returns:
        Sorted list of frame indices.
    """
    indices = {0}
    for i in range(0, n_frames, prior_stride):
        if i < denial_start or i > denial_end:
            indices.add(i)
    return sorted(indices)


def score_denial_drift(
    est_poses: list[np.ndarray],
    gt_poses: list[np.ndarray],
    start: int,
    end: int,
) -> dict[str, float]:
    """Compute drift metrics within a denial window.

    Returns:
        Dict with ``ape_rmse``, ``ape_mean``, ``window_length_m``,
        and ``drift_per_meter``.
    """
    est_window = est_poses[start : end + 1]
    gt_window = gt_poses[start : end + 1]
    n = min(len(est_window), len(gt_window))
    if n < 2:
        return {
            "ape_rmse": float("nan"),
            "ape_mean": float("nan"),
            "window_length_m": 0.0,
            "drift_per_meter": float("nan"),
        }

    metrics = evaluate_odometry(est_window[:n], gt_window[:n])
    ape_mean = float(metrics["ape"]["mean"])

    gt_trans = np.array([p[:3, 3] for p in gt_window[:n]])
    diffs = np.diff(gt_trans, axis=0)
    window_length = float(np.sum(np.linalg.norm(diffs, axis=1)))

    return {
        "ape_rmse": float(metrics["ape"]["rmse"]),
        "ape_mean": ape_mean,
        "window_length_m": window_length,
        "drift_per_meter": ape_mean / window_length if window_length > 0 else float("nan"),
    }
