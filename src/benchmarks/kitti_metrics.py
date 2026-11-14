"""KITTI odometry relative error (the metric of the official KITTI devkit).

APE answers "how far is the trajectory from the reference after one global
alignment", so a single heading error early in a sequence dominates it and
the number depends on the alignment choice. The KITTI metric instead compares
relative motion over sub-trajectories of fixed path length (100 m ... 800 m),
starting every 10th frame, and normalizes the error by that length. It is
alignment-free and is the number published odometry papers report, which is
why it is the reference point for checking the odometry stage against
literature.

This is a direct port of ``evaluate_odometry.cpp`` from the KITTI odometry
devkit. Poses must be in the left camera frame (the frame of the KITTI
``poses/XX.txt`` files) because the devkit is defined there.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SEGMENT_LENGTHS_M = (100, 200, 300, 400, 500, 600, 700, 800)
STEP_SIZE_FRAMES = 10


@dataclass
class KittiSegmentErrors:
    """Per-segment errors, kept so several sequences can be pooled."""

    t_err_per_m: np.ndarray  # translation error / segment length [m/m]
    r_err_per_m: np.ndarray  # rotation error / segment length [rad/m]
    length_m: np.ndarray  # segment length the error belongs to [m]
    first_frame: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    # Translation error vector / segment length, camera axes (x right,
    # y down, z forward) of the segment end. Not part of the devkit; used to
    # tell vertical drift from along-track (scale) or lateral drift.
    t_err_xyz_per_m: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))

    @property
    def count(self) -> int:
        return int(self.t_err_per_m.size)

    def t_rel_percent(self) -> float:
        """Mean translational error in % (the headline KITTI number)."""
        return float(np.mean(self.t_err_per_m) * 100.0) if self.count else float("nan")

    def r_rel_deg_per_100m(self) -> float:
        """Mean rotational error in deg/100 m."""
        if not self.count:
            return float("nan")
        return float(np.degrees(np.mean(self.r_err_per_m)) * 100.0)

    def axis_breakdown_percent(self) -> dict[str, float]:
        """Mean absolute error per camera axis, in % of segment length."""
        if not self.count:
            return {"lateral": float("nan"), "vertical": float("nan"), "forward": float("nan")}
        m = np.mean(np.abs(self.t_err_xyz_per_m), axis=0) * 100.0
        return {"lateral": float(m[0]), "vertical": float(m[1]), "forward": float(m[2])}

    def by_length(self) -> dict[int, tuple[float, float, int]]:
        """``{length: (t_rel %, r_rel deg/100m, n_segments)}``."""
        out: dict[int, tuple[float, float, int]] = {}
        for length in SEGMENT_LENGTHS_M:
            m = self.length_m == length
            if not m.any():
                continue
            out[length] = (
                float(np.mean(self.t_err_per_m[m]) * 100.0),
                float(np.degrees(np.mean(self.r_err_per_m[m])) * 100.0),
                int(m.sum()),
            )
        return out

    @staticmethod
    def pool(parts: list[KittiSegmentErrors]) -> KittiSegmentErrors:
        """Concatenate segments of several sequences.

        The devkit averages over all segments of all sequences, so long
        sequences weigh more than short ones. That pooled value is what
        papers report as "average over 00-10".
        """
        return KittiSegmentErrors(
            t_err_per_m=np.concatenate([p.t_err_per_m for p in parts]),
            r_err_per_m=np.concatenate([p.r_err_per_m for p in parts]),
            length_m=np.concatenate([p.length_m for p in parts]),
            first_frame=np.concatenate([p.first_frame for p in parts]),
            t_err_xyz_per_m=np.concatenate([p.t_err_xyz_per_m for p in parts]),
        )


def _trajectory_distances(poses: np.ndarray) -> np.ndarray:
    steps = np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(steps)])


def _rotation_angle(R: np.ndarray) -> float:
    # Same clamp as the devkit: acos is undefined outside [-1, 1] and the
    # trace of a numerically perturbed rotation can leave that interval.
    d = 0.5 * (np.trace(R) - 1.0)
    return float(np.arccos(max(min(d, 1.0), -1.0)))


def kitti_segment_errors(gt_poses: np.ndarray, est_poses: np.ndarray) -> KittiSegmentErrors:
    """Compute devkit segment errors for one sequence.

    Args:
        gt_poses: ``(N, 4, 4)`` ground-truth poses, camera frame.
        est_poses: ``(N, 4, 4)`` estimated poses, camera frame, same indexing.

    Segment lengths are measured along the ground-truth path, as in the
    devkit, so the set of evaluated segments does not depend on the estimate.
    """
    gt = np.asarray(gt_poses, dtype=np.float64)
    est = np.asarray(est_poses, dtype=np.float64)
    if gt.shape != est.shape or gt.ndim != 3 or gt.shape[1:] != (4, 4):
        raise ValueError(f"pose arrays must both be (N, 4, 4); got {gt.shape} and {est.shape}")

    dist = _trajectory_distances(gt)
    n = gt.shape[0]
    t_err, r_err, lengths, firsts, t_xyz = [], [], [], [], []
    for first in range(0, n, STEP_SIZE_FRAMES):
        for length in SEGMENT_LENGTHS_M:
            # First frame whose path distance exceeds the segment length.
            candidates = np.nonzero(dist[first:] > dist[first] + length)[0]
            if candidates.size == 0:
                continue
            last = first + int(candidates[0])
            delta_gt = np.linalg.inv(gt[first]) @ gt[last]
            delta_est = np.linalg.inv(est[first]) @ est[last]
            err = np.linalg.inv(delta_est) @ delta_gt
            t_err.append(np.linalg.norm(err[:3, 3]) / length)
            r_err.append(_rotation_angle(err[:3, :3]) / length)
            t_xyz.append(err[:3, 3] / length)
            lengths.append(length)
            firsts.append(first)
    return KittiSegmentErrors(
        t_err_per_m=np.asarray(t_err, dtype=np.float64),
        r_err_per_m=np.asarray(r_err, dtype=np.float64),
        length_m=np.asarray(lengths, dtype=int),
        first_frame=np.asarray(firsts, dtype=int),
        t_err_xyz_per_m=np.asarray(t_xyz, dtype=np.float64).reshape(-1, 3),
    )
