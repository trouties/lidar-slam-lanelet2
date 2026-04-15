"""Pose graph construction and optimization using GTSAM.

Builds a factor graph from odometry and loop closure constraints,
then optimizes with Levenberg-Marquardt.
"""

from __future__ import annotations

import gtsam
import numpy as np


class PoseGraphOptimizer:
    """GTSAM-based pose graph optimizer.

    Builds a factor graph from sequential odometry poses and optional
    loop closure constraints, then runs Levenberg-Marquardt optimization.
    """

    def __init__(
        self,
        odom_sigmas: list[float] | None = None,
        prior_sigmas: list[float] | None = None,
    ) -> None:
        """Initialize optimizer.

        Args:
            odom_sigmas: 6-element [tx,ty,tz,rx,ry,rz] noise sigmas for
                odometry factors. Reordered internally to GTSAM convention
                [rx,ry,rz,tx,ty,tz]. Defaults to [0.1,0.1,0.1,0.01,0.01,0.01].
            prior_sigmas: 6-element [tx,ty,tz,rx,ry,rz] noise sigmas for
                the prior factor on pose 0. Defaults to tight values.
        """
        if odom_sigmas is None:
            odom_sigmas = [0.1, 0.1, 0.1, 0.01, 0.01, 0.01]
        if prior_sigmas is None:
            prior_sigmas = [0.01, 0.01, 0.01, 0.001, 0.001, 0.001]

        # Reorder config [tx,ty,tz,rx,ry,rz] → GTSAM [rx,ry,rz,tx,ty,tz]
        self.odom_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([*odom_sigmas[3:], *odom_sigmas[:3]])
        )
        self.prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([*prior_sigmas[3:], *prior_sigmas[:3]])
        )

        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_values = gtsam.Values()
        self.n_poses = 0
        self.result_values: gtsam.Values | None = None

    def build_graph(
        self,
        poses: list[np.ndarray],
        prior_indices: list[int] | None = None,
        gt_poses: list[np.ndarray] | None = None,
    ) -> None:
        """Build pose graph from sequential odometry poses.

        Adds priors and BetweenFactorPose3 for each consecutive pair.

        Args:
            poses: List of 4x4 SE(3) pose matrices from odometry.
            prior_indices: Frame indices that receive an absolute prior.
                If ``None``, only frame 0 gets a prior (v3 default).
                Use :func:`src.benchmarks.gnss_denial.make_prior_indices`
                to generate indices for GNSS-denial experiments.
            gt_poses: Ground-truth poses used as prior values when
                *prior_indices* is given.  Falls back to estimated *poses*
                if ``None``.
        """
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_values = gtsam.Values()
        self.n_poses = len(poses)

        # Determine which frames get absolute priors
        if prior_indices is None:
            prior_set = {0}
        else:
            prior_set = set(prior_indices)
            prior_set.add(0)  # frame 0 always anchored

        prior_source = gt_poses if gt_poses is not None else poses

        for idx in sorted(prior_set):
            if 0 <= idx < len(poses):
                self.graph.add(
                    gtsam.PriorFactorPose3(idx, gtsam.Pose3(prior_source[idx]), self.prior_noise)
                )

        # Insert initial values and between factors
        for i, pose in enumerate(poses):
            self.initial_values.insert(i, gtsam.Pose3(pose))

            if i > 0:
                # Relative transform: delta = T_{i-1}^{-1} @ T_i
                T_prev_inv = np.linalg.inv(poses[i - 1])
                delta = T_prev_inv @ pose
                self.graph.add(
                    gtsam.BetweenFactorPose3(i - 1, i, gtsam.Pose3(delta), self.odom_noise)
                )

    def add_loop_closure(
        self,
        i: int,
        j: int,
        relative_pose: np.ndarray,
        sigmas: list[float] | None = None,
    ) -> None:
        """Add a loop closure constraint between poses i and j.

        Args:
            i: Source pose index.
            j: Target pose index.
            relative_pose: 4x4 relative SE(3) transform from i to j.
            sigmas: Optional 6-element [tx,ty,tz,rx,ry,rz] noise sigmas.
                Uses odometry noise if not specified.
        """
        if sigmas is not None:
            noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([*sigmas[3:], *sigmas[:3]]))
        else:
            noise = self.odom_noise

        self.graph.add(gtsam.BetweenFactorPose3(i, j, gtsam.Pose3(relative_pose), noise))

    def optimize(self) -> list[np.ndarray]:
        """Run Levenberg-Marquardt optimization.

        Returns:
            List of optimized 4x4 SE(3) pose matrices.
        """
        params = gtsam.LevenbergMarquardtParams()
        optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.initial_values, params)
        result = optimizer.optimize()
        self.result_values = result

        optimized_poses = []
        for i in range(self.n_poses):
            optimized_poses.append(result.atPose3(i).matrix())
        return optimized_poses

    def get_position_marginals(
        self,
        keys: list[int] | None = None,
    ) -> dict[int, np.ndarray]:
        """Extract the 3x3 position marginal covariance for selected keys.

        GTSAM Pose3 uses tangent order [rx,ry,rz,tx,ty,tz], so the
        translation block is ``cov[3:6, 3:6]``.

        Args:
            keys: Pose indices to query. ``None`` => all ``n_poses`` keys.

        Returns:
            Dict mapping pose index to ``(3, 3)`` covariance in m^2.

        Raises:
            RuntimeError: If :meth:`optimize` has not been called.
        """
        if self.result_values is None:
            raise RuntimeError("optimize() must be called before get_position_marginals()")
        if keys is None:
            keys = list(range(self.n_poses))

        marginals = gtsam.Marginals(self.graph, self.result_values)

        out: dict[int, np.ndarray] = {}
        try:
            key_vec = gtsam.KeyVector(keys)
            joint = marginals.jointMarginalCovariance(key_vec)
            for k in keys:
                cov6 = np.asarray(joint.at(k, k))
                out[k] = np.ascontiguousarray(cov6[3:6, 3:6])
        except Exception:
            for k in keys:
                cov6 = np.asarray(marginals.marginalCovariance(k))
                out[k] = np.ascontiguousarray(cov6[3:6, 3:6])
        return out

    @property
    def graph_size(self) -> int:
        """Number of factors in the graph."""
        return self.graph.size()
