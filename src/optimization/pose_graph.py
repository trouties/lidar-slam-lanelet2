"""Pose graph construction and optimization using GTSAM.

Builds a factor graph from odometry and loop closure constraints,
then optimizes with Levenberg-Marquardt.
"""

from __future__ import annotations

import gtsam
import numpy as np


def _noise_from_override(
    override: list[float] | np.ndarray,
) -> gtsam.noiseModel.Base:
    """Build a GTSAM noise model from a per-edge override.

    Accepts either a length-6 list in config order ``[tx, ty, tz, rx, ry, rz]``
    (diagonal) or a ``(6, 6)`` ndarray in GTSAM tangent order
    ``[rx, ry, rz, tx, ty, tz]`` (full Gaussian covariance). The degeneracy
    downgrade uses both: uniform tx/ty/tz inflation and inflation along the
    least-observed direction only.
    """
    if isinstance(override, np.ndarray):
        cov = np.asarray(override, dtype=np.float64)
        if cov.shape != (6, 6):
            raise ValueError(f"ndarray edge sigma must be (6,6), got {cov.shape}")
        return gtsam.noiseModel.Gaussian.Covariance(cov)
    s = override
    return gtsam.noiseModel.Diagonal.Sigmas(np.array([*s[3:], *s[:3]]))


def _make_robust(
    kernel: str | None,
    scale: float,
    base: gtsam.noiseModel.Base,
) -> gtsam.noiseModel.Base:
    """Wrap a loop-closure noise model in a Cauchy M-estimator.

    Args:
        kernel: ``None``, ``"none"`` or ``"cauchy"``. Case-insensitive.
        scale: Cauchy ``k`` in whitened units (sigmas). Ignored when
            ``kernel`` is off.
        base: Gaussian noise model of the factor.

    Returns:
        ``base`` wrapped in :class:`noiseModel.Robust`, or ``base`` unchanged
        when ``kernel`` is off.
    """
    if kernel is None or kernel.lower() in ("", "none"):
        return base
    if kernel.lower() != "cauchy":
        raise ValueError(f"unknown robust kernel: {kernel!r} (expected none|cauchy)")
    estimator = gtsam.noiseModel.mEstimator.Cauchy.Create(float(scale))
    return gtsam.noiseModel.Robust.Create(estimator, base)


class PoseGraphOptimizer:
    """GTSAM-based pose graph optimizer.

    Builds a factor graph from sequential odometry poses and optional
    loop closure constraints, then runs Levenberg-Marquardt optimization.
    """

    def __init__(
        self,
        odom_sigmas: list[float] | None = None,
        prior_sigmas: list[float] | None = None,
        robust_kernel: str | None = None,
        robust_scale: float = 1.0,
        loop_sigmas: list[float] | None = None,
    ) -> None:
        """Initialize optimizer.

        Args:
            odom_sigmas: 6-element [tx,ty,tz,rx,ry,rz] noise sigmas for
                odometry factors. Reordered internally to GTSAM convention
                [rx,ry,rz,tx,ty,tz]. Defaults to [0.1,0.1,0.1,0.01,0.01,0.01].
            prior_sigmas: 6-element [tx,ty,tz,rx,ry,rz] noise sigmas for
                the prior factor on pose 0. Defaults to tight values.
            robust_kernel: Optional M-estimator robust kernel wrapping
                loop closure noise only (odometry + prior stay Gaussian).
                ``None`` / ``"none"`` keeps every factor Gaussian. See
                :func:`_make_robust` for supported names.
            robust_scale: Kernel scale parameter passed through to
                :func:`_make_robust`. Only read when ``robust_kernel`` is
                an active kernel name.
            loop_sigmas: 6-element [tx,ty,tz,rx,ry,rz] noise sigmas of the
                loop-closure factors. Defaults to the odometry sigmas.
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
        self.loop_noise = (
            self.odom_noise
            if loop_sigmas is None
            else gtsam.noiseModel.Diagonal.Sigmas(np.array([*loop_sigmas[3:], *loop_sigmas[:3]]))
        )

        self.robust_kernel = robust_kernel
        self.robust_scale = float(robust_scale)

        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_values = gtsam.Values()
        self.n_poses = 0
        self.result_values: gtsam.Values | None = None
        self._marginals: gtsam.Marginals | None = None

    @classmethod
    def from_config(cls, gtsam_cfg: dict, lc_cfg: dict | None = None) -> PoseGraphOptimizer:
        """The Stage 3 stochastic model from the ``gtsam`` and ``loop_closure`` sections.

        Every script that builds a Stage 3 graph goes through here, so that a
        sigma in the config means the same thing everywhere.
        """
        lc_cfg = lc_cfg or {}
        kernel = lc_cfg.get("robust_kernel")
        if isinstance(kernel, str) and kernel.lower() in ("", "none"):
            kernel = None
        return cls(
            odom_sigmas=gtsam_cfg.get("odom_sigmas"),
            prior_sigmas=gtsam_cfg.get("prior_sigmas"),
            robust_kernel=kernel,
            robust_scale=float(lc_cfg.get("robust_scale", 1.0)),
            loop_sigmas=gtsam_cfg.get("loop_closure_sigmas"),
        )

    def build_graph(
        self,
        poses: list[np.ndarray],
        prior_indices: list[int] | None = None,
        gt_poses: list[np.ndarray] | None = None,
        edge_sigmas: list[list[float] | np.ndarray | None] | None = None,
    ) -> None:
        """Build pose graph from sequential odometry poses.

        Adds priors and BetweenFactorPose3 for each consecutive pair.

        Args:
            poses: List of 4x4 SE(3) pose matrices from odometry.
            prior_indices: Frame indices that receive an absolute prior
                on the full pose. If ``None``, only frame 0 gets one, which
                fixes the datum (the graph's gauge freedom) without
                constraining its shape.
            gt_poses: Ground-truth poses used as prior values when
                *prior_indices* is given.  Falls back to estimated *poses*
                if ``None``.
            edge_sigmas: Optional per-edge noise override. Length must be
                ``len(poses)``; entry ``i`` applies to the edge
                ``(i-1, i)`` (entry ``0`` is unused). ``None`` entries
                fall back to the global :attr:`odom_noise`. Two forms are
                accepted per entry:
                  * ``list[float]`` of length 6 in config order
                    ``[tx, ty, tz, rx, ry, rz]`` — builds a diagonal noise
                    model (degeneracy ``uniform`` mode).
                  * ``np.ndarray`` of shape ``(6, 6)`` in GTSAM tangent
                    order ``[rx, ry, rz, tx, ty, tz]`` — builds a full
                    Gaussian covariance noise model (degeneracy
                    ``directional`` mode, where only the degenerate
                    translation axis is inflated via a rank-1 update).
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

                edge_noise = self.odom_noise
                if edge_sigmas is not None and i < len(edge_sigmas):
                    s = edge_sigmas[i]
                    if s is not None:
                        edge_noise = _noise_from_override(s)

                self.graph.add(gtsam.BetweenFactorPose3(i - 1, i, gtsam.Pose3(delta), edge_noise))

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
                Defaults to the optimizer's loop-closure sigmas.
        """
        if sigmas is not None:
            noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([*sigmas[3:], *sigmas[:3]]))
        else:
            noise = self.loop_noise

        # Robust M-estimator wrapping is loop-closure-only; odometry stays
        # strict Gaussian so normal between-factor geometry is preserved.
        noise = _make_robust(self.robust_kernel, self.robust_scale, noise)

        self.graph.add(gtsam.BetweenFactorPose3(i, j, gtsam.Pose3(relative_pose), noise))

    def add_position_prior(self, i: int, position: np.ndarray, sigmas: list[float]) -> None:
        """Constrain the position of pose ``i`` alone, as a GNSS fix would.

        Args:
            i: Pose index.
            position: ``(3,)`` position in the map frame [m].
            sigmas: ``[sx, sy, sz]`` in map axes [m].
        """
        noise = gtsam.noiseModel.Diagonal.Sigmas(np.asarray(sigmas, dtype=np.float64))
        self.graph.add(gtsam.GPSFactor(i, gtsam.Point3(*np.asarray(position, dtype=float)), noise))

    def optimize(self) -> list[np.ndarray]:
        """Run Levenberg-Marquardt optimization.

        Returns:
            List of optimized 4x4 SE(3) pose matrices.
        """
        params = gtsam.LevenbergMarquardtParams()
        optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.initial_values, params)
        result = optimizer.optimize()
        self.result_values = result
        self._marginals = None

        optimized_poses = []
        for i in range(self.n_poses):
            optimized_poses.append(result.atPose3(i).matrix())
        return optimized_poses

    def marginals(self) -> gtsam.Marginals:
        """GTSAM marginals at the optimum, computed once per :meth:`optimize`."""
        if self.result_values is None:
            raise RuntimeError("optimize() must be called before querying covariances")
        if self._marginals is None:
            self._marginals = gtsam.Marginals(self.graph, self.result_values)
        return self._marginals

    def get_position_marginals(
        self,
        keys: list[int] | None = None,
    ) -> dict[int, np.ndarray]:
        """3x3 position covariance of selected poses, in map axes.

        GTSAM perturbs a pose on the right, ``T * Exp(xi)`` with ``xi`` in
        tangent order ``[rx, ry, rz, tx, ty, tz]``, so the translation block
        ``cov[3:6, 3:6]`` of its marginal is expressed in the pose's own body
        axes. It is rotated into map axes here, ``R cov R^T``; drawn as is in
        the map, an ellipse would turn with the vehicle heading.

        Args:
            keys: Pose indices to query. ``None`` => all ``n_poses`` keys.

        Returns:
            Dict mapping pose index to ``(3, 3)`` covariance in m^2.

        Raises:
            RuntimeError: If :meth:`optimize` has not been called.
        """
        marginals = self.marginals()
        if keys is None:
            keys = list(range(self.n_poses))
        out: dict[int, np.ndarray] = {}
        for k in keys:
            R = self.result_values.atPose3(k).rotation().matrix()
            cov = np.asarray(marginals.marginalCovariance(k))[3:6, 3:6]
            out[k] = np.ascontiguousarray(R @ cov @ R.T)
        return out

    def relative_pose_covariances(self, first: int, others: list[int]) -> dict[int, np.ndarray]:
        """Covariance of the relative pose ``T_first^-1 T_j`` for each ``j``.

        In GTSAM tangent order ``[rx, ry, rz, tx, ty, tz]``, body axes of
        pose ``j``. With right perturbations ``T_i Exp(d_i)``, the relative
        pose is perturbed by ``d_j - Ad(T_ij^-1) d_i``, so its covariance
        follows from the joint marginal of the two poses. Unlike an absolute
        pose covariance, it does not depend on how the datum is fixed (the
        prior on frame 0): a relative pose is an estimable quantity.
        """
        marginals = self.marginals()
        others = [int(j) for j in others]
        keys = list(dict.fromkeys([int(first), *others]))
        joint = marginals.jointMarginalCovariance(gtsam.KeyVector(keys))
        Ti = self.result_values.atPose3(first)
        Sii = np.asarray(joint.at(first, first))
        out: dict[int, np.ndarray] = {}
        for j in others:
            if j == first:
                out[j] = np.zeros((6, 6))
                continue
            A = Ti.between(self.result_values.atPose3(j)).inverse().AdjointMap()
            Sjj = np.asarray(joint.at(j, j))
            Sji = np.asarray(joint.at(j, first))
            S = Sjj + A @ Sii @ A.T - Sji @ A.T - A @ Sji.T
            out[j] = 0.5 * (S + S.T)
        return out

    @property
    def graph_size(self) -> int:
        """Number of factors in the graph."""
        return self.graph.size()
