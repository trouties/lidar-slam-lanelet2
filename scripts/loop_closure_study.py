#!/usr/bin/env python3
"""Why did the pose graph make the KITTI trajectories worse, and what fixes it?

The expensive part (Scan Context and ICP on every candidate) runs once and is
stored, so that every graph experiment afterwards takes seconds:

  detect   The Stage 3 front end on the cached Stage 2 poses, querying every
           frame (stride 1), keeping every candidate with its verification
           ICP result instead of only the accepted ones.
  refine   The Stage 3 refinement (``loop_closure.refine``) on the stored
           candidates.
  retry    The Stage 3 retry of rejected candidates from the Scan Context
           alignment (``loop_closure.retry_from_scan_context``).
  analyze  True error of every loop constraint against ground truth, and the
           pose graph re-solved under different constraint sources and
           stochastic models. The query stride of the current config is
           emulated on the stored candidates (``j % stride == 0``), so the
           ``after`` variant reproduces what ``run_pipeline.py`` computes.
  gps      Ground-truth quality at the revisits from the GPS fix status in the
           KITTI raw OxTS data, and the calibration of the loop-closure sigma.

Ground truth is used to diagnose (true errors, upper bounds). It enters the
fix in one place: ``gtsam.loop_closure_sigmas`` is the RMS error of the
refined loop closures on the sequence-05 loops whose two ends both have an
RTK-integer GPS fix (``gps`` subcommand).

Usage:
  python scripts/run_pipeline.py --sequence 00 --until odometry
  python -m scripts.loop_closure_study detect --sequences 00
  python -m scripts.loop_closure_study refine --sequences 00 --accepted-only
  python -m scripts.loop_closure_study analyze --sequences 00
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass, replace
from pathlib import Path

import gtsam
import numpy as np
import open3d as o3d
from evo.core.trajectory import PosePath3D

from scripts.eval_kitti_odometry import _evaluate_sequence
from scripts.run_pipeline import load_config
from src.cache import LayeredCache
from src.cache.layered_cache import compute_hash
from src.data.kitti_loader import KITTIDataset
from src.data.kitti_raw import ODOM_TO_RAW
from src.export.georef import (
    georeference_from_gnss,
    load_imu_lever_arm,
    load_oxts_monotonic,
    oxts_dir_for_sequence,
)
from src.odometry.degeneracy import (
    directional_covariance,
    downgraded_edge_covariances,
    sustained_mask,
)
from src.odometry.kiss_icp_wrapper import evaluate_odometry
from src.optimization.loop_closure import LoopClosureDetector
from src.optimization.pose_graph import PoseGraphOptimizer
from src.optimization.scan_context import make_scan_context

DEFAULT_OUT = Path("results/loop_closure")
SEQUENCES_WITH_LOOPS = ("00", "02", "05", "06", "07", "08", "09")


def _sequence_config(cfg: dict, sequence: str) -> dict:
    return {**cfg, "data": {**cfg.get("data", {}), "sequence": sequence}}


def _load_stage2(cfg: dict, sequence: str) -> np.ndarray:
    cache_root = cfg.get("cache", {}).get("root", "cache/kitti")
    cached = LayeredCache(cache_root, sequence).load_odometry(cfg)
    if cached is None:
        raise SystemExit(
            f"{sequence}: no fresh Stage 2 cache for this config; run "
            f"`python scripts/run_pipeline.py --sequence {sequence} --until odometry` first"
        )
    return cached[0]


def _deduplicate(cands: list[tuple[int, int, float]]) -> list[tuple[int, int, float]]:
    """Same order-preserving de-duplication as ``LoopClosureDetector.detect``."""
    seen: set[tuple[int, int]] = set()
    out = []
    for i, j, d in cands:
        key = (min(i, j), max(i, j))
        if key not in seen:
            seen.add(key)
            out.append((i, j, d))
    return out


def cmd_detect(args: argparse.Namespace) -> None:
    base_cfg = load_config(args.config)
    lc_cfg = base_cfg.get("loop_closure", {})
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for seq in args.sequences:
        cfg = _sequence_config(base_cfg, seq)
        poses = _load_stage2(cfg, seq)
        dataset = KITTIDataset.from_config(cfg["data"], seq)
        sc_all = {**lc_cfg.get("scan_context", {}), "query_stride": 1}
        detector = LoopClosureDetector.from_config({**lc_cfg, "scan_context": sc_all})
        detector._downsample_cache = {}

        t0 = time.perf_counter()
        cands = _deduplicate(detector.detect_candidates_sc(dataset, len(poses)))
        t_sc = time.perf_counter() - t0

        n = len(cands)
        T_init = np.zeros((n, 4, 4))
        T_icp = np.zeros((n, 4, 4))
        fitness = np.zeros(n)
        inlier_rmse = np.zeros(n)
        t0 = time.perf_counter()
        for k, (i, j, _) in enumerate(cands):
            T_init[k] = np.linalg.inv(poses[i]) @ poses[j]
            res = detector.register(j, i, dataset, T_init[k])
            T_icp[k] = np.asarray(res.transformation)
            fitness[k] = res.fitness
            inlier_rmse[k] = res.inlier_rmse
            if (k + 1) % 1000 == 0:
                print(f"  [{seq}] ICP {k + 1}/{n}", flush=True)
        t_icp = time.perf_counter() - t0

        accepted = fitness >= detector.icp_fitness_threshold
        path = out_dir / f"{seq}.npz"
        np.savez_compressed(
            path,
            i=np.array([c[0] for c in cands], dtype=np.int64),
            j=np.array([c[1] for c in cands], dtype=np.int64),
            sc_dist=np.array([c[2] for c in cands]),
            T_init=T_init,
            T_icp=T_icp,
            fitness=fitness,
            inlier_rmse=inlier_rmse,
            accepted=accepted,
            fitness_threshold=detector.icp_fitness_threshold,
            odometry_hash=LayeredCache.hash_for("odometry", cfg),
            loop_closure_hash=compute_hash(lc_cfg),
        )
        print(
            f"[{seq}] candidates={n} accepted={int(accepted.sum())} "
            f"SC {t_sc:.0f} s, ICP {t_icp:.0f} s -> {path}",
            flush=True,
        )


# ---------------------------------------------------------------------------
# refine: re-register stored candidates with the Stage 3 refinement
# ---------------------------------------------------------------------------

# Fit of every refined candidate is also measured at a tight correspondence
# distance on the finest refinement voxels.
TIGHT_CORR_M = 0.3


def cmd_refine(args: argparse.Namespace) -> None:
    base_cfg = load_config(args.config)
    lc_cfg = base_cfg.get("loop_closure", {})
    if not (lc_cfg.get("refine") or {}).get("stages"):
        raise SystemExit("loop_closure.refine.stages is not set in the config")
    reg = o3d.pipelines.registration
    for seq in args.sequences:
        cfg = _sequence_config(base_cfg, seq)
        c = np.load(Path(args.in_dir) / f"{seq}.npz")
        sel = np.flatnonzero(c["accepted"]) if args.accepted_only else np.arange(len(c["i"]))
        if args.limit:
            sel = sel[: args.limit]
        dataset = KITTIDataset.from_config(cfg["data"], seq)
        det = LoopClosureDetector.from_config(lc_cfg)
        finest = min(v for v, _ in det.refine_stages)
        n = len(sel)
        T = np.zeros((n, 4, 4))
        cols = np.zeros((n, 4))
        t0 = time.perf_counter()
        for k, idx in enumerate(sel):
            i, j = int(c["i"][idx]), int(c["j"][idx])
            res = det.refine(j, i, dataset, c["T_init"][idx])
            T[k] = np.asarray(res.transformation)
            ev = reg.evaluate_registration(
                det._refine_cloud(j, dataset, finest),
                det._refine_cloud(i, dataset, finest),
                TIGHT_CORR_M,
                T[k],
            )
            cols[k] = (res.fitness, res.inlier_rmse, ev.fitness, ev.inlier_rmse)
            if (k + 1) % 500 == 0:
                print(f"  [{seq}] refine {k + 1}/{n}", flush=True)
        suffix = "_accepted" if args.accepted_only else ""
        path = Path(args.out_dir or args.in_dir) / f"{seq}_refined{suffix}.npz"
        np.savez_compressed(
            path,
            index=sel,
            T=T,
            fitness=cols[:, 0],
            inlier_rmse=cols[:, 1],
            tight_fitness=cols[:, 2],
            tight_inlier_rmse=cols[:, 3],
        )
        print(f"[{seq}] refined {n} pairs in {time.perf_counter() - t0:.0f} s -> {path}")


# ---------------------------------------------------------------------------
# retry: candidates the prior-started verification rejected, from Scan Context
# ---------------------------------------------------------------------------


def cmd_retry(args: argparse.Namespace) -> None:
    """Retry rejected candidates from the Scan Context alignment, as Stage 3 does.

    Uses the detector's own ``scan_context_start``, ``refine`` and
    verification, so the stored result is what ``run_pipeline.py`` computes
    for the same candidate. Every rejected candidate is retried (query
    stride 1); the stride and the drift check are applied in ``analyze``.
    """
    base_cfg = load_config(args.config)
    lc_cfg = base_cfg.get("loop_closure", {})
    if not (lc_cfg.get("refine") or {}).get("stages"):
        raise SystemExit("loop_closure.refine.stages is not set in the config")
    for seq in args.sequences:
        cfg = _sequence_config(base_cfg, seq)
        c = np.load(Path(args.in_dir) / f"{seq}.npz")
        on_stride = c["j"] % args.stride == 0
        sel = np.flatnonzero(~c["accepted"] & (~on_stride if args.rest else on_stride))
        dataset = KITTIDataset.from_config(cfg["data"], seq)
        det = LoopClosureDetector.from_config({**lc_cfg, "retry_from_scan_context": True})
        det._downsample_cache = {}
        sc_args = (det.sc_num_rings, det.sc_num_sectors, det.sc_max_range)
        n = len(sel)
        T = np.zeros((n, 4, 4))
        fitness = np.zeros(n)
        t0 = time.perf_counter()
        for k, idx in enumerate(sel):
            i, j = int(c["i"][idx]), int(c["j"][idx])
            for f in (i, j):
                if f not in det._sc_by_frame:
                    det._sc_by_frame[f] = make_scan_context(dataset[f][0][:, :3], *sc_args)
            start = det.scan_context_start(i, j)
            T[k] = np.asarray(det.refine(j, i, dataset, start).transformation)
            fitness[k] = det.register(j, i, dataset, T[k]).fitness
            if (k + 1) % 500 == 0:
                print(f"  [{seq}] retry {k + 1}/{n}", flush=True)
        suffix = f"_retry_off_stride{args.stride}" if args.rest else "_retry"
        path = Path(args.out_dir or args.in_dir) / f"{seq}{suffix}.npz"
        np.savez_compressed(path, index=sel, T=T, fitness=fitness)
        ok = int((fitness >= det.icp_fitness_threshold).sum())
        dt = time.perf_counter() - t0
        print(f"[{seq}] retried {n}, {ok} pass verification, {dt:.0f} s -> {path}")


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------


def _relative_error(T_meas: np.ndarray, T_true: np.ndarray) -> np.ndarray:
    """``Logmap(inv(T_true) @ T_meas)`` per pose pair, GTSAM order [rx ry rz tx ty tz]."""
    return np.array(
        [gtsam.Pose3.Logmap(gtsam.Pose3(np.linalg.inv(t) @ m)) for m, t in zip(T_meas, T_true)]
    )


def _relative(P: np.ndarray, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    return np.linalg.inv(P[i]) @ P[j]


@dataclass(frozen=True)
class Model:
    """Stage 3 settings that the experiments vary.

    ``loop_sigmas=None`` gives loop closures the odometry noise, as
    ``PoseGraphOptimizer`` does without ``loop_closure_sigmas``.
    ``degeneracy`` relaxes the odometry edges of degenerate frames along the
    least-observed direction: ``"body"`` rotates that direction into the
    frame the edge noise is expressed in, ``"map"`` uses it unrotated, as the
    pipeline did before this study. ``None``: no relaxation (Stage 3).
    """

    odom_sigmas: tuple[float, ...]
    loop_sigmas: tuple[float, ...] | None
    query_stride: int
    refined: bool
    robust_kernel: str | None
    robust_scale: float = 1.0
    retry: bool = False
    max_drift_ratio: float | None = None
    degeneracy: str | None = None


# Stage 3 as it was configured before this study: the configuration that the
# stored candidates were detected with.
BEFORE = Model(
    odom_sigmas=(0.1, 0.1, 0.1, 0.01, 0.01, 0.01),
    loop_sigmas=None,
    query_stride=1,
    refined=False,
    robust_kernel=None,
    degeneracy="map",
)


def model_from_config(cfg: dict) -> Model:
    g = cfg.get("gtsam", {})
    lc = cfg.get("loop_closure", {})
    rk = lc.get("robust_kernel")
    if isinstance(rk, str) and rk.lower() in ("", "none"):
        rk = None
    loop = g.get("loop_closure_sigmas")
    return Model(
        odom_sigmas=tuple(g.get("odom_sigmas") or BEFORE.odom_sigmas),
        loop_sigmas=tuple(loop) if loop else None,
        query_stride=int(lc.get("scan_context", {}).get("query_stride", 1)),
        refined=bool((lc.get("refine") or {}).get("stages")),
        robust_kernel=rk,
        robust_scale=float(lc.get("robust_scale", 1.0)),
        retry=bool(lc.get("retry_from_scan_context", False)),
        max_drift_ratio=lc.get("max_drift_ratio"),
    )


def degeneracy_edge_covariances(
    cfg: dict, sequence: str, odom: np.ndarray, base_sigmas, mode: str | None
) -> list | None:
    """Per-edge odometry covariances of the degeneracy relaxation (see :class:`Model`)."""
    if mode is None:
        return None
    d = cfg.get("degeneracy", {}) or {}
    cache_root = cfg.get("cache", {}).get("root", "cache/kitti")
    scores = LayeredCache(cache_root, sequence).load_degeneracy(_sequence_config(cfg, sequence))
    if scores is None:
        raise SystemExit(f"{sequence}: degeneracy scores missing; run run_pipeline first")
    if mode == "body":
        return downgraded_edge_covariances(scores, odom, list(base_sigmas), d)
    if mode != "map":
        raise ValueError(f"unknown degeneracy mode {mode!r}")
    factor = float(d.get("sigma_inflation_factor", 1.0))
    sustained = sustained_mask(
        scores[:, 0],
        threshold=float(d.get("cond_threshold", 0.0)),
        ema_alpha=float(d.get("ema_alpha", 0.3)),
        min_consecutive=int(d.get("min_consecutive", 5)),
    )
    return [
        directional_covariance(list(base_sigmas), scores[k, 4:7], factor)
        if k > 0 and sustained[k]
        else None
        for k in range(len(scores))
    ]


class Study:
    """Everything one sequence's experiments need, loaded once."""

    def __init__(self, cfg: dict, sequence: str, in_dir: Path) -> None:
        self.cfg = _sequence_config(cfg, sequence)
        self.seq = sequence
        self.odom = _load_stage2(self.cfg, sequence)
        ds = KITTIDataset.from_config(self.cfg["data"], sequence)
        Tr = ds.calibration["Tr"]
        Tr_inv = np.linalg.inv(Tr)
        self.gt = np.array([Tr_inv @ g @ Tr for g in ds.poses])
        self.timestamps = np.asarray(ds.timestamps)

        c = np.load(in_dir / f"{sequence}.npz")
        if str(c["odometry_hash"]) != LayeredCache.hash_for("odometry", self.cfg):
            raise SystemExit(f"{sequence}: candidates come from other Stage 2 poses; rerun detect")
        self.cand = {k: c[k] for k in c.files}
        acc = self.cand["accepted"]
        self.i = self.cand["i"][acc]
        self.j = self.cand["j"][acc]
        self.T_init = self.cand["T_init"][acc]
        self.T_icp = self.cand["T_icp"][acc]
        self.T_gt = _relative(self.gt, self.i, self.j)
        self.T_refined = None
        refined_path = in_dir / f"{sequence}_refined.npz"
        if not refined_path.exists():
            refined_path = in_dir / f"{sequence}_refined_accepted.npz"
        if refined_path.exists():
            r = np.load(refined_path)
            full = np.full((len(acc), 4, 4), np.nan)
            full[r["index"]] = r["T"]
            self.T_refined = full[acc]
            if np.isnan(self.T_refined).any():
                raise SystemExit(f"{refined_path}: does not cover every accepted candidate")
        self.prior_sigmas = self.cfg.get("gtsam", {}).get("prior_sigmas")

        # Candidates the prior-started verification rejected, retried from the
        # Scan Context alignment (``retry`` subcommand). Kept apart: every
        # analysis of "the accepted loop closures" above means the first group.
        self.r_i = self.r_j = np.zeros(0, dtype=np.int64)
        self.r_T = self.r_T_init = self.r_T_gt = np.zeros((0, 4, 4))
        # Retry results may come in two files: the candidates on the pipeline's
        # query stride, and optionally the others (for the stride-1 variant).
        files = sorted(in_dir.glob(f"{sequence}_retry*.npz"))
        self.retry_complete = any("_off_stride" in f.name for f in files)
        if files:
            parts = [np.load(f) for f in files]
            index = np.concatenate([r["index"] for r in parts])
            T_all = np.concatenate([r["T"] for r in parts])
            fit = np.concatenate([r["fitness"] for r in parts])
            ok = fit >= float(self.cand["fitness_threshold"])
            idx = index[ok]
            self.r_i, self.r_j = self.cand["i"][idx], self.cand["j"][idx]
            self.r_T = T_all[ok]
            self.r_T_init = self.cand["T_init"][idx]
            self.r_T_gt = _relative(self.gt, self.r_i, self.r_j)
        t = self.odom[:, :3, 3]
        self.path = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(t, axis=0), axis=1))])

    def degeneracy_edge_covariances(self, model: Model) -> list | None:
        """Per-edge odometry covariances of ``model``'s degeneracy relaxation, if any."""
        return degeneracy_edge_covariances(
            self.cfg, self.seq, self.odom, model.odom_sigmas, model.degeneracy
        )

    def loops(
        self, model: Model, source: str = "model"
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(i, j, relative poses)`` of the loop closures the graph gets under ``model``.

        ``source`` is ``"model"`` (verification or refinement result, as the
        model says), ``"icp"``, ``"refined"`` or ``"gt"``. Which loop closures
        enter is decided on the measurements, as in the pipeline; ``"gt"``
        only replaces their relative poses. Retried loop closures exist only
        with refinement (the retry refines).
        """
        if source == "model":
            source = "refined" if model.refined else "icp"
        if source in ("refined", "gt") and model.refined and self.T_refined is None:
            raise SystemExit(f"{self.seq}: no refined candidates; run refine first")
        refined = source == "refined" or (source == "gt" and model.refined)
        meas = self.T_refined if refined else self.T_icp
        i, j, T, T_init, T_gt = self.i, self.j, meas, self.T_init, self.T_gt
        if model.retry and refined and model.query_stride == 1 and not self.retry_complete:
            raise SystemExit(f"{self.seq}: retry covers the query stride only; run retry --rest")
        if model.retry and refined and len(self.r_i):
            i, j = np.concatenate([i, self.r_i]), np.concatenate([j, self.r_j])
            T = np.concatenate([T, self.r_T])
            T_init = np.concatenate([T_init, self.r_T_init])
            T_gt = np.concatenate([T_gt, self.r_T_gt])
        keep = j % model.query_stride == 0
        if model.max_drift_ratio is not None:
            drift = np.linalg.norm(T[:, :3, 3] - T_init[:, :3, 3], axis=1)
            keep &= drift <= model.max_drift_ratio * (self.path[j] - self.path[i])
        if source == "gt":
            T = T_gt
        return i[keep], j[keep], T[keep]

    def solve(
        self,
        model: Model,
        source: str = "model",
        edge_covariances: list | None = None,
        loop_sigmas=None,
    ) -> tuple[np.ndarray, PoseGraphOptimizer]:
        """Stage 3 graph as ``run_pipeline.py`` builds it for ``model``."""
        li, lj, T = self.loops(model, source)
        opt = PoseGraphOptimizer(
            odom_sigmas=list(model.odom_sigmas),
            prior_sigmas=self.prior_sigmas,
            robust_kernel=model.robust_kernel,
            robust_scale=model.robust_scale,
        )
        es = edge_covariances
        if es is None:
            es = self.degeneracy_edge_covariances(model)
        opt.build_graph(list(self.odom), edge_sigmas=es)
        ls = loop_sigmas if loop_sigmas is not None else model.loop_sigmas
        for i, j, Tij in zip(li, lj, T):
            opt.add_loop_closure(int(i), int(j), Tij, sigmas=None if ls is None else list(ls))
        return np.array(opt.optimize()), opt


# ---------------------------------------------------------------------------
# Variance component estimation
# ---------------------------------------------------------------------------

VCE_GROUPS = ("odometry_rot", "odometry_trans", "loop_rot", "loop_trans")


def _gtsam_cov(sig) -> np.ndarray:
    """Config order [tx ty tz rx ry rz] sigmas -> GTSAM order covariance."""
    return np.diag([sig[3] ** 2, sig[4] ** 2, sig[5] ** 2, sig[0] ** 2, sig[1] ** 2, sig[2] ** 2])


def _scaled_odometry_covariances(st: Study, model: Model, var_rot: float, var_trans: float):
    """The model's per-edge odometry covariances, rotation and translation blocks scaled.

    Edges downgraded for degeneracy keep their directional shape.
    """
    edges = st.degeneracy_edge_covariances(model)
    scale = np.sqrt(np.array([var_rot] * 3 + [var_trans] * 3))
    out: list = [None]
    for k in range(1, len(st.odom)):
        o = None if edges is None else edges[k]
        if o is None:
            cov = _gtsam_cov(model.odom_sigmas)
        elif isinstance(o, np.ndarray):
            cov = o
        else:
            cov = _gtsam_cov(o)
        out.append(cov * np.outer(scale, scale))
    return out


def _redundancy_by_group(opt: PoseGraphOptimizer, n_poses: int) -> dict[str, tuple[float, float]]:
    """``(sum of squared whitened residuals, redundancy)`` per VCE group.

    The redundancy of a scalar observation is ``1 - h``, where ``h`` is the
    diagonal of the hat matrix ``A Q_xx A^T`` (whitened Jacobian ``A``,
    parameter cofactor ``Q_xx`` from the joint marginal of the factor's two
    poses). Rows 0-2 of a Pose3 between factor are rotation, rows 3-5
    translation, also after whitening because every covariance used here is
    block diagonal in rotation and translation. Factor order is fixed by
    ``PoseGraphOptimizer``: one prior, ``n_poses - 1`` odometry edges, then
    the loop closures.
    """
    vals = opt.result_values
    marg = gtsam.Marginals(opt.graph, vals)
    acc = {g: [0.0, 0.0] for g in VCE_GROUPS}
    for k in range(1, opt.graph.size()):
        lin = opt.graph.at(k).linearize(vals)
        A, b = lin.jacobian()
        keys = list(lin.keys())
        jm = marg.jointMarginalCovariance(gtsam.KeyVector(keys))
        Q = np.block([[jm.at(a, c) for c in keys] for a in keys])
        h = np.einsum("ij,jk,ik->i", A, Q, A)
        kind = "odometry" if k < n_poses else "loop"
        for rows, part in ((slice(0, 3), "rot"), (slice(3, 6), "trans")):
            acc[f"{kind}_{part}"][0] += float(np.sum(b[rows] ** 2))
            acc[f"{kind}_{part}"][1] += float(np.sum(1.0 - h[rows]))
    return {g: (v[0], v[1]) for g, v in acc.items()}


def solve_vce(
    st: Study, model: Model, source: str, max_iter: int = 30, tol: float = 0.01
) -> tuple[np.ndarray, dict[str, float]]:
    """Pose graph with one variance factor per group, estimated from its own residuals.

    Iterative variance component estimation (Foerstner 1979): start from
    ``model``'s stochastic model and repeatedly rescale each group by its
    a-posteriori variance factor ``Omega_g / r_g`` until all factors are 1.
    Uses no ground truth. Loop closures start from the model's loop sigma, or
    the odometry sigma if it has none.
    """
    var = dict.fromkeys(VCE_GROUPS, 1.0)
    base_loop = model.loop_sigmas or model.odom_sigmas
    for _ in range(max_iter):
        covs = _scaled_odometry_covariances(st, model, var["odometry_rot"], var["odometry_trans"])
        lsig = [base_loop[k] * np.sqrt(var["loop_trans"]) for k in range(3)] + [
            base_loop[k] * np.sqrt(var["loop_rot"]) for k in range(3, 6)
        ]
        poses, opt = st.solve(model, source, edge_covariances=covs, loop_sigmas=lsig)
        stats = _redundancy_by_group(opt, len(poses))
        factors = {g: stats[g][0] / stats[g][1] for g in VCE_GROUPS}
        for g in VCE_GROUPS:
            var[g] *= factors[g]
        if all(abs(f - 1.0) < tol for f in factors.values()):
            break
    o = model.odom_sigmas
    sig = {
        "vce_odometry_rot_mrad": 1e3 * o[3] * np.sqrt(var["odometry_rot"]),
        "vce_odometry_trans_m": o[0] * np.sqrt(var["odometry_trans"]),
        "vce_loop_rot_mrad": 1e3 * base_loop[3] * np.sqrt(var["loop_rot"]),
        "vce_loop_trans_m": base_loop[0] * np.sqrt(var["loop_trans"]),
    }
    return poses, sig


# ---------------------------------------------------------------------------
# Constraint errors
# ---------------------------------------------------------------------------


def _revisit_events(i: np.ndarray, j: np.ndarray, gap: int = 50) -> np.ndarray:
    """Label loop closures by revisit event.

    Two closures belong to the same event when both their endpoints are
    within ``gap`` frames of each other (single linkage).
    """
    n = len(i)
    parent = np.arange(n)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    order = np.argsort(j, kind="stable")
    for a_pos, a in enumerate(order):
        for b in order[a_pos + 1 :]:
            if j[b] - j[a] > gap:
                break
            if abs(int(i[b]) - int(i[a])) <= gap:
                parent[find(a)] = find(b)
    roots = np.array([find(x) for x in range(n)])
    _, labels = np.unique(roots, return_inverse=True)
    return labels


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def _lag_autocorrelation(e: np.ndarray, st: Study, labels: np.ndarray, lag: int) -> float:
    """Autocorrelation of the error at ``lag`` loop closures within a revisit event.

    Loop closures of an event are ordered by query frame, each error
    component is demeaned per event, and the correlation is averaged over
    the components, weighting events by size. Events with fewer than
    ``lag + 10`` loop closures are skipped.
    """
    num = den = 0.0
    for ev in np.unique(labels):
        m = np.flatnonzero(labels == ev)
        if len(m) < lag + 10:
            continue
        x = e[m[np.argsort(st.j[m], kind="stable")]]
        x = x - x.mean(axis=0)
        r = [np.corrcoef(x[:-lag, k], x[lag:, k])[0, 1] for k in range(x.shape[1])]
        num += len(m) * float(np.nanmean(r))
        den += len(m)
    return num / den if den else float("nan")


def _constraint_row(st: Study) -> dict:
    """Errors of the accepted loop closures (stride 1) against ground truth."""
    acc = st.cand["accepted"]
    labels = _revisit_events(st.i, st.j)
    e_prior = _relative_error(st.T_init, st.T_gt)
    e_icp = _relative_error(st.T_icp, st.T_gt)
    row = {
        "sequence": st.seq,
        "candidates": int(len(acc)),
        "accepted": int(acc.sum()),
        "events": int(labels.max() + 1),
    }
    for name, e in (("prior", e_prior), ("icp", e_icp)):
        row[f"{name}_t_median_m"] = round(float(np.median(np.linalg.norm(e[:, 3:], axis=1))), 3)
    row["icp_r_median_mrad"] = round(
        1e3 * float(np.median(np.linalg.norm(e_icp[:, :3], axis=1))), 2
    )
    t_icp = np.linalg.norm(e_icp[:, 3:], axis=1)
    row["spearman_icp_error_vs_fitness"] = round(_spearman(t_icp, st.cand["fitness"][acc]), 2)
    if st.T_refined is not None:
        e_ref = _relative_error(st.T_refined, st.T_gt)
        row["refined_t_median_m"] = round(float(np.median(np.linalg.norm(e_ref[:, 3:], axis=1))), 3)
        row["refined_r_median_mrad"] = round(
            1e3 * float(np.median(np.linalg.norm(e_ref[:, :3], axis=1))), 2
        )
        # In the frame of loop end i, z is up: vertical vs horizontal disagreement.
        dz = np.abs(st.T_gt[:, 2, 3] - st.T_refined[:, 2, 3])
        dxy = np.linalg.norm(st.T_gt[:, :2, 3] - st.T_refined[:, :2, 3], axis=1)
        row["refined_vertical_median_m"] = round(float(np.median(dz)), 3)
        row["refined_horizontal_median_m"] = round(float(np.median(dxy)), 3)
        row["refined_vertical_max_m"] = round(float(dz.max()), 2)
        row["scan_dz_max_abs_m"] = round(float(np.abs(st.T_refined[:, 2, 3]).max()), 3)
        shift = np.linalg.norm(st.T_refined[:, :3, 3] - st.T_init[:, :3, 3], axis=1)
        row["refined_max_shift_from_prior_m"] = round(float(shift.max()), 2)
        for lag in (1, 10):
            for part, cols in (("t", slice(3, 6)), ("r", slice(0, 3))):
                ac = _lag_autocorrelation(e_ref[:, cols], st, labels, lag)
                row[f"refined_{part}_autocorr_lag{lag}"] = round(ac, 2)
    return row


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------


def _ape_se3_horizontal(est: np.ndarray, gt: np.ndarray) -> float:
    """SE(3)-aligned APE of the horizontal position only (Velodyne z is up)."""
    traj = PosePath3D(poses_se3=list(est))
    traj.align(PosePath3D(poses_se3=list(gt)), correct_scale=False)
    d = np.asarray(traj.positions_xyz)[:, :2] - gt[:, :2, 3]
    return float(np.sqrt(np.mean(np.sum(d**2, axis=1))))


# Segments of every evaluated (variant, sequence), for the pooled 00-10 rows.
_SEGMENTS: dict[tuple[str, str], object] = {}


def _trajectory_row(st: Study, label: str, est_velo: np.ndarray, n_loops: int) -> dict:
    root = Path(st.cfg["data"]["kitti_root"]).expanduser()
    row, seg = _evaluate_sequence(root, st.seq, est_velo)
    _SEGMENTS[(label, st.seq)] = seg
    rpe = evaluate_odometry(list(est_velo), list(st.gt))["rpe"]["rmse"]
    return {
        "variant": label,
        "loops": n_loops,
        **row,
        "ape_se3_horizontal_m": round(_ape_se3_horizontal(est_velo, st.gt), 4),
        "rpe_1frame_m": round(float(rpe), 4),
    }


def _pooled_rows(cfg: dict, labels: list[str]) -> list[dict]:
    """KITTI relative error pooled over 00-10 for every variant.

    The sequences without loop closures enter with their odometry, which is
    what Stage 3 returns for them.
    """
    from src.benchmarks.kitti_metrics import KittiSegmentErrors

    root = Path(cfg["data"]["kitti_root"]).expanduser()
    for seq in (f"{k:02d}" for k in range(11)):
        if seq not in SEQUENCES_WITH_LOOPS:
            odom = _load_stage2(_sequence_config(cfg, seq), seq)
            _SEGMENTS[("odometry", seq)] = _evaluate_sequence(root, seq, odom)[1]
    rows = []
    for label in labels:
        seqs = [f"{k:02d}" for k in range(11)]
        parts = [_SEGMENTS.get((label, q), _SEGMENTS[("odometry", q)]) for q in seqs]
        pooled = KittiSegmentErrors.pool(parts)
        rows.append(
            {
                "variant": label,
                "sequence": "00-10",
                "t_rel_pct": round(pooled.t_rel_percent(), 4),
                "r_rel_deg_per_100m": round(pooled.r_rel_deg_per_100m(), 4),
            }
        )
    return rows


# A retried candidate whose two scans are farther apart than this on the ground
# truth (horizontally) is a false match: Scan Context matched two different places.
FALSE_MATCH_M = 20.0


def _pct(x: float) -> float:
    return round(100 * float(x), 3)


def _retry_row(st: Study, model: Model) -> dict:
    """What the retry adds on the query stride, and what the drift check removes.

    Ground truth only labels the matches (two different places or not); the
    drift check itself uses the odometry alone.
    """
    on = st.r_j % model.query_stride == 0
    i, j = st.r_i[on], st.r_j[on]
    drift = np.linalg.norm(st.r_T[on, :3, 3] - st.r_T_init[on, :3, 3], axis=1)
    ratio = drift / (st.path[j] - st.path[i])
    gt_dist = np.linalg.norm(st.gt[j, :2, 3] - st.gt[i, :2, 3], axis=1)
    false = gt_dist > FALSE_MATCH_M
    c = st.cand
    retried = int(((~c["accepted"]) & (c["j"] % model.query_stride == 0)).sum())
    thr = model.max_drift_ratio
    keep = ratio <= thr if thr is not None else np.ones_like(ratio, dtype=bool)
    err = np.zeros(0)
    if on.any():
        err = np.linalg.norm(_relative_error(st.r_T[on], st.r_T_gt[on])[:, 3:], axis=1)
    return {
        "sequence": st.seq,
        "retried": retried,
        "pass_verification": int(on.sum()),
        "false_matches": int(false.sum()),
        "true_max_drift_ratio_pct": _pct(ratio[~false].max()) if (~false).any() else "",
        "false_min_drift_ratio_pct": _pct(ratio[false].min()) if false.any() else "",
        "kept_by_drift_check": int(keep.sum()),
        "false_kept": int((keep & false).sum()),
        "true_rejected": int((~keep & ~false).sum()),
        "kept_drift_min_m": round(float(drift[keep].min()), 2) if keep.any() else "",
        "kept_drift_max_m": round(float(drift[keep].max()), 2) if keep.any() else "",
        "kept_error_median_m": round(float(np.median(err[keep])), 3) if keep.any() else "",
        "kept_error_max_m": round(float(err[keep].max()), 3) if keep.any() else "",
    }


def _variants(after: Model) -> dict[str, tuple[Model, str]]:
    """Graph variants: (model, loop source). Leave-one-out ablations start from ``after``."""
    return {
        "before": (BEFORE, "model"),
        "before, GT loops": (BEFORE, "gt"),
        "after": (after, "model"),
        "after, GT loops": (after, "gt"),
        "after without refinement": (after, "icp"),
        "after without query stride": (replace(after, query_stride=1), "model"),
        "after without odometry sigma": (replace(after, odom_sigmas=BEFORE.odom_sigmas), "model"),
        "after without loop sigma": (replace(after, loop_sigmas=BEFORE.odom_sigmas), "model"),
        "after with Cauchy kernel": (replace(after, robust_kernel="cauchy"), "model"),
        "after without retry": (replace(after, retry=False), "model"),
        "after without drift check": (replace(after, max_drift_ratio=None), "model"),
        "after with degeneracy relaxation": (replace(after, degeneracy="body"), "model"),
        "after with degeneracy relaxation, map-frame direction": (
            replace(after, degeneracy="map"),
            "model",
        ),
    }


def cmd_analyze(args: argparse.Namespace) -> None:
    base_cfg = load_config(args.config)
    after = model_from_config(base_cfg)
    print(f"after = {after}")
    rows, constraint_rows, vce_rows, retry_rows = [], [], [], []
    in_dir = Path(args.in_dir)
    for seq in args.sequences:
        st = Study(base_cfg, seq, in_dir)
        print(f"\n=== {seq}: {len(st.cand['i'])} candidates, {len(st.i)} accepted ===")
        c = _constraint_row(st)
        constraint_rows.append(c)
        if after.retry:
            retry_rows.append(_retry_row(st, after))
            print("  retry: " + "  ".join(f"{k}={v}" for k, v in retry_rows[-1].items()))
        print("  " + "  ".join(f"{k}={v}" for k, v in c.items() if k != "sequence"))

        rows.append(_trajectory_row(st, "odometry", st.odom, 0))
        for label, (model, source) in _variants(after).items():
            poses, _ = st.solve(model, source)
            rows.append(_trajectory_row(st, label, poses, len(st.loops(model, source)[0])))
            if label == "after":
                cached = LayeredCache(
                    st.cfg.get("cache", {}).get("root", "cache/kitti"), seq
                ).load_optimized(st.cfg)
                if cached is not None:
                    d = np.linalg.norm(cached[:, :3, 3] - poses[:, :3, 3], axis=1).max()
                    rows[-1]["max_diff_to_pipeline_m"] = float(f"{d:.2e}")
                    print(f"  'after' vs cached Stage 3 of the current config: max {d:.2e} m")
        if args.vce:
            for source in ("icp", "refined"):
                poses, sig = solve_vce(st, BEFORE, source)
                n = len(st.loops(BEFORE)[0])
                label = f"before, VCE weights, {source} loops"
                rows.append({**_trajectory_row(st, label, poses, n)})
                vce_rows.append(
                    {"sequence": seq, "loops": source, **{k: round(v, 4) for k, v in sig.items()}}
                )
                print(f"  VCE ({source}): " + ", ".join(f"{k} {v:.3g}" for k, v in sig.items()))
        for r in rows:
            if r.get("sequence") == seq:
                print(
                    f"  {r['variant']:>38s}: loops {r['loops']:5d}  t_rel {r['t_rel_pct']:.3f} %  "
                    f"APE SE(3) {r['ape_se3_m']:.2f} m (horizontal {r['ape_se3_horizontal_m']:.2f})"
                    f"  first {r['ape_first_m']:.2f} m  RPE {r['rpe_1frame_m']:.4f} m"
                )

    if set(SEQUENCES_WITH_LOOPS) <= set(args.sequences):
        labels = list(dict.fromkeys(r["variant"] for r in rows))
        for r in _pooled_rows(base_cfg, labels):
            rows.append(r)
            print(f"  pooled 00-10 {r['variant']:>38s}: t_rel {r['t_rel_pct']:.4f} %")

    out = Path(args.out_dir) if args.out_dir else None
    if out:
        _write_csv(out / "graph_variants.csv", rows)
        _write_csv(out / "constraints.csv", constraint_rows)
        if vce_rows:
            _write_csv(out / "vce.csv", vce_rows)
        if retry_rows:
            _write_csv(out / "retry.csv", retry_rows)


# ---------------------------------------------------------------------------
# plot
# ---------------------------------------------------------------------------


def _se3_error_by_frame(est: np.ndarray, gt: np.ndarray) -> np.ndarray:
    traj = PosePath3D(poses_se3=list(est))
    traj.align(PosePath3D(poses_se3=list(gt)), correct_scale=False)
    return np.linalg.norm(np.asarray(traj.positions_xyz) - gt[:, :3, 3], axis=1)


def cmd_plot(args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base_cfg = load_config(args.config)
    after = model_from_config(base_cfg)
    st = Study(base_cfg, args.sequence, Path(args.in_dir))
    dist = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(st.gt[:, :3, 3], axis=0), axis=1))]
    series = [
        ("Stage 2 odometry", st.odom, "#7a7974", "-"),
        ("Stage 3 before", st.solve(BEFORE)[0], "#eb6834", "-"),
        ("Stage 3 after", st.solve(after)[0], "#2a78d6", "-"),
        ("after, ground-truth loops", st.solve(after, "gt")[0], "#1baf7a", "--"),
    ]
    ink, muted, grid = "#0b0b0b", "#52514e", "#e4e3df"
    fig, ax = plt.subplots(figsize=(8.0, 3.6), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    # Shade the stretches that loop closures connect (both passes).
    for ends in (st.i, st.j):
        lo, hi = dist[ends.min()], dist[ends.max()]
        ax.axvspan(lo, hi, color="#f1f0ec", zorder=0, lw=0)
    for label, poses, color, ls in series:
        err = _se3_error_by_frame(poses, st.gt)
        ax.plot(dist, err, color=color, ls=ls, lw=1.6, label=label, zorder=3)
    ax.set_xlabel("distance along the sequence [m]", color=muted)
    ax.set_ylabel("position error after\nSE(3) alignment [m]", color=muted)
    ax.set_title(
        f"KITTI {args.sequence}: position error along the trajectory "
        "(shaded: stretches joined by loop closures)",
        color=ink,
        fontsize=10,
        loc="left",
    )
    ax.grid(axis="y", color=grid, lw=0.8)
    ax.set_ylim(bottom=0)
    ax.set_xlim(0, dist[-1])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(grid)
    ax.tick_params(colors=muted, labelsize=8)
    leg = ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
    for t in leg.get_texts():
        t.set_color(ink)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=fig.get_facecolor())
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# gps: ground-truth quality at the revisits
# ---------------------------------------------------------------------------

# OxTS position mode codes (OxTS NCOM); only these occur in the KITTI drives
# used here.
POSMODE = {0: "none", 4: "differential", 5: "RTK float", 6: "RTK integer"}
_OXTS_POSMODE_COL = 27


def _oxts_posmode_per_frame(st: Study, raw_root: Path) -> tuple[np.ndarray, float, float]:
    """GPS position mode at every odometry frame, the time offset and its fit residual.

    The raw drives here are the unsynchronised 100 Hz OxTS logs, so the
    odometry frames are matched in time with the offset of the
    georeferencing (``src/export/georef.py``): a coarse speed match refined
    by the rigid fit of the ground-truth trajectory to the RTK positions.
    The speed match alone is about 0.1 s off, about 1 m along track.
    """
    oxts = oxts_dir_for_sequence(raw_root, st.seq)
    if oxts is None:
        raise SystemExit(f"{st.seq}: no OxTS log under {raw_root}")
    lever = load_imu_lever_arm(raw_root, ODOM_TO_RAW[st.seq][0])
    g = georeference_from_gnss(st.gt, st.timestamps, oxts, lever)
    data, t_ox = load_oxts_monotonic(oxts)
    idx = np.searchsorted(t_ox, st.timestamps + g.time_offset_s).clip(0, len(t_ox) - 1)
    return (
        data[idx, _OXTS_POSMODE_COL].astype(int),
        float(g.time_offset_s),
        float(g.residual_horizontal_rms_m),
    )


def cmd_gps(args: argparse.Namespace) -> None:
    base_cfg = load_config(args.config)
    raw_root = Path(args.kitti_raw).expanduser()
    rows = []
    for seq in args.sequences:
        st = Study(base_cfg, seq, Path(args.in_dir))
        if st.T_refined is None:
            raise SystemExit(f"{seq}: run refine first")
        pm, offset, fit_rms = _oxts_posmode_per_frame(st, raw_root)
        share = {POSMODE.get(m, str(m)): float(np.mean(pm == m)) for m in np.unique(pm)}
        print(f"{seq}: time offset {offset:.3f} s (fit RMS {fit_rms:.2f} m), posmode {share}")
        fixed = (pm[st.i] == 6) & (pm[st.j] == 6)
        for label, m in (("both ends RTK integer", fixed), ("other", ~fixed)):
            if not m.any():
                continue
            dz = np.abs(st.T_gt[m, 2, 3] - st.T_refined[m, 2, 3])
            dxy = np.linalg.norm(st.T_gt[m, :2, 3] - st.T_refined[m, :2, 3], axis=1)
            row = {
                "sequence": seq,
                "loops": label,
                "n": int(m.sum()),
                "rtk_integer_share_of_frames": round(share.get("RTK integer", 0.0), 3),
                "gt_vs_refined_vertical_median_m": round(float(np.median(dz)), 3),
                "gt_vs_refined_horizontal_median_m": round(float(np.median(dxy)), 3),
            }
            # RMS error per axis against ground truth, [rx ry rz] mrad and [tx ty tz] m.
            for name, T in (("icp", st.T_icp), ("refined", st.T_refined)):
                e = _relative_error(T[m], st.T_gt[m])
                rms = np.sqrt(np.mean(e**2, axis=0))
                for k, ax in enumerate(("rx", "ry", "rz")):
                    row[f"{name}_rms_{ax}_mrad"] = round(1e3 * float(rms[k]), 2)
                for k, ax in enumerate(("tx", "ty", "tz")):
                    row[f"{name}_rms_{ax}_m"] = round(float(rms[3 + k]), 3)
            rows.append(row)
            print("  " + "  ".join(f"{k}={v}" for k, v in row.items() if k != "sequence"))
    if args.out_dir:
        _write_csv(Path(args.out_dir) / "gps.csv", rows)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("detect", help="run SC + ICP on every candidate and store the results")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--sequences", nargs="+", default=list(SEQUENCES_WITH_LOOPS))
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))
    p.set_defaults(func=cmd_detect)

    p = sub.add_parser("refine", help="apply the Stage 3 refinement to stored candidates")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--sequences", nargs="+", default=list(SEQUENCES_WITH_LOOPS))
    p.add_argument("--in-dir", default=str(DEFAULT_OUT))
    p.add_argument("--out-dir", default=None, help="defaults to --in-dir")
    p.add_argument("--accepted-only", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="first N pairs only (smoke test)")
    p.set_defaults(func=cmd_refine)

    p = sub.add_parser("retry", help="retry rejected candidates from the Scan Context alignment")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--sequences", nargs="+", default=list(SEQUENCES_WITH_LOOPS))
    p.add_argument("--in-dir", default=str(DEFAULT_OUT))
    p.add_argument("--out-dir", default=None)
    p.add_argument("--stride", type=int, default=1, help="only candidates with j % stride == 0")
    p.add_argument(
        "--rest", action="store_true", help="the candidates with j % stride != 0 instead"
    )
    p.set_defaults(func=cmd_retry)

    p = sub.add_parser("analyze", help="constraint errors and pose-graph variants")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--sequences", nargs="+", default=list(SEQUENCES_WITH_LOOPS))
    p.add_argument("--in-dir", default=str(DEFAULT_OUT))
    p.add_argument("--out-dir", default=None, help="write graph_variants/constraints/vce CSVs")
    p.add_argument("--vce", action="store_true", help="also run variance component estimation")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("gps", help="ground-truth quality at revisits from the GPS fix status")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--sequences", nargs="+", default=["00", "05"])
    p.add_argument("--in-dir", default=str(DEFAULT_OUT))
    p.add_argument("--kitti-raw", default="~/data/kitti_raw")
    p.add_argument("--out-dir", default=None, help="write gps.csv")
    p.set_defaults(func=cmd_gps)

    p = sub.add_parser("plot", help="position error along one sequence, before and after")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--sequence", default="06")
    p.add_argument("--in-dir", default=str(DEFAULT_OUT))
    p.add_argument("--out", default="docs/figures/loop_closure_06.png")
    p.set_defaults(func=cmd_plot)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
