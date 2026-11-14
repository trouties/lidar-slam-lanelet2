#!/usr/bin/env python3
"""Do the pose-graph covariances describe the actual trajectory errors?

A pose graph's covariance is a statement about the error: the 1-sigma of a
segment's relative pose, or of a pose once the datum is fixed. This script
compares that statement with the error against the KITTI ground truth.

  segments  Every devkit-style segment (100, 200, 400 and 800 m of path,
            one starting at every 10th frame): the relative-pose error and
            the covariance the graph gives that relative pose. Graphs:
              odometry      the Stage 3 graph without loop closures; its
                            optimum is the Stage 2 odometry (00-10)
              stage3        the Stage 3 graph as run_pipeline.py builds it
              gt_loops      stage3 with every loop closure's relative pose
                            replaced by the ground-truth one
              sqrt_n        stage3 with the loop-closure sigma of each
                            revisit scaled by sqrt(n), n the number of its
                            loop closures
            The last two separate causes; they are diagnostics, not fixes.
            Errors are taken against the KITTI ground truth and, where the
            KITTI raw OxTS log exists, against its metric version
            (src/benchmarks/metric_reference.py).
  control   The odometry graph with ground-truth positions as control points
            every S metres of path (position only, as a GNSS fix would
            give): the error between control points and its predicted sigma.
  motion    Vertical motion per frame in the body frame of the previous
            pose, ground truth and odometry. A car does not move along its
            own vertical axis by more than suspension travel and mounting
            tilt allow, so a reference that does is in error.
  plot      One sequence along the drive: the horizontal error and the
            predicted 2-sigma band for odometry, Stage 3 and control points.

Axes are the Velodyne body axes at the segment end (or at the pose): x
forward, y left, z up. Consecutive segments overlap and share errors, so the
statistics describe, they do not test: an "in the 95 % ellipsoid" share far
from 95 % means a miscalibrated covariance, but there is no meaningful
p-value.

Ground truth enters as the reference everywhere, and in ``control`` also as
the control points.

Usage:
  python -m scripts.uncertainty_study segments --out-dir benchmarks/uncertainty
  python -m scripts.uncertainty_study control --out-dir benchmarks/uncertainty
  python -m scripts.uncertainty_study motion --out-dir benchmarks/uncertainty
  python -m scripts.uncertainty_study plot --sequence 05 --out docs/figures/uncertainty_05.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import gtsam
import numpy as np

from scripts.loop_closure_study import (
    SEQUENCES_WITH_LOOPS,
    Study,
    _load_stage2,
    _revisit_events,
    _sequence_config,
    degeneracy_edge_covariances,
    model_from_config,
)
from scripts.run_pipeline import load_config
from src.benchmarks.metric_reference import metric_ground_truth
from src.data.kitti_loader import KITTIDataset
from src.data.kitti_raw import ODOM_TO_RAW
from src.export.georef import load_imu_lever_arm, oxts_dir_for_sequence
from src.optimization.pose_graph import PoseGraphOptimizer

SEQUENCES = tuple(f"{i:02d}" for i in range(11))
SEGMENT_LENGTHS_M = (100, 200, 400, 800)
STEP_FRAMES = 10
SPACINGS_M = (100, 200, 400, 800)
CHI2_3_95 = 7.8147  # 95 % quantile of chi-square with 3 degrees of freedom
AXES = ("fwd", "lat", "up")
GRAPHS = ("odometry", "stage3", "gt_loops", "sqrt_n")


class Track:
    """Stage 2 odometry, ground truth and the odometry edge noise of one sequence."""

    def __init__(self, cfg: dict, sequence: str, raw_root: Path, loop_dir: Path) -> None:
        self.seq = sequence
        self.cfg = _sequence_config(cfg, sequence)
        self.odom = _load_stage2(self.cfg, sequence)
        ds = KITTIDataset.from_config(self.cfg["data"], sequence)
        self.Tr = ds.calibration["Tr"]
        Tr_inv = np.linalg.inv(self.Tr)
        self.gt = np.array([Tr_inv @ g @ self.Tr for g in ds.poses])
        self.timestamps = np.asarray(ds.timestamps)
        self.model = model_from_config(self.cfg)
        self.edges = degeneracy_edge_covariances(
            self.cfg, sequence, self.odom, self.model.odom_sigmas, self.model.degeneracy
        )
        self.prior_sigmas = self.cfg.get("gtsam", {}).get("prior_sigmas")
        self.study = Study(cfg, sequence, loop_dir) if sequence in SEQUENCES_WITH_LOOPS else None
        self.references = {"kitti": self.gt}
        oxts = oxts_dir_for_sequence(raw_root, sequence)
        if oxts is not None:
            lever = load_imu_lever_arm(raw_root, ODOM_TO_RAW[sequence][0])
            self.references["metric"] = metric_ground_truth(
                self.gt, self.timestamps, oxts, lever
            ).poses

    def path(self) -> np.ndarray:
        """Distance along the ground-truth path [m], as the devkit measures it."""
        t = self.gt[:, :3, 3]
        return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(t, axis=0), axis=1))])

    def odometry_graph(self) -> PoseGraphOptimizer:
        opt = PoseGraphOptimizer(
            odom_sigmas=list(self.model.odom_sigmas), prior_sigmas=self.prior_sigmas
        )
        opt.build_graph(list(self.odom), edge_sigmas=self.edges)
        return opt

    def graph(self, name: str) -> tuple[np.ndarray, PoseGraphOptimizer]:
        """Optimised poses and optimiser of one of :data:`GRAPHS`."""
        if name == "odometry":
            opt = self.odometry_graph()
            poses = np.array(opt.optimize())
            assert np.abs(poses[:, :3, 3] - self.odom[:, :3, 3]).max() < 1e-6
            return poses, opt
        if self.study is None:
            raise ValueError(f"{self.seq} has no loop closures")
        if name == "stage3":
            poses, opt = self.study.solve(self.model)
            self._check_against_pipeline(poses)
            return poses, opt
        source = "gt" if name == "gt_loops" else "model"
        li, lj, T = self.study.loops(self.model, source)
        scale = np.ones(len(li))
        if name == "sqrt_n":
            event = _revisit_events(li, lj)
            scale = np.sqrt(np.bincount(event)[event])
        opt = self.odometry_graph()
        for i, j, Tij, s in zip(li, lj, T, scale):
            opt.add_loop_closure(
                int(i), int(j), Tij, sigmas=list(np.array(self.model.loop_sigmas) * s)
            )
        return np.array(opt.optimize()), opt

    def _check_against_pipeline(self, poses: np.ndarray) -> None:
        out_dir = Path(self.cfg.get("output", {}).get("dir", "results"))
        path = out_dir / f"poses_optimized_{self.seq}.txt"
        if not path.exists():
            print(f"  {self.seq}: {path} missing, Stage 3 not checked against the pipeline")
            return
        cam = np.loadtxt(path).reshape(-1, 3, 4)
        T = np.tile(np.eye(4), (len(cam), 1, 1))
        T[:, :3, :] = cam
        velo = np.linalg.inv(self.Tr) @ T @ self.Tr
        d = float(np.abs(velo[:, :3, 3] - poses[:, :3, 3]).max())
        if d > 1e-4:
            raise SystemExit(
                f"{self.seq}: Stage 3 differs from {path} by {d:.3g} m; rerun the pipeline"
            )


def _errors(gt_rel: np.ndarray, est_rel: np.ndarray) -> np.ndarray:
    """``[rx, ry, rz, x, y, z]`` of ``gt_rel^-1 est_rel``: the error in the graph's convention.

    The estimate is the truth perturbed on the right, so the error is in the
    body axes of the (true) end pose, like the graph's covariance.
    """
    d = np.linalg.inv(gt_rel) @ est_rel
    return np.concatenate([gtsam.Rot3.Logmap(gtsam.Rot3(d[:3, :3])), d[:3, 3]])


def _segments(track: Track, poses: np.ndarray, opt: PoseGraphOptimizer) -> list[dict]:
    """One record per devkit segment: errors per reference and the predicted covariance."""
    dist = track.path()
    out = []
    for first in range(0, len(poses), STEP_FRAMES):
        ends = []
        for length in SEGMENT_LENGTHS_M:
            beyond = np.nonzero(dist[first:] > dist[first] + length)[0]
            if beyond.size:
                ends.append((first + int(beyond[0]), length))
        if not ends:
            continue
        cov = opt.relative_pose_covariances(first, [j for j, _ in ends])
        for j, length in ends:
            est = np.linalg.inv(poses[first]) @ poses[j]
            rec = {"first": first, "last": j, "length": length, "cov": cov[j]}
            for ref, gt in track.references.items():
                rec[ref] = _errors(np.linalg.inv(gt[first]) @ gt[j], est)
            out.append(rec)
    return out


def _summary(records: list[dict], ref: str) -> dict:
    """Error and predicted sigma per axis, normalised error, 95 % ellipsoid share."""
    e = np.array([r[ref] for r in records])
    C = np.array([r["cov"] for r in records])
    sig = np.sqrt(np.einsum("nii->ni", C))
    P = C[:, 3:, 3:]
    nees = np.einsum("ni,ni->n", e[:, 3:], np.linalg.solve(P, e[:, 3:, None])[..., 0])
    row = {"segments": len(e)}
    for k, ax in enumerate(AXES):
        row[f"err_rms_{ax}_m"] = round(float(np.sqrt(np.mean(e[:, 3 + k] ** 2))), 4)
    for k, ax in enumerate(AXES):
        row[f"sigma_rms_{ax}_m"] = round(float(np.sqrt(np.mean(sig[:, 3 + k] ** 2))), 4)
    for k, ax in enumerate(AXES):
        row[f"z_rms_{ax}"] = round(float(np.sqrt(np.mean((e[:, 3 + k] / sig[:, 3 + k]) ** 2))), 3)
    row["err_rms_yaw_mrad"] = round(1e3 * float(np.sqrt(np.mean(e[:, 2] ** 2))), 3)
    row["sigma_rms_yaw_mrad"] = round(1e3 * float(np.sqrt(np.mean(sig[:, 2] ** 2))), 3)
    row["z_rms_yaw"] = round(float(np.sqrt(np.mean((e[:, 2] / sig[:, 2]) ** 2))), 3)
    row["in_95_pct"] = round(100.0 * float(np.mean(nees <= CHI2_3_95)), 1)
    return row


def _pooled_label(seqs: list[str]) -> str:
    return "00-10" if tuple(sorted(seqs)) == SEQUENCES else ",".join(sorted(seqs))


def cmd_segments(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    raw_root = Path(args.kitti_raw).expanduser()
    records: dict[tuple[str, str], list[dict]] = {}
    refs: dict[str, list[str]] = {}
    for seq in args.sequences:
        track = Track(cfg, seq, raw_root, Path(args.loop_dir))
        refs[seq] = list(track.references)
        for name in GRAPHS:
            if name != "odometry" and track.study is None:
                continue
            poses, opt = track.graph(name)
            records[(seq, name)] = _segments(track, poses, opt)
            s = _summary(records[(seq, name)], "kitti")
            print(
                f"{seq} {name:9s} z_rms fwd/lat/up {s['z_rms_fwd']:.2f}/{s['z_rms_lat']:.2f}/"
                f"{s['z_rms_up']:.2f}  in 95 % ellipsoid {s['in_95_pct']:.0f} %"
            )

    rows = []

    def emit(label: str, name: str, ref: str, recs: list[dict]) -> None:
        for length in SEGMENT_LENGTHS_M:
            sel = [r for r in recs if r["length"] == length]
            if len(sel) >= args.min_segments:
                rows.append(
                    {
                        "sequence": label,
                        "graph": name,
                        "reference": ref,
                        "length_m": length,
                        **_summary(sel, ref),
                    }
                )

    for (seq, name), recs in records.items():
        for ref in refs[seq]:
            emit(seq, name, ref, recs)
    # Pooled like the devkit: all segments of all sequences together.
    loop_seqs = [s for s in args.sequences if s in SEQUENCES_WITH_LOOPS]
    for ref in ("kitti", "metric"):
        for name, seqs in (("odometry", list(args.sequences)), *((g, loop_seqs) for g in GRAPHS)):
            seqs = [s for s in seqs if ref in refs[s] and (s, name) in records]
            if len(seqs) > 1:
                emit(_pooled_label(seqs), name, ref, [r for s in seqs for r in records[(s, name)]])
    _write_csv(Path(args.out_dir) / "segments.csv", rows)


def _control_frames(dist: np.ndarray, spacing: float) -> np.ndarray:
    """Frame 0, the first frame past every multiple of ``spacing`` metres, and the last frame."""
    marks = np.arange(spacing, dist[-1], spacing)
    idx = np.searchsorted(dist, marks)
    return np.unique(np.concatenate([[0], idx, [len(dist) - 1]]))


def control_graph(
    track: Track, spacing: float, sigma: float
) -> tuple[np.ndarray, PoseGraphOptimizer, np.ndarray]:
    """Odometry graph plus ground-truth positions as control points every ``spacing`` metres."""
    ctrl = _control_frames(track.path(), spacing)
    opt = track.odometry_graph()
    for k in ctrl:
        opt.add_position_prior(int(k), track.gt[k, :3, 3], [sigma] * 3)
    return np.array(opt.optimize()), opt, ctrl


def cmd_control(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    raw_root = Path(args.kitti_raw).expanduser()
    rows = []
    pooled: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {s: [] for s in SPACINGS_M}
    for seq in args.sequences:
        track = Track(cfg, seq, raw_root, Path(args.loop_dir))
        for spacing in SPACINGS_M:
            poses, opt, ctrl = control_graph(track, spacing, args.sigma)
            if len(ctrl) < 3:
                continue
            frames = np.setdiff1d(np.arange(0, len(poses), args.frame_step), ctrl)
            cov_w = opt.get_position_marginals([int(k) for k in frames])
            R = track.gt[frames, :3, :3]
            e = np.einsum("nji,nj->ni", R, poses[frames, :3, 3] - track.gt[frames, :3, 3])
            C = np.einsum("nji,njk,nkl->nil", R, np.array([cov_w[int(k)] for k in frames]), R)
            pooled[spacing].append((e, C))
            rows.append(
                {
                    "sequence": seq,
                    "spacing_m": spacing,
                    "gaps": len(ctrl) - 1,
                    **_control_summary(e, C),
                }
            )
            r = rows[-1]
            print(
                f"{seq} every {spacing:4d} m: horizontal error max "
                f"{r['err_max_horizontal_m']:.2f} m, z_rms fwd/lat/up "
                f"{r['z_rms_fwd']:.2f}/{r['z_rms_lat']:.2f}/{r['z_rms_up']:.2f}"
            )
    for spacing in SPACINGS_M:
        if len(pooled[spacing]) > 1:
            e = np.concatenate([p[0] for p in pooled[spacing]])
            C = np.concatenate([p[1] for p in pooled[spacing]])
            gaps = sum(r["gaps"] for r in rows if r["spacing_m"] == spacing)
            seqs = [r["sequence"] for r in rows if r["spacing_m"] == spacing]
            rows.append(
                {
                    "sequence": _pooled_label(seqs),
                    "spacing_m": spacing,
                    "gaps": gaps,
                    **_control_summary(e, C),
                }
            )
    _write_csv(Path(args.out_dir) / "control_points.csv", rows)


def _control_summary(e: np.ndarray, C: np.ndarray) -> dict:
    sig = np.sqrt(np.einsum("nii->ni", C))
    nees = np.einsum("ni,ni->n", e, np.linalg.solve(C, e[..., None])[..., 0])
    row = {
        "frames": len(e),
        "err_max_horizontal_m": round(float(np.hypot(e[:, 0], e[:, 1]).max()), 3),
    }
    for k, ax in enumerate(AXES):
        row[f"err_rms_{ax}_m"] = round(float(np.sqrt(np.mean(e[:, k] ** 2))), 4)
    for k, ax in enumerate(AXES):
        row[f"sigma_rms_{ax}_m"] = round(float(np.sqrt(np.mean(sig[:, k] ** 2))), 4)
    for k, ax in enumerate(AXES):
        row[f"z_rms_{ax}"] = round(float(np.sqrt(np.mean((e[:, k] / sig[:, k]) ** 2))), 3)
    row["in_95_pct"] = round(100.0 * float(np.mean(nees <= CHI2_3_95)), 1)
    return row


def cmd_motion(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    rows = []
    for seq in args.sequences:
        c = _sequence_config(cfg, seq)
        ds = KITTIDataset.from_config(c["data"], seq)
        Tr_inv = np.linalg.inv(ds.calibration["Tr"])
        gt = Tr_inv @ np.asarray(ds.poses) @ ds.calibration["Tr"]
        for source, P in (("ground_truth", gt), ("odometry", _load_stage2(c, seq))):
            t = (np.linalg.inv(P[:-1]) @ P[1:])[:, :3, 3]
            moving = np.linalg.norm(t, axis=1) > args.min_step
            dz = np.abs(t[moving, 2])
            rows.append(
                {
                    "sequence": seq,
                    "source": source,
                    "frames": int(moving.sum()),
                    "vertical_rms_mm": round(1e3 * float(np.sqrt(np.mean(dz**2))), 1),
                    "vertical_p99_mm": round(1e3 * float(np.percentile(dz, 99)), 1),
                    "vertical_max_mm": round(1e3 * float(dz.max()), 1),
                }
            )
            print("  ".join(f"{k}={v}" for k, v in rows[-1].items()))
    _write_csv(Path(args.out_dir) / "body_motion.csv", rows)


def cmd_plot(args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = load_config(args.config)
    track = Track(cfg, args.sequence, Path(args.kitti_raw).expanduser(), Path(args.loop_dir))
    dist = track.path() / 1000.0
    frames = np.arange(0, len(track.odom), args.frame_step)
    panels = []
    # Horizontal error magnitude and the predicted RMS horizontal error
    # sigma_h = sqrt(sigma_x^2 + sigma_y^2). Both are invariant to the heading,
    # unlike a lateral component, which swaps with along-track in every turn.
    # For a consistent covariance the error stays below 2 sigma_h about 98 %
    # of the time (exactly so for a circular 2-D Gaussian).
    for name, title in (
        ("odometry", "Odometry only"),
        ("stage3", "Stage 3 pose graph (loop closures)"),
    ):
        poses, opt = track.graph(name)
        cov = opt.relative_pose_covariances(0, [int(k) for k in frames])
        g0, p0 = np.linalg.inv(track.gt[0]), np.linalg.inv(poses[0])
        e = np.array([np.hypot(*_errors(g0 @ track.gt[k], p0 @ poses[k])[3:5]) for k in frames])
        s = np.sqrt(np.array([cov[int(k)][3, 3] + cov[int(k)][4, 4] for k in frames]))
        marks = None
        if name == "stage3":
            li, lj, _ = track.study.loops(track.model)
            marks = ("loop-closure frames", dist[np.unique(np.concatenate([li, lj]))])
        panels.append((f"{title}, relative to the first frame", e, s, marks))
    poses, opt, ctrl = control_graph(track, args.spacing, args.sigma)
    cov_w = opt.get_position_marginals([int(k) for k in frames])
    e = np.hypot(*(poses[frames, :2, 3] - track.gt[frames, :2, 3]).T)
    s = np.sqrt(np.array([cov_w[int(k)][0, 0] + cov_w[int(k)][1, 1] for k in frames]))
    title = f"Ground-truth positions as control points every {args.spacing:.0f} m"
    panels.append((title, e, s, ("control points", dist[ctrl])))

    # Reference data-viz palette: series slot 1, neutral band, text tokens.
    surface, ink, muted, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    line, band = "#2a78d6", "#d9d8d3"
    fig, axes = plt.subplots(len(panels), 1, figsize=(9, 7.5), sharex=True, constrained_layout=True)
    fig.patch.set_facecolor(surface)
    for ax, (title, e, s, marks) in zip(axes, panels):
        x = dist[frames]
        ax.set_facecolor(surface)
        ax.grid(axis="y", color=grid, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.fill_between(x, 0, 2 * s, color=band, linewidth=0, label="predicted 2σ_h")
        ax.plot(x, e, color=line, linewidth=1.4, label="actual horizontal error")
        top = 1.12 * max(e.max(), 2 * s.max())
        ax.set_ylim(-0.08 * top, top)
        if marks is not None:
            label, xs = marks
            ax.plot(
                xs,
                np.full(len(xs), -0.04 * top),
                "|",
                color=muted,
                markersize=6,
                markeredgewidth=1.0,
                label=label,
            )
        ax.set_title(title, loc="left", fontsize=10, color=ink)
        ax.set_ylabel("horizontal [m]", color=muted, fontsize=9)
        ax.tick_params(colors=muted, labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(muted)
        ax.legend(loc="upper left", frameon=False, fontsize=8, labelcolor=ink, ncol=3)
    axes[-1].set_xlabel(f"distance along the drive [km], sequence {args.sequence}", color=muted)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, facecolor=surface)
    print(f"wrote {args.out}")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name, fn, help_ in (
        ("segments", cmd_segments, "relative-pose covariance vs. error on devkit segments"),
        ("control", cmd_control, "control points every S metres: error vs. predicted sigma"),
        ("motion", cmd_motion, "vertical motion per frame in the body frame"),
        ("plot", cmd_plot, "lateral error and predicted band along one sequence"),
    ):
        p = sub.add_parser(name, help=help_)
        p.set_defaults(fn=fn)
        p.add_argument("--config", default="configs/default.yaml")
        p.add_argument("--kitti-raw", default="~/data/kitti_raw")
        p.add_argument(
            "--loop-dir", default="results/loop_closure", help="loop_closure_study output"
        )
        p.add_argument("--sigma", type=float, default=0.05, help="control point sigma [m]")
        p.add_argument("--frame-step", type=int, default=5, help="evaluate every n-th frame")
        if name == "plot":
            p.add_argument("--sequence", default="05")
            p.add_argument("--spacing", type=float, default=400.0)
            p.add_argument("--out", default="docs/figures/uncertainty_05.png")
        else:
            p.add_argument("--sequences", nargs="+", default=list(SEQUENCES))
            p.add_argument("--out-dir", default="benchmarks/uncertainty")
            p.add_argument("--min-segments", type=int, default=10)
            p.add_argument("--min-step", type=float, default=0.3, help="moving: step > this [m]")
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
