"""3D confidence ellipsoid visualization for pose graph marginal covariances.

Used by SUP-06 to render each keyframe's position uncertainty as an ellipsoid
along the optimized trajectory, and to animate how those ellipsoids balloon
and deflate across a GNSS-denial window.

Sample frames fall into three buckets:
    - prior-anchored (``is_prior``) : GNSS prior pins marginal to ``prior_sigma^2``
    - drift          (non-prior, non-denial, non-tail) : dead-reckoning steady state
    - tail           (last N frames) : pose graph edge effect (no downstream prior)

The trace time series is drawn as two separate series so readers don't
mistake the prior/drift 2-3 order-of-magnitude gap for noise.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3d projection

# ----------------------------------------------------------------------
# Pure geometry
# ----------------------------------------------------------------------


def cov_to_ellipsoid_mesh(
    center: np.ndarray,
    cov: np.ndarray,
    n_std: float = 2.0,
    n_u: int = 16,
    n_v: int = 8,
    display_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a 3x3 covariance into an ellipsoid mesh for ``plot_surface``.

    ``display_scale`` is a visual exaggeration multiplier — true 2-sigma
    ellipsoids on a 500 m trajectory are sub-meter and invisible. KITTI Seq 00
    needs ~50x to make the denial-window bulge visually obvious.
    """
    C = 0.5 * (np.asarray(cov) + np.asarray(cov).T)
    eigvals, eigvecs = np.linalg.eigh(C)
    eigvals = np.clip(eigvals, a_min=1e-12, a_max=None)
    radii = n_std * np.sqrt(eigvals) * display_scale

    u = np.linspace(0.0, 2.0 * np.pi, n_u)
    v = np.linspace(0.0, np.pi, n_v)
    su, cu = np.sin(u), np.cos(u)
    sv, cv = np.sin(v), np.cos(v)

    x = radii[0] * np.outer(sv, cu)
    y = radii[1] * np.outer(sv, su)
    z = radii[2] * np.outer(cv, np.ones_like(u))

    pts = np.stack([x, y, z], axis=-1)
    pts_world = pts @ eigvecs.T + np.asarray(center)[None, None, :]
    return pts_world[..., 0], pts_world[..., 1], pts_world[..., 2]


def _set_equal_aspect_3d(ax, trajectory: np.ndarray) -> None:
    extents = np.array([np.ptp(trajectory[:, i]) for i in range(3)])
    max_ext = max(float(extents.max()), 1.0)
    mid = trajectory.mean(axis=0)
    ax.set_xlim(mid[0] - max_ext / 2, mid[0] + max_ext / 2)
    ax.set_ylim(mid[1] - max_ext / 2, mid[1] + max_ext / 2)
    ax.set_zlim(mid[2] - max_ext / 2, mid[2] + max_ext / 2)


# ----------------------------------------------------------------------
# Helpers for sample-flag splitting
# ----------------------------------------------------------------------


def _split_samples(
    sample_frames: list[int],
    covariances: dict[int, np.ndarray],
    sample_flags: dict[int, tuple[bool, bool, bool]],
) -> dict[str, tuple[list[int], list[float]]]:
    """Group samples into (prior, denial, tail, drift) buckets with traces."""
    out = {
        "prior": ([], []),
        "denial": ([], []),
        "tail": ([], []),
        "drift": ([], []),
    }
    for k in sample_frames:
        is_prior, in_denial, is_tail = sample_flags[k]
        t = float(np.trace(covariances[k]))
        if in_denial:
            bucket = "denial"
        elif is_tail:
            bucket = "tail"
        elif is_prior:
            bucket = "prior"
        else:
            bucket = "drift"
        out[bucket][0].append(k)
        out[bucket][1].append(t)
    return out


def _drift_baseline(buckets: dict[str, tuple[list[int], list[float]]]) -> float:
    """Median of non-prior / non-denial / non-tail drift samples."""
    drift_traces = buckets["drift"][1]
    return float(np.median(drift_traces)) if drift_traces else float("nan")


# ----------------------------------------------------------------------
# Static 3D plot
# ----------------------------------------------------------------------


def plot_trajectory_with_ellipsoids(
    trajectory: np.ndarray,
    sample_frames: list[int],
    covariances: dict[int, np.ndarray],
    sample_flags: dict[int, tuple[bool, bool, bool]] | None = None,
    denial_window: tuple[int, int] | None = None,
    tail_start: int | None = None,
    n_std: float = 2.0,
    display_scale: float = 30.0,
    ellipsoid_stride: int = 5,
    metrics: dict | None = None,
    output_path: Path | None = None,
    title: str = "Keyframe Position Uncertainty",
) -> plt.Figure:
    """Static 3D plot: trajectory + ellipsoids at sampled frames.

    When ``sample_flags`` is provided, drift ellipsoids are drawn on the
    main 3D axes but prior-anchor ellipsoids are hidden (they would be
    invisible dots anyway — ``3e-4 m^2`` trace is sub-voxel).

    The trajectory is colour-segmented:
        - grey (pre-denial, non-tail)
        - crimson (GNSS denied)
        - grey (post-denial, non-tail)
        - gold  (tail region, if ``tail_start`` given)
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    n = len(trajectory)
    if denial_window is not None:
        ds, de = denial_window
    else:
        ds, de = None, None

    # Trajectory backbone first (thin grey)
    ax.plot(
        trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
        color="0.55", lw=1.0, label="trajectory",
    )

    # Pick which samples to draw as ellipsoids — skip priors and subsample
    if sample_flags is None:
        non_prior_frames = list(sample_frames)
    else:
        non_prior_frames = [k for k in sample_frames if not sample_flags[k][0]]
    draw_frames = non_prior_frames[::max(1, ellipsoid_stride)]

    # FORCE include peak frames so bulges are anchored regardless of stride
    forced = set()
    peak_in_denial_frame: int | None = None
    peak_in_tail_frame: int | None = None
    if sample_flags is not None and ds is not None:
        in_denial_frames = [k for k in sample_frames if sample_flags[k][1]]
        if in_denial_frames:
            traces_in = [float(np.trace(covariances[k])) for k in in_denial_frames]
            peak_in_denial_frame = in_denial_frames[int(np.argmax(traces_in))]
            forced.add(peak_in_denial_frame)
        tail_frames = [k for k in sample_frames if sample_flags[k][2]]
        if tail_frames:
            traces_t = [float(np.trace(covariances[k])) for k in tail_frames]
            peak_in_tail_frame = tail_frames[int(np.argmax(traces_t))]
            forced.add(peak_in_tail_frame)
            # Also include all tail samples so end-region ellipsoids are visible
            forced.update(tail_frames)
    if forced:
        draw_frames = sorted(set(draw_frames) | forced)

    if draw_frames:
        traces = np.array([float(np.trace(covariances[k])) for k in draw_frames])
        log_traces = np.log10(np.maximum(traces, 1e-12))
        vmin, vmax = float(log_traces.min()), float(log_traces.max())
        if vmax - vmin < 1e-6:
            vmax = vmin + 1.0
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap("viridis")

        for k, lt in zip(draw_frames, log_traces):
            X, Y, Z = cov_to_ellipsoid_mesh(
                trajectory[k], covariances[k], n_std=n_std, display_scale=display_scale,
            )
            color = cmap(norm(lt))
            ax.plot_surface(
                X, Y, Z,
                color=color, alpha=0.55, linewidth=0, antialiased=True, shade=True,
            )

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.1)
        cbar.set_label(
            r"$\log_{10}\,\mathrm{tr}(\Sigma_\mathrm{pos})$ [m$^2$]"
            f"\n(2σ ellipsoids drawn at {display_scale:g}× scale)"
        )

    # Overlay the denial / tail trajectory segments AFTER ellipsoids
    # using thicker lines + scatter markers so they survive the 3D zorder
    # quirks and remain visible through the surfaces.
    if ds is not None:
        seg = trajectory[ds : de + 1]
        ax.plot(
            seg[:, 0], seg[:, 1], seg[:, 2],
            color="crimson", lw=4.0, alpha=0.95, label="GNSS denied",
        )
        ax.scatter(
            seg[::5, 0], seg[::5, 1], seg[::5, 2],
            color="crimson", s=12, alpha=1.0, zorder=10, depthshade=False,
        )
    if tail_start is not None and tail_start < n:
        seg_t = trajectory[tail_start:]
        ax.plot(
            seg_t[:, 0], seg_t[:, 1], seg_t[:, 2],
            color="goldenrod", lw=3.5, alpha=0.95, label="end tail",
        )
        ax.scatter(
            seg_t[::3, 0], seg_t[::3, 1], seg_t[::3, 2],
            color="goldenrod", s=12, alpha=1.0, zorder=10, depthshade=False,
        )

    # Annotate peak ellipsoids in 2D axes coordinates so they ALWAYS render
    # on top — matplotlib 3D ax.text() honors zorder unreliably and gets
    # occluded by surfaces. ax.text2D draws after the 3D projection.
    label_lines: list[str] = []
    if peak_in_denial_frame is not None:
        peak_trace = float(np.trace(covariances[peak_in_denial_frame]))
        label_lines.append(
            f"peak (denial): frame {peak_in_denial_frame}, {peak_trace:.2f} m²"
        )
    if peak_in_tail_frame is not None and peak_in_tail_frame != peak_in_denial_frame:
        tail_trace = float(np.trace(covariances[peak_in_tail_frame]))
        label_lines.append(
            f"tail edge:     frame {peak_in_tail_frame}, {tail_trace:.2f} m²"
        )
    if label_lines:
        ax.text2D(
            0.02, 0.92, "\n".join(label_lines),
            transform=ax.transAxes,
            fontsize=9, weight="bold", color="black",
            verticalalignment="top",
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="lemonchiffon",
                edgecolor="crimson",
                linewidth=1.5,
                alpha=0.95,
            ),
            zorder=100,
        )

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")

    full_title = title
    if metrics is not None and metrics.get("denial_ratio") is not None:
        full_title = (
            f"{title}\n"
            f"peak/baseline = {metrics['denial_ratio']:.2f}×   "
            f"post/baseline = {metrics['recovery_ratio']:.2f}×   "
            f"baseline = {metrics['pre_trace']:.2e} m²"
        )
    ax.set_title(full_title, fontsize=10)
    ax.legend(loc="upper right", fontsize=8)

    _set_equal_aspect_3d(ax, trajectory)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


# ----------------------------------------------------------------------
# Animation
# ----------------------------------------------------------------------


def animate_uncertainty_evolution(
    trajectory: np.ndarray,
    sample_frames: list[int],
    covariances: dict[int, np.ndarray],
    denial_window: tuple[int, int],
    pre_denial_trace: float,
    output_path: Path,
    sample_flags: dict[int, tuple[bool, bool, bool]] | None = None,
    tail_start: int | None = None,
    fps: int = 10,
    n_std: float = 2.0,
    display_scale: float = 30.0,
    frame_skip: int = 2,
    metrics: dict | None = None,
    baseline_window_label: str | None = None,
    figsize: tuple[float, float] = (10.0, 4.5),
    dpi: int = 75,
    title: str = "SUP-06 Uncertainty Evolution",
) -> Path:
    """Render a GIF animation showing ellipsoid growth/shrinkage across denial.

    Left panel: 3D trajectory (colour-segmented) + current drift ellipsoid.
    Right panel: trace(cov_pos) vs frame, with **two series** — prior-anchor
        samples (green dots, ~prior_sigma^2) and drift samples (steelblue
        line, dead-reckoning steady state). Denial window shaded crimson,
        tail region shaded gold, reference lines at baseline / 1.5x / 2x.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(trajectory)
    ds, de = denial_window

    fig = plt.figure(figsize=figsize, dpi=dpi)
    # GridSpec with explicit wspace keeps 3D tick labels from colliding with
    # the right-hand 2D panel — matplotlib's default 1x2 subplot spacing is
    # too tight once the 3D axes claim their projection margin.
    gs = fig.add_gridspec(
        1, 2, wspace=0.32, left=0.04, right=0.97, top=0.88, bottom=0.12,
    )
    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    ax2d = fig.add_subplot(gs[0, 1])

    # --- Left panel: full grey backbone, then thick denial/tail overlays ---
    if tail_start is None:
        tail_start = n
    ax3d.plot(
        trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
        color="0.55", lw=1.0,
    )
    if de >= ds:
        seg_d = trajectory[ds : de + 1]
        ax3d.plot(
            seg_d[:, 0], seg_d[:, 1], seg_d[:, 2],
            color="crimson", lw=4.0, alpha=0.95, label="GNSS denied",
        )
        ax3d.scatter(
            seg_d[::5, 0], seg_d[::5, 1], seg_d[::5, 2],
            color="crimson", s=10, alpha=1.0, zorder=10, depthshade=False,
        )
    if tail_start < n:
        seg_t = trajectory[tail_start:]
        ax3d.plot(
            seg_t[:, 0], seg_t[:, 1], seg_t[:, 2],
            color="goldenrod", lw=3.5, alpha=0.95, label="end tail",
        )
        ax3d.scatter(
            seg_t[::3, 0], seg_t[::3, 1], seg_t[::3, 2],
            color="goldenrod", s=10, alpha=1.0, zorder=10, depthshade=False,
        )
    _set_equal_aspect_3d(ax3d, trajectory)
    ax3d.set_xlabel("x [m]")
    ax3d.set_ylabel("y [m]")
    ax3d.set_zlabel("z [m]")
    ax3d.legend(loc="upper right", fontsize=8)

    # --- Right panel: split trace series ---
    if sample_flags is not None:
        buckets = _split_samples(sample_frames, covariances, sample_flags)
    else:
        all_traces = [float(np.trace(covariances[k])) for k in sample_frames]
        buckets = {
            "prior": ([], []),
            "denial": (sample_frames, all_traces),  # collapse everything as "drift" when no flags
            "tail": ([], []),
            "drift": (sample_frames, all_traces),
        }

    # Drift + denial + tail plotted as the main curve (all non-prior samples)
    non_prior_frames: list[int] = []
    non_prior_traces: list[float] = []
    for bkey in ("drift", "denial", "tail"):
        fs, ts = buckets[bkey]
        non_prior_frames.extend(fs)
        non_prior_traces.extend(ts)
    order = np.argsort(non_prior_frames)
    non_prior_frames_np = np.array(non_prior_frames)[order]
    non_prior_traces_np = np.array(non_prior_traces)[order]

    ax2d.semilogy(
        non_prior_frames_np, non_prior_traces_np,
        color="steelblue", lw=1.5, label="non-prior drift",
    )
    # Prior anchors as small green dots
    prior_frames, prior_traces = buckets["prior"]
    if prior_frames:
        ax2d.scatter(
            prior_frames, prior_traces,
            s=14, color="forestgreen", marker="o", alpha=0.7,
            label="GNSS prior anchor",
        )

    # Shading
    ax2d.axvspan(ds, de, color="crimson", alpha=0.15, label="GNSS denied")
    if tail_start < n:
        ax2d.axvspan(tail_start, n, color="goldenrod", alpha=0.12, label="end tail")

    # Reference lines (on the drift baseline) — keep legend label terse so
    # it doesn't overflow the right edge; the full baseline-window
    # definition goes in a small footer annotation below.
    if pre_denial_trace and np.isfinite(pre_denial_trace) and pre_denial_trace > 0:
        ax2d.axhline(
            pre_denial_trace, color="gray", ls="--", lw=1.0,
            label=f"baseline = {pre_denial_trace:.2e} m²",
        )
        ax2d.axhline(2.0 * pre_denial_trace, color="orange", ls="--", lw=1.0,
                     label="2× baseline")
        ax2d.axhline(1.5 * pre_denial_trace, color="green", ls="--", lw=1.0,
                     label="1.5× baseline")

    ax2d.set_xlabel("frame index")
    ax2d.set_ylabel(r"$\mathrm{tr}(\Sigma_\mathrm{pos})$ [m$^2$]")
    ax2d.legend(loc="upper left", fontsize=7, ncol=2)
    ax2d.grid(True, which="both", alpha=0.3)

    if baseline_window_label:
        ax2d.text(
            0.99, 0.02, f"baseline = {baseline_window_label}",
            transform=ax2d.transAxes,
            fontsize=6, color="0.35",
            ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="0.7", linewidth=0.6, alpha=0.85),
        )

    # Dynamic artists
    current_surface: list = []
    (current_marker,) = ax2d.plot([], [], "ro", markersize=9, zorder=10)

    # Build an ordered list of frames we actually animate (skip prior-anchor —
    # sub-voxel ellipsoid invisible; keep drift, denial, tail). Frame-skip
    # halves the GIF size without losing the bulge envelope.
    if sample_flags is not None:
        anim_frames = [k for k in sample_frames if not sample_flags[k][0]]
    else:
        anim_frames = list(sample_frames)
    if frame_skip > 1:
        # Always keep first/last and the in-denial peak so the bulge is anchored
        keep_idx = set(range(0, len(anim_frames), frame_skip))
        keep_idx.add(0)
        keep_idx.add(len(anim_frames) - 1)
        if sample_flags is not None:
            in_denial_in_anim = [
                i for i, k in enumerate(anim_frames) if sample_flags[k][1]
            ]
            if in_denial_in_anim:
                trs = [float(np.trace(covariances[anim_frames[i]])) for i in in_denial_in_anim]
                peak_local = in_denial_in_anim[int(np.argmax(trs))]
                keep_idx.add(peak_local)
        anim_frames = [anim_frames[i] for i in sorted(keep_idx)]
    anim_traces = {k: float(np.trace(covariances[k])) for k in anim_frames}

    # Compose suptitle with metrics (peak/baseline, post/baseline) when available
    suptitle = title
    if metrics is not None and metrics.get("denial_ratio") is not None:
        suptitle = (
            f"{title}    "
            f"peak/baseline={metrics['denial_ratio']:.2f}×   "
            f"post/baseline={metrics['recovery_ratio']:.2f}×   "
            f"(2σ ellipsoids @ {display_scale:g}× scale)"
        )
    fig.suptitle(suptitle, fontsize=11)

    def update(frame_idx: int):
        k = anim_frames[frame_idx]
        while current_surface:
            current_surface.pop().remove()
        X, Y, Z = cov_to_ellipsoid_mesh(
            trajectory[k], covariances[k], n_std=n_std, display_scale=display_scale,
        )
        surf = ax3d.plot_surface(
            X, Y, Z, color="darkorange", alpha=0.7, linewidth=0, antialiased=True,
        )
        current_surface.append(surf)
        current_marker.set_data([k], [anim_traces[k]])
        return [surf, current_marker]

    anim = FuncAnimation(
        fig, update, frames=len(anim_frames), interval=1000.0 / fps, blit=False
    )

    writer = PillowWriter(fps=fps)
    anim.save(str(output_path), writer=writer)
    plt.close(fig)
    return output_path
