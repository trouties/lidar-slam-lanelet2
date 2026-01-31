"""Multi-sequence benchmark for Stage 5 parameter iteration.

Runs the pipeline on one or more KITTI Odometry sequences, honoring the
layered cache so that after an initial cold pass, subsequent runs that only
touch Stage 5 parameters finish in a few minutes.

Outputs go to ``results/stage5/run_<YYYYMMDD-HHMMSS>_<label>/``:

- ``benchmark.md``          — Markdown summary table (rows = metric,
                              columns = sequence + mean/median).
- ``per_seq/<seq>.yaml``    — Full metrics dict for each sequence.
- ``config_snapshot.yaml``  — Copy of the active config (audit trail).

Typical usage::

    python scripts/benchmark_stage5.py --label pre-trim --force-rebuild all
    # (edit feature_extraction.py and default.yaml)
    python scripts/benchmark_stage5.py --label post-trim --force-rebuild stage5
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import median

import yaml

# Allow importing from scripts/ and src/ when invoked as a plain file
# (``python scripts/benchmark_stage5.py``).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.run_pipeline import (  # noqa: E402
    _FORCE_REBUILD_CHOICES,
    load_config,
    run_pipeline_cached,
)
from src.cache import LayeredCache  # noqa: E402

# KITTI Odometry sequences 00-10 ship with ground truth.
_DEFAULT_SEQUENCES = [f"{i:02d}" for i in range(11)]


def _format_number(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if math.isnan(value):
            return "-"
        return f"{value:.3f}"
    return str(value)


def _collect_rows(all_summaries: list[dict]) -> list[tuple[str, list]]:
    """Return a list of (metric_label, [per_seq_value, ...]) rows."""
    rows: list[tuple[str, list]] = []

    def _col(path: str):
        vals = []
        for s in all_summaries:
            cur = s
            for key in path.split("."):
                if cur is None:
                    break
                cur = cur.get(key) if isinstance(cur, dict) else None
            vals.append(cur)
        return vals

    rows.append(("frame_count", _col("frame_count")))
    rows.append(("odom APE RMSE", _col("metrics.odometry.ape_rmse")))
    rows.append(("odom RPE RMSE", _col("metrics.odometry.rpe_rmse")))
    rows.append(("opt APE RMSE", _col("metrics.optimized.ape_rmse")))
    rows.append(("opt RPE RMSE", _col("metrics.optimized.rpe_rmse")))
    rows.append(("fused APE RMSE", _col("metrics.fused.ape_rmse")))
    rows.append(("fused RPE RMSE", _col("metrics.fused.rpe_rmse")))
    rows.append(("loop_closures", _col("loop_closures")))
    rows.append(("master points", _col("metrics.map_master.point_count")))
    rows.append(("stage5 points", _col("metrics.stage5.working_point_count")))
    rows.append(("road points", _col("metrics.stage5.road_point_count")))
    rows.append(("lane candidates", _col("metrics.stage5.lane_candidate_count")))
    rows.append(("curb candidates", _col("metrics.stage5.curb_point_count")))
    rows.append(("curb clusters", _col("metrics.stage5.curb_cluster_count")))
    rows.append(("cluster count", _col("metrics.stage5.count")))
    rows.append(("cluster p05", _col("metrics.stage5.p05")))
    rows.append(("cluster p50", _col("metrics.stage5.p50")))
    rows.append(("cluster p95", _col("metrics.stage5.p95")))
    rows.append(("cluster max", _col("metrics.stage5.max")))
    rows.append(("thin ways", _col("metrics.stage6.lane.line_thin")))
    rows.append(("thick ways", _col("metrics.stage6.lane.line_thick")))
    rows.append(("area ways", _col("metrics.stage6.lane.area")))
    rows.append(("lane dropped", _col("metrics.stage6.lane.dropped")))
    rows.append(("lane inputs", _col("metrics.stage6.lane.total_input")))
    rows.append(("curb ways", _col("metrics.stage6.curb.kept")))
    rows.append(("curb dropped", _col("metrics.stage6.curb.dropped")))
    rows.append(("curb inputs", _col("metrics.stage6.curb.total_input")))
    rows.append(("wall time (s)", _col("wall_time_s")))

    return rows


def _write_benchmark_md(
    path: Path,
    sequences: list[str],
    rows: list[tuple[str, list]],
    label: str,
) -> None:
    """Write a markdown table with metric rows and per-sequence columns."""
    header_cols = ["metric", *sequences, "mean", "median"]
    lines: list[str] = [
        f"# Stage 5 Benchmark — label=`{label}`",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "| " + " | ".join(header_cols) + " |",
        "|" + "|".join(["---"] * len(header_cols)) + "|",
    ]
    for metric, values in rows:
        numeric = [v for v in values if isinstance(v, (int, float)) and v is not None]
        mean_val = sum(numeric) / len(numeric) if numeric else None
        median_val = median(numeric) if numeric else None
        cells = [metric] + [_format_number(v) for v in values]
        cells.append(_format_number(mean_val))
        cells.append(_format_number(median_val))
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 5 multi-sequence benchmark")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
    )
    parser.add_argument(
        "--sequences",
        type=str,
        default=",".join(_DEFAULT_SEQUENCES),
        help="Comma-separated KITTI Odometry sequence ids",
    )
    parser.add_argument(
        "--force-rebuild",
        type=str,
        default="none",
        choices=_FORCE_REBUILD_CHOICES,
        help="Invalidate cache entries before running (propagates downstream)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Limit frames per sequence (sanity checking only)",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="unlabeled",
        help="Short tag appended to the archive directory name",
    )
    parser.add_argument(
        "--archive-root",
        type=str,
        default="results/stage5",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Bypass the cache (every run is cold)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    sequences = [s.strip() for s in args.sequences.split(",") if s.strip()]

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    archive_dir = Path(args.archive_root) / f"run_{stamp}_{args.label}"
    archive_dir.mkdir(parents=True, exist_ok=True)
    (archive_dir / "per_seq").mkdir(exist_ok=True)

    # Audit trail.
    (archive_dir / "config_snapshot.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (archive_dir / "args.yaml").write_text(
        yaml.safe_dump(
            {
                "sequences": sequences,
                "force_rebuild": args.force_rebuild,
                "max_frames": args.max_frames,
                "label": args.label,
                "no_cache": args.no_cache,
            },
            sort_keys=False,
        )
    )

    cache_cfg = config.get("cache", {})
    cache_enabled = cache_cfg.get("enabled", False) and not args.no_cache
    cache_root = cache_cfg.get("root", "cache/kitti")

    all_summaries: list[dict] = []

    for seq in sequences:
        print(f"\n{'=' * 60}")
        print(f" Sequence {seq} (label={args.label})")
        print("=" * 60)
        cache = LayeredCache(cache_root, seq) if cache_enabled else None
        t0 = time.monotonic()
        try:
            summary = run_pipeline_cached(
                config=copy.deepcopy(config),
                sequence=seq,
                cache=cache,
                force_rebuild=args.force_rebuild,
                max_frames=args.max_frames,
                output_dir=Path(config.get("output", {}).get("dir", "results")),
                verbose=True,
            )
        except Exception as e:  # noqa: BLE001 — we want to continue with the next seq
            print(f"  !! Sequence {seq} failed: {e}")
            summary = {
                "sequence": seq,
                "error": str(e),
                "metrics": {},
                "cache_hits": {},
            }
        summary["wall_time_s"] = round(time.monotonic() - t0, 2)
        all_summaries.append(summary)

        (archive_dir / "per_seq" / f"{seq}.yaml").write_text(
            yaml.safe_dump(summary, sort_keys=False)
        )

    rows = _collect_rows(all_summaries)
    md_path = archive_dir / "benchmark.md"
    _write_benchmark_md(md_path, sequences, rows, args.label)

    print(f"\nBenchmark written to {md_path}")


if __name__ == "__main__":
    main()
