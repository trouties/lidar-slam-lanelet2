#!/usr/bin/env python3
"""SUP-02 Round 2 bonus: 2×3 APE matrix over (sup07_enabled, max_matches_per_query).

For each combination of ``sup07.enabled`` ∈ {false, true} and
``loop_closure.scan_context.max_matches_per_query`` ∈ {5, 10, 15}, reruns
the optimized stage of the pipeline with ``force_rebuild="optimized"`` and
collects APE metrics. Six full Stage 3 rebuilds; expect 6-9 hours on
KITTI seq 00 depending on K and the downstream Stage 4/5 rebuild cost
(which is triggered automatically because ``optimized`` is the upstream
dependency of ``fused``).

Artifacts in ``--output-dir`` (default ``results/``) are overwritten by
each cell; only the last cell's poses/map/osm remain on disk. The APE
matrix CSV is the durable output.

Usage:
    python scripts/bonus_ape_matrix.py --sequence 00
"""

from __future__ import annotations

import argparse
import copy
import csv
import time
from pathlib import Path

from scripts.run_pipeline import load_config, run_pipeline_cached
from src.cache import LayeredCache


def _run_cell(
    base_cfg: dict,
    sequence: str,
    sup07_enabled: bool,
    k: int,
    output_dir: Path,
) -> dict:
    """Run one (sup07, K) cell and extract APE metrics from the summary."""
    cfg = copy.deepcopy(base_cfg)
    cfg.setdefault("sup07", {})["enabled"] = sup07_enabled
    cfg.setdefault("loop_closure", {}).setdefault("scan_context", {})[
        "max_matches_per_query"
    ] = k

    cache_cfg = cfg.get("cache", {})
    cache = None
    if cache_cfg.get("enabled", False):
        cache = LayeredCache(
            root=cache_cfg.get("root", "cache/kitti"),
            sequence=sequence,
        )

    t0 = time.time()
    summary = run_pipeline_cached(
        config=cfg,
        sequence=sequence,
        cache=cache,
        force_rebuild="optimized",
        output_dir=output_dir,
        verbose=False,
    )
    elapsed = time.time() - t0

    opt_metrics = summary.get("metrics", {}).get("optimized", {}) or {}
    return {
        "sup07_enabled": "true" if sup07_enabled else "false",
        "K": k,
        "ape_rmse_m": round(float(opt_metrics.get("ape_rmse", float("nan"))), 4),
        "ape_mean_m": round(float(opt_metrics.get("ape_mean", float("nan"))), 4),
        "n_closures": int(summary.get("loop_closures", 0)),
        "wall_time_s": round(elapsed, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SUP-02 Round 2 bonus: SUP-07 × K APE matrix",
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--sequence", default="00")
    parser.add_argument(
        "--k-values",
        default="5,10,15",
        help="Comma-separated max_matches_per_query sweep (default: 5,10,15)",
    )
    parser.add_argument(
        "--sup07",
        default="false,true",
        help="Comma-separated sup07.enabled toggles (default: false,true)",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Where run_pipeline_cached writes per-stage artifacts (overwritten per cell)",
    )
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    seq = args.sequence

    k_values = [int(x.strip()) for x in args.k_values.split(",") if x.strip()]
    sup07_values = [
        x.strip().lower() == "true" for x in args.sup07.split(",") if x.strip()
    ]
    n_cells = len(sup07_values) * len(k_values)

    print(f"Running APE matrix on seq {seq}: sup07={sup07_values} × K={k_values}")
    print(f"  {n_cells} cells total")

    rows: list[dict] = []
    for sup07_enabled in sup07_values:
        for k in k_values:
            cell_idx = len(rows) + 1
            print(f"\n=== cell {cell_idx}/{n_cells}: sup07={sup07_enabled} K={k} ===")
            row = _run_cell(
                base_cfg=base_cfg,
                sequence=seq,
                sup07_enabled=sup07_enabled,
                k=k,
                output_dir=Path(args.output_dir),
            )
            rows.append(row)
            print(
                f"  APE RMSE={row['ape_rmse_m']} m  mean={row['ape_mean_m']} m  "
                f"closures={row['n_closures']}  wall={row['wall_time_s']}s"
            )

    out_path = Path("results") / f"sup02_round2_ape_matrix_seq{seq}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nAPE matrix written to {out_path}")

    print("\n| sup07 | K | APE RMSE (m) | APE mean (m) | closures | wall (s) |")
    print("|---|---|---|---|---|---|")
    for row in rows:
        print(
            f"| {row['sup07_enabled']} | {row['K']} | "
            f"{row['ape_rmse_m']} | {row['ape_mean_m']} | "
            f"{row['n_closures']} | {row['wall_time_s']} |"
        )


if __name__ == "__main__":
    main()
