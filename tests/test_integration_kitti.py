"""End-to-end smoke integration test on real KITTI Seq 00 (first 200 frames).

Marked `slow` — opt-in via ``pytest -m slow``. Skipped automatically when
the KITTI dataset is not present at the configured path.

Why this exists: the rest of the test suite uses synthetic fixtures and
does not catch breakage at the boundaries between stages (e.g., a refactor
that changes pose array dtype, or an export that drops a feature class).
This test is the cheapest possible regression net for cross-stage contracts.
The 200-frame cap keeps wall time bounded; full-sequence runs belong in
``scripts/benchmark_stage5.py``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from scripts.run_pipeline import run_pipeline_cached

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"
SMOKE_FRAMES = 200
SEQUENCE = "00"


def _kitti_seq_dir(cfg: dict) -> Path:
    """Resolve the KITTI sequence directory from a parsed config dict."""
    root = Path(os.path.expanduser(cfg["data"]["kitti_root"]))
    return root / "sequences" / SEQUENCE / "velodyne"


@pytest.fixture(scope="module")
def kitti_config():
    """Load default config and skip the whole module if KITTI is missing."""
    with DEFAULT_CONFIG_PATH.open() as f:
        cfg = yaml.safe_load(f)
    seq_dir = _kitti_seq_dir(cfg)
    if not seq_dir.exists():
        pytest.skip(f"KITTI Seq {SEQUENCE} not found at {seq_dir}; skipping smoke test")
    return cfg


@pytest.mark.slow
def test_kitti_seq00_smoke_pipeline(kitti_config, tmp_path):
    """Run the full pipeline on KITTI Seq 00 first 200 frames; assert basic invariants.

    This is a cross-stage contract test, NOT a quality test. The thresholds
    are set loose enough that the test passes for any sane implementation,
    but tight enough to catch a regression that breaks the inter-stage flow
    (e.g., wrong dtype, dropped frames, malformed export).
    """
    summary = run_pipeline_cached(
        config=kitti_config,
        sequence=SEQUENCE,
        cache=None,  # bypass cache to actually exercise every stage
        max_frames=SMOKE_FRAMES,
        output_dir=tmp_path,
        verbose=False,
    )

    # --- Stage 1 contract ---
    assert summary["sequence"] == SEQUENCE
    assert summary["frame_count"] == SMOKE_FRAMES
    assert summary["has_gt"] is True

    # --- Stage 2 contract ---
    odom = summary["metrics"]["odometry"]
    assert "ape_rmse" in odom and odom["ape_rmse"] >= 0.0
    # Loose ceiling: KISS-ICP on Seq 00 first 200 frames is ~2.3 m mean
    # APE under default config (verified in smoke profile). The ceiling at
    # 5.0 m catches catastrophic regressions while leaving headroom for
    # numerical jitter and minor parameter tweaks.
    assert odom["ape_rmse"] < 5.0, (
        f"Stage 2 odometry APE RMSE {odom['ape_rmse']:.3f} m exceeds smoke ceiling 5.0 m — "
        "likely a Stage 2 contract regression"
    )

    # --- Stage 3 contract ---
    opt = summary["metrics"]["optimized"]
    assert "ape_rmse" in opt
    # Optimized cannot be worse than odometry by more than a tiny epsilon
    # (200 frames is too short for loop closures to fire, so optimized
    # should equal odometry up to numerical noise).
    assert opt["ape_rmse"] <= odom["ape_rmse"] + 0.05, (
        f"Stage 3 optimization regressed: optimized={opt['ape_rmse']:.4f} m vs "
        f"odometry={odom['ape_rmse']:.4f} m (200-frame window — no closures expected)"
    )

    # --- Stage 4 / 5 contract ---
    assert summary["metrics"]["map"]["point_count"] > 100_000, "map suspiciously small"
    curbs = summary["metrics"]["curbs"]
    assert curbs["working_point_count"] > 100_000, "working map suspiciously small"
    assert curbs["curb_point_count"] > 0, "curb extraction returned no candidates"
    assert curbs["curb_cluster_count"] >= 0

    # --- Stage 6 contract ---
    export = summary["metrics"]["export"]
    for key in ("kept", "rescued", "dropped", "total_input", "total_length_m"):
        assert key in export, f"Stage 6 curb export missing field '{key}'"
    assert export["total_input"] == curbs["curb_cluster_count"]

    # The OSM file must have been written and be parseable as XML
    osm_path = tmp_path / f"map_{SEQUENCE}.osm"
    assert osm_path.exists(), "Stage 6 did not write the .osm file"
    import xml.etree.ElementTree as ET

    tree = ET.parse(osm_path)
    root = tree.getroot()
    assert root.tag == "osm", f"Lanelet2 export root tag is '{root.tag}', expected 'osm'"
    # OSM must contain at least one <node> (curbs typically yield ≥1 node on 200 frames)
    assert root.find("node") is not None, "Stage 6 OSM file has no <node> elements"
