#!/usr/bin/env python3
"""Diagnose IMU timing pathologies in a LIO-SAM run (SUP-01 R3).

Symptom that motivates this tool: every LIO-SAM Seq 00 attempt logs hundreds
of ``Large velocity, reset IMU-preintegration!`` warnings. That message fires
inside ``imuPreintegration.cpp`` when the body-frame velocity integrated since
the last keyframe exceeds 30 m/s — physically impossible for KITTI urban
driving (≤ 15 m/s). The candidate root causes split into three buckets:

1. Bag-side: non-monotonic IMU stamps, large dt gaps in /imu/data, IMU
   topic dropouts. ``fix_imu_timestamps.py`` covers monotonicity, but
   gaps and rate variability slip through.
2. Runtime-side: rosbag play queue overflows under -r 0.5 (bag time vs
   wall time ratio 2× → IMU 100Hz queues 200 msgs/s into LIO-SAM's
   ``imu_callback``).
3. IMU content: body-frame specific force misinterpreted as kinematic
   accel, double-applies gravity, ~+2g spurious accel on every step.

This tool extracts the bag-time of every reset event from the LIO-SAM stderr
log, then correlates with IMU stream stats from the bag itself: rate, dt
percentiles, gaps > 50 ms. The output report tells you whether resets cluster
around real IMU gaps (bag/runtime issue) or are evenly distributed (content/
math issue).

Usage::

    python3 -m scripts.diagnose_liosam_run \\
        --log /tmp/baseline_lio_sam_00.log \\
        --bag ~/data/kitti_bags_cache/kitti_00_fixed_navaccel.bag \\
        [--report-only]

The --bag argument is optional (omit if the bag isn't on the host); without it
only the log-side analysis runs. With --bag, the script docker-execs the bag
read inside slam-baselines/bag-builder (rosbag isn't installable on host).
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import textwrap
from collections import Counter
from pathlib import Path

import numpy as np

RESET_RE = re.compile(
    r"\[WARN\]\s*\[(?P<wall>[\d.]+),\s*(?P<bag>[\d.]+)\]:\s*"
    r"Large velocity, reset IMU-preintegration!"
)
BAGTIME_RE = re.compile(r"Bag Time:\s*([\d.]+)\s+Duration:\s*([\d.]+)\s*/\s*([\d.]+)")


def parse_log(log_path: Path) -> dict:
    """Extract reset-event bag times + bag duration from a LIO-SAM stderr log."""
    text = log_path.read_text(errors="replace")
    resets = [(float(m["wall"]), float(m["bag"])) for m in RESET_RE.finditer(text)]
    durations = [(float(b), float(d), float(t)) for b, d, t in BAGTIME_RE.findall(text)]
    bag_t0 = durations[0][0] if durations else None
    bag_t1 = durations[-1][0] if durations else None
    total_dur = durations[-1][2] if durations else None
    return {
        "n_resets": len(resets),
        "resets_wall_bag": resets,
        "bag_t0": bag_t0,
        "bag_t1": bag_t1,
        "bag_duration_s": total_dur,
    }


def _imu_stats_inside_container(bag_path: Path, imu_topic: str = "/imu/data") -> dict:
    """Read IMU stamps from bag inside slam-baselines/bag-builder container.

    Returns a JSON-serialisable dict with per-second rate samples and dt percentiles.
    """
    py_in_container = textwrap.dedent(f"""
        import json, sys
        import rosbag
        stamps_ns = []
        with rosbag.Bag('/cache/{bag_path.name}', 'r') as bag:
            for topic, msg, _t in bag.read_messages():
                if topic == '{imu_topic}':
                    stamps_ns.append(int(msg.header.stamp.to_nsec()))
        s = sorted(stamps_ns)
        if len(s) < 2:
            print(json.dumps({{'error': 'fewer than 2 IMU samples found'}}))
            sys.exit(0)
        import statistics
        dts_ns = [s[i+1] - s[i] for i in range(len(s)-1)]
        # gaps > 50 ms
        gap_ns = 50_000_000
        gaps = [(s[i], dts_ns[i]) for i in range(len(dts_ns)) if dts_ns[i] > gap_ns]
        # per-second bucket
        t0 = s[0]
        per_sec = {{}}
        for ns in s:
            sec = (ns - t0) // 1_000_000_000
            per_sec[sec] = per_sec.get(sec, 0) + 1
        rates = sorted(per_sec.values())
        out = {{
            'n_imu': len(s),
            'first_stamp_ns': s[0],
            'last_stamp_ns': s[-1],
            'duration_s': (s[-1] - s[0]) / 1e9,
            'dt_min_us': min(dts_ns) / 1e3,
            'dt_p50_us': statistics.median(dts_ns) / 1e3,
            'dt_p95_us': sorted(dts_ns)[int(len(dts_ns) * 0.95)] / 1e3,
            'dt_p99_us': sorted(dts_ns)[int(len(dts_ns) * 0.99)] / 1e3,
            'dt_max_us': max(dts_ns) / 1e3,
            'gaps_over_50ms': len(gaps),
            'gap_stamps_ns': [g[0] for g in gaps],
            'gap_examples': gaps[:10],
            'rate_min': min(rates),
            'rate_p50': rates[len(rates)//2],
            'rate_max': max(rates),
            'n_seconds': len(per_sec),
        }}
        print(json.dumps(out))
    """).strip()

    if shutil.which("docker") is None:
        return {"error": "docker not available on PATH"}
    cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{bag_path.parent}:/cache:ro",
        "--entrypoint",
        "python3",
        "slam-baselines/bag-builder:latest",
        "-c",
        py_in_container,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        return {"error": f"docker run failed: {r.stderr[:300]}"}
    import json

    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as e:
        return {"error": f"could not parse output: {e}; raw={r.stdout[:200]}"}


def correlate(log: dict, imu: dict) -> dict:
    """For each reset event, find the IMU gap (if any) within ±0.5s of the bag time."""
    if "error" in imu:
        return {"error": imu["error"]}
    if not log["resets_wall_bag"]:
        return {"hits": 0, "near_gap_ratio": 0.0}
    # Prefer the full gap_stamps_ns list. Fall back to gap_examples for bags
    # built before the serialization fix (still partial, emits a note).
    full_stamps = imu.get("gap_stamps_ns")
    if full_stamps is None:
        legacy = imu.get("gap_examples", [])
        full_stamps = [g[0] for g in legacy]
        partial_note = (
            "fell back to gap_examples (first 10) — rebuild IMU stats for full coverage"
            if legacy
            else None
        )
    else:
        partial_note = None

    if not full_stamps:
        return {
            "hits": 0,
            "near_gap_ratio": 0.0,
            "note": "no IMU gaps > 50ms in bag → resets are content/runtime, not bag",
        }
    gap_secs = [s / 1e9 for s in full_stamps]
    near = 0
    for _, bag_t in log["resets_wall_bag"]:
        for gs in gap_secs:
            if abs(gs - bag_t) < 0.5:
                near += 1
                break
    out = {
        "n_resets": len(log["resets_wall_bag"]),
        "n_resets_near_gap": near,
        "n_gaps_total": len(full_stamps),
        "near_gap_ratio": near / max(len(log["resets_wall_bag"]), 1),
    }
    if partial_note:
        out["note"] = partial_note
    return out


def reset_hist(resets: list[tuple[float, float]], bag_dur: float | None) -> str:
    """ASCII histogram of reset bag-times over 20 buckets."""
    if not resets or not bag_dur:
        return "(no resets or unknown duration)"
    bag_times = np.array([b for _, b in resets])
    rel = bag_times - bag_times.min()
    n_bins = 20
    bin_w = max(rel.max() / n_bins, 1e-9)
    hist = Counter(int(t / bin_w) for t in rel)
    max_count = max(hist.values()) if hist else 1
    lines = []
    for b in range(n_bins):
        c = hist.get(b, 0)
        bar = "#" * int(40 * c / max_count) if max_count else ""
        lines.append(f"  bin {b:2d}  count={c:4d}  {bar}")
    return "\n".join(lines)


def render(log_path: Path, log: dict, imu: dict, corr: dict) -> str:
    out = [f"=== LIO-SAM Run Diagnostic — {log_path.name} ==="]
    out.append("")
    out.append("[Log analysis]")
    out.append(f"  bag duration:        {log['bag_duration_s']!r} s")
    out.append(f"  total reset events:  {log['n_resets']}")
    if log["resets_wall_bag"]:
        out.append("  reset distribution over bag time (ASCII histogram):")
        out.append(reset_hist(log["resets_wall_bag"], log["bag_duration_s"]))
    out.append("")

    out.append("[IMU stream analysis (from bag)]")
    if "error" in imu:
        out.append(f"  ERROR: {imu['error']}")
    else:
        out.append(f"  n_imu_samples:       {imu['n_imu']}")
        out.append(f"  duration (bag):      {imu['duration_s']:.2f} s")
        out.append(
            f"  dt percentiles (us): min={imu['dt_min_us']:.1f}  p50={imu['dt_p50_us']:.1f}  "
            f"p95={imu['dt_p95_us']:.1f}  p99={imu['dt_p99_us']:.1f}  max={imu['dt_max_us']:.1f}"
        )
        out.append(f"  gaps > 50 ms:        {imu['gaps_over_50ms']}")
        if imu["gap_examples"]:
            out.append("    first 10 (stamp_ns, dt_ns):")
            for s, d in imu["gap_examples"][:10]:
                out.append(f"      ns={s}  dt={d / 1e6:.2f} ms")
        out.append(
            f"  per-second rate:     min={imu['rate_min']}  p50={imu['rate_p50']}  "
            f"max={imu['rate_max']}  ({imu['n_seconds']} seconds covered)"
        )
    out.append("")

    out.append("[Correlation]")
    if "error" in corr:
        out.append(f"  ERROR: {corr['error']}")
    elif "note" in corr:
        out.append(f"  {corr['note']}")
    else:
        out.append(
            f"  {corr['n_resets_near_gap']}/{corr['n_resets']} resets within "
            f"±0.5s of an IMU gap >50ms ({corr['near_gap_ratio']:.1%})"
        )
        if corr["near_gap_ratio"] < 0.1:
            out.append(
                "  → resets do NOT correlate with bag IMU gaps. "
                "Root cause is IMU content (body-frame double-gravity) or "
                "runtime queue overflow, not stamp pathology."
            )
        elif corr["near_gap_ratio"] > 0.5:
            out.append(
                "  → resets DO correlate with IMU gaps. "
                "Bag has dropouts; investigate kitti_to_rosbag.py sync logic "
                "or fix_imu_timestamps.py drift."
            )
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--log", type=Path, required=True, help="LIO-SAM stderr log")
    ap.add_argument("--bag", type=Path, help="(optional) bag path on host")
    ap.add_argument("--imu-topic", default="/imu/data")
    args = ap.parse_args()

    if not args.log.exists():
        print(f"ERROR: log not found: {args.log}", file=sys.stderr)
        return 1

    log = parse_log(args.log)
    imu = (
        _imu_stats_inside_container(args.bag, args.imu_topic)
        if args.bag and args.bag.exists()
        else {"error": "no bag provided or bag missing"}
    )
    corr = correlate(log, imu) if "error" not in imu else {"error": imu["error"]}
    print(render(args.log, log, imu, corr))
    return 0


if __name__ == "__main__":
    sys.exit(main())
