#!/usr/bin/env python3
"""SUP-01 Stage B host entry: verify a cached KITTI rosbag via the bag-builder container.

Thin wrapper: delegates to ``slam-baselines/bag-builder:latest`` running
``/scripts/verify_bag_impl.py``. Exits with the container's exit code.

Usage::

    python -m scripts.verify_bag_smoke --bag ~/data/kitti_bags_cache/kitti_00_fixed.bag --sequence 00

Prerequisites:
    make build-bagbuilder && make prepare-bags SEQUENCES=00
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bag", type=Path, required=True, help="Path to kitti_SS_fixed.bag on host")
    ap.add_argument("--sequence", required=True, help="KITTI sequence id (00-10)")
    ap.add_argument(
        "--image",
        default="slam-baselines/bag-builder:latest",
        help="Bag-builder container image (default: slam-baselines/bag-builder:latest)",
    )
    args = ap.parse_args()

    bag_host = args.bag.expanduser().resolve()
    if not bag_host.exists():
        print(f"ERROR: bag not found on host: {bag_host}", file=sys.stderr)
        return 2

    cache_dir = bag_host.parent
    bag_name = bag_host.name

    cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{cache_dir}:/cache:ro",
        "--entrypoint",
        "python3",
        args.image,
        "/scripts/verify_bag_impl.py",
        "--bag",
        f"/cache/{bag_name}",
        "--sequence",
        args.sequence,
    ]
    print("+ " + " ".join(cmd), file=sys.stderr)
    return subprocess.call(cmd, env=os.environ)


if __name__ == "__main__":
    sys.exit(main())
