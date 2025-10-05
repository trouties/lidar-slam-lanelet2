"""Main entry point for the LiDAR SLAM HD Map pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def load_config(config_path: str | Path) -> dict:
    """Load YAML configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="LiDAR SLAM HD Map Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"Loaded config from {args.config}")
    print(f"KITTI root: {config['data']['kitti_root']}")
    print("Pipeline stages not yet implemented. See src/ modules.")


if __name__ == "__main__":
    main()
