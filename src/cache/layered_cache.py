"""Layered cache for Stage 1-5 artifacts.

Goal: avoid re-running Stage 1-4 every time Stage 5 parameters change. Each
pipeline stage has its own cache slot under ``cache/kitti/<seq>/`` with:

- binary artifact (npz / pcd / geojson)
- an entry in a shared ``metadata.yaml`` recording ``config_hash``,
  ``upstream_hash``, parameter snapshot, metrics, and timestamp.

Invalidation: a stage cache loads successfully only if (a) its own
``config_hash`` matches the current config subtree relevant to that stage,
and (b) its ``upstream_hash`` matches the current hash of the immediate
upstream stage. Walking the chain transitively would be redundant because
every stage stores its upstream, so fresh-computing the immediate hash is
sufficient.

Stages form a linear dependency chain::

    odometry -> optimized -> fused -> map_master -> stage5

``map_master`` is the voxel=0.05m global map produced after Stage 4 fusion.
It sits between Stage 4 and Stage 5 so that Stage 5 can re-downsample to its
working voxel (default 0.15m) without re-running Stage 1-4 or the full
per-frame aggregation.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
import yaml

# Linear dependency chain. Each entry's immediate predecessor (if any) is the
# previous element of this list. ``invalidate(stage)`` clears this stage and
# everything after it.
STAGE_ORDER = ["odometry", "optimized", "fused", "map_master", "stage5"]

# Artifact file names relative to the per-sequence cache directory.
#
# Point clouds are stored as ``.npz`` rather than ``.pcd`` because Open3D's
# PCD writer quantizes the ``colors`` channel to uint8, which would corrupt
# the float reflectance we tunnel through it (intensity aliases on the
# ``0.40`` lane-marking threshold would otherwise flip cluster membership).
_STAGE_FILES: dict[str, list[str]] = {
    "odometry": ["odometry.npz"],
    "optimized": ["optimized.npz"],
    "fused": ["fused.npz"],
    "map_master": ["global_map_master.npz"],
    "stage5": ["stage5.npz", "stage5_features.geojson"],
}


def _config_subtree(stage: str, config: dict) -> dict:
    """Extract the config subtree relevant to a given stage.

    The subtree is what gets hashed to decide whether the cached artifact for
    this stage is still valid. The result excludes ``data.sequence`` because
    the cache is already partitioned per-sequence on disk, so including it
    would just make the hash per-sequence without adding discriminating power.
    """
    data_clean = {k: v for k, v in config.get("data", {}).items() if k != "sequence"}
    if stage == "odometry":
        return {"data": data_clean, "kiss_icp": config.get("kiss_icp", {})}
    if stage == "optimized":
        return {
            "gtsam": config.get("gtsam", {}),
            "loop_closure": config.get("loop_closure", {}),
        }
    if stage == "fused":
        return {"ekf": config.get("ekf", {})}
    if stage == "map_master":
        m = config.get("mapping", {})
        return {
            "master_voxel_size": m.get("master_voxel_size"),
            "max_range": m.get("max_range"),
            "downsample_every": m.get("downsample_every"),
        }
    if stage == "stage5":
        # Entire mapping subtree — changing any Stage 5 param invalidates the
        # cluster list, but not the master map.
        return {"mapping": config.get("mapping", {})}
    raise ValueError(f"Unknown stage: {stage!r}")


def compute_hash(subtree: dict) -> str:
    """Return a 16-char SHA256 hex digest of a canonical YAML dump."""
    canonical = yaml.safe_dump(subtree, sort_keys=True, default_flow_style=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _upstream(stage: str) -> str | None:
    idx = STAGE_ORDER.index(stage)
    return STAGE_ORDER[idx - 1] if idx > 0 else None


class LayeredCache:
    """Per-sequence layered cache for the SLAM pipeline."""

    def __init__(self, root: str | Path, sequence: str) -> None:
        self.root = Path(root) / sequence
        self.sequence = sequence
        self.meta_path = self.root / "metadata.yaml"

    # ------------------------------------------------------------------
    # Metadata I/O
    # ------------------------------------------------------------------

    def _load_metadata(self) -> dict:
        if not self.meta_path.exists():
            return {}
        with open(self.meta_path) as f:
            return yaml.safe_load(f) or {}

    def _save_metadata(self, meta: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with open(self.meta_path, "w") as f:
            yaml.safe_dump(meta, f, sort_keys=False)

    def metadata_snapshot(self) -> dict:
        """Public read-only copy of the metadata.yaml contents."""
        return self._load_metadata()

    # ------------------------------------------------------------------
    # Hash helpers (shared by load and save)
    # ------------------------------------------------------------------

    @staticmethod
    def hash_for(stage: str, config: dict) -> str:
        return compute_hash(_config_subtree(stage, config))

    def _upstream_hash(self, stage: str, config: dict) -> str | None:
        upstream = _upstream(stage)
        if upstream is None:
            return None
        return self.hash_for(upstream, config)

    def _is_fresh(self, stage: str, config: dict) -> bool:
        meta = self._load_metadata()
        entry = meta.get(stage)
        if entry is None:
            return False
        if entry.get("config_hash") != self.hash_for(stage, config):
            return False
        if entry.get("upstream_hash") != self._upstream_hash(stage, config):
            return False
        for fname in _STAGE_FILES[stage]:
            if not (self.root / fname).exists():
                return False
        return True

    # ------------------------------------------------------------------
    # Invalidation
    # ------------------------------------------------------------------

    def invalidate(self, stage: str) -> None:
        """Drop this stage's cache and all downstream stages.

        ``stage`` may be one of ``STAGE_ORDER``, or the aliases ``"all"``
        (= ``odometry``) / ``"none"`` (no-op).
        """
        if stage == "none":
            return
        if stage == "all":
            stage = STAGE_ORDER[0]
        if stage not in STAGE_ORDER:
            raise ValueError(f"Unknown stage: {stage!r}")

        idx = STAGE_ORDER.index(stage)
        meta = self._load_metadata()
        for s in STAGE_ORDER[idx:]:
            meta.pop(s, None)
            for fname in _STAGE_FILES[s]:
                p = self.root / fname
                if p.exists():
                    p.unlink()
        self._save_metadata(meta)

    # ------------------------------------------------------------------
    # Per-stage load/save
    # ------------------------------------------------------------------

    def load_odometry(self, config: dict) -> tuple[np.ndarray, np.ndarray] | None:
        if not self._is_fresh("odometry", config):
            return None
        data = np.load(self.root / "odometry.npz")
        return data["poses"], data["timestamps"]

    def save_odometry(
        self,
        poses: np.ndarray,
        timestamps: np.ndarray,
        config: dict,
        metrics: dict | None = None,
    ) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            self.root / "odometry.npz",
            poses=np.asarray(poses),
            timestamps=np.asarray(timestamps),
        )
        self._write_metadata_entry("odometry", config, metrics)

    def load_optimized(self, config: dict) -> np.ndarray | None:
        if not self._is_fresh("optimized", config):
            return None
        data = np.load(self.root / "optimized.npz")
        return data["poses"]

    def save_optimized(self, poses: np.ndarray, config: dict, metrics: dict | None = None) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.root / "optimized.npz", poses=np.asarray(poses))
        self._write_metadata_entry("optimized", config, metrics)

    def load_fused(self, config: dict) -> np.ndarray | None:
        if not self._is_fresh("fused", config):
            return None
        data = np.load(self.root / "fused.npz")
        return data["poses"]

    def save_fused(self, poses: np.ndarray, config: dict, metrics: dict | None = None) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.root / "fused.npz", poses=np.asarray(poses))
        self._write_metadata_entry("fused", config, metrics)

    def load_global_map_master(self, config: dict) -> o3d.geometry.PointCloud | None:
        if not self._is_fresh("map_master", config):
            return None
        return self._load_pcd_npz(self.root / "global_map_master.npz")

    def save_global_map_master(
        self,
        pcd: o3d.geometry.PointCloud,
        config: dict,
        metrics: dict | None = None,
    ) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self._save_pcd_npz(pcd, self.root / "global_map_master.npz")
        self._write_metadata_entry("map_master", config, metrics)

    def load_stage5(self, config: dict) -> tuple[o3d.geometry.PointCloud, list[np.ndarray]] | None:
        if not self._is_fresh("stage5", config):
            return None
        pcd = self._load_pcd_npz(self.root / "stage5.npz")
        clusters = self._load_clusters_geojson(self.root / "stage5_features.geojson")
        return pcd, clusters

    def save_stage5(
        self,
        pcd: o3d.geometry.PointCloud,
        clusters: list[np.ndarray],
        config: dict,
        metrics: dict | None = None,
    ) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self._save_pcd_npz(pcd, self.root / "stage5.npz")
        self._save_clusters_geojson(clusters, self.root / "stage5_features.geojson")
        self._write_metadata_entry("stage5", config, metrics)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _write_metadata_entry(self, stage: str, config: dict, metrics: dict | None) -> None:
        meta = self._load_metadata()
        subtree = _config_subtree(stage, config)
        meta[stage] = {
            "config_hash": compute_hash(subtree),
            "upstream_hash": self._upstream_hash(stage, config),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "params": subtree,
            "metrics": metrics or {},
        }
        self._save_metadata(meta)

    @staticmethod
    def _save_pcd_npz(pcd: o3d.geometry.PointCloud, path: Path) -> None:
        """Write points + reflectance to .npz preserving full float precision.

        Reflectance lives in the Open3D ``colors`` channel as ``[I, I, I]``
        (see ``src/mapping/map_builder.py``); we pull just the first column.
        """
        pts = np.asarray(pcd.points, dtype=np.float32)
        colors = np.asarray(pcd.colors)
        intensity = (
            colors[:, 0].astype(np.float32)
            if colors.size > 0
            else np.zeros(pts.shape[0], dtype=np.float32)
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, points=pts, intensity=intensity)

    @staticmethod
    def _load_pcd_npz(path: Path) -> o3d.geometry.PointCloud:
        """Rebuild an Open3D point cloud from an npz written by ``_save_pcd_npz``."""
        data = np.load(path)
        pts = data["points"].astype(np.float64)
        intensity = data["intensity"].astype(np.float64)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        if pts.shape[0] > 0:
            pcd.colors = o3d.utility.Vector3dVector(np.repeat(intensity[:, None], 3, axis=1))
        return pcd

    @staticmethod
    def _save_clusters_geojson(clusters: list[np.ndarray], path: Path) -> None:
        """Minimal GeoJSON dump of cluster points (same schema as Stage 5 output)."""
        features: list[dict[str, Any]] = []
        for idx, cluster in enumerate(clusters):
            coords = [[float(x), float(y), float(z)] for x, y, z in cluster]
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "MultiPoint", "coordinates": coords},
                    "properties": {
                        "id": idx,
                        "type": "lane_marking",
                        "point_count": int(cluster.shape[0]),
                    },
                }
            )
        collection = {
            "type": "FeatureCollection",
            "crs": {
                "type": "name",
                "properties": {"name": "local:velodyne_meters"},
            },
            "features": features,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(collection, f)

    @staticmethod
    def _load_clusters_geojson(path: Path) -> list[np.ndarray]:
        with open(path) as f:
            data = json.load(f)
        clusters: list[np.ndarray] = []
        for feature in data.get("features", []):
            coords = feature["geometry"]["coordinates"]
            clusters.append(np.asarray(coords, dtype=np.float64))
        return clusters
