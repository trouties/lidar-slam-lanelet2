"""Layered cache of the pipeline's expensive intermediate results.

Each stage has its own cache slot under ``cache/kitti/<seq>/``: a binary
artifact (npz / geojson) and an entry in a shared ``metadata.yaml`` recording
``config_hash``, ``upstream_hash``, the parameter snapshot, metrics and a
timestamp.

Invalidation: a stage cache loads only if (a) its own ``config_hash`` matches
the current config subtree of that stage, and (b) its ``upstream_hash``
matches the hash of the config subtrees of *all* upstream stages. Checking
only the immediate upstream is not enough: a Stage 2 change leaves the map
subtree untouched, yet the map was built from the old trajectory.

Only the config is hashed, not the code. After changing a stage's
implementation, invalidate that stage explicitly (``--force-rebuild``).

Stages form a linear dependency chain::

    odometry -> optimized -> map -> curbs

``map`` is the global map at ``mapping.master_voxel_size``, built from the
Stage 3 poses. ``curbs`` holds the map re-voxelized to
``mapping.voxel_size`` and the curb clusters extracted from it, so the curb
parameters can change without re-running the per-frame accumulation.
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

# Linear dependency chain. ``invalidate(stage)`` clears this stage and
# everything after it.
STAGE_ORDER = ["odometry", "optimized", "map", "curbs"]

# Artifact file names relative to the per-sequence cache directory.
#
# Point clouds are stored as ``.npz`` rather than ``.pcd`` because Open3D's
# PCD writer quantizes the ``colors`` channel, which carries the float
# reflectance, to uint8.
_STAGE_FILES: dict[str, list[str]] = {
    "odometry": ["odometry.npz"],
    "optimized": ["optimized.npz"],
    "map": ["map.npz"],
    "curbs": ["curbs_map.npz", "curbs.geojson"],
}


def _config_subtree(stage: str, config: dict) -> dict:
    """Extract the config subtree relevant to a given stage.

    The subtree is what gets hashed to decide whether the cached artifact for
    this stage is still valid. The result excludes ``data.sequence`` because
    the cache is already partitioned per-sequence on disk.
    """
    data_clean = {k: v for k, v in config.get("data", {}).items() if k != "sequence"}
    if stage == "odometry":
        return {"data": data_clean, "kiss_icp": config.get("kiss_icp", {})}
    if stage == "optimized":
        # The degeneracy settings change the odometry edge noise of the pose
        # graph, not the KISS-ICP registration, so they belong here and not
        # to the odometry subtree.
        return {
            "gtsam": config.get("gtsam", {}),
            "loop_closure": config.get("loop_closure", {}),
            "degeneracy": config.get("degeneracy", {}),
        }
    if stage == "map":
        m = config.get("mapping", {})
        return {
            "master_voxel_size": m.get("master_voxel_size"),
            "max_range": m.get("max_range"),
            "downsample_every": m.get("downsample_every"),
        }
    if stage == "curbs":
        # The whole mapping subtree: any curb parameter invalidates the
        # clusters, but not the map.
        return {"mapping": config.get("mapping", {})}
    raise ValueError(f"Unknown stage: {stage!r}")


def compute_hash(subtree: dict) -> str:
    """Return a 16-char SHA256 hex digest of a canonical YAML dump."""
    canonical = yaml.safe_dump(subtree, sort_keys=True, default_flow_style=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


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
        # Hash the config of *every* upstream stage, not only the previous
        # one: a Stage 2 change alters the poses that the map is built from
        # even though the map's own and its parent's subtrees are unchanged.
        idx = STAGE_ORDER.index(stage)
        if idx == 0:
            return None
        return compute_hash({s: _config_subtree(s, config) for s in STAGE_ORDER[:idx]})

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
    # Metadata queries
    # ------------------------------------------------------------------

    def get_stage_metrics(self, stage: str, config: dict) -> dict | None:
        """Return stored metrics for *stage* if the cache entry is fresh."""
        if not self._is_fresh(stage, config):
            return None
        meta = self._load_metadata()
        entry = meta.get(stage)
        return entry.get("metrics") if entry else None

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

    # ------------------------------------------------------------------
    # Degeneracy scores (independent of STAGE_ORDER)
    # ------------------------------------------------------------------
    #
    # Stored alongside odometry because they are computed from the same
    # scan stream during Stage 2. Not part of STAGE_ORDER: the settings that
    # decide how the scores enter the pose graph are in the ``optimized``
    # subtree hash.
    #
    # Fresh when (a) the odometry cache for this config is fresh and (b) the
    # analyzer settings (not the threshold or the inflation) hash the same as
    # at the last write.

    _DEGENERACY_FILE = "degeneracy.npz"
    _DEGENERACY_META_KEY = "degeneracy"

    @staticmethod
    def _degeneracy_analyzer_subtree(config: dict) -> dict:
        d = config.get("degeneracy", {}) or {}
        return {
            "max_correspondences": d.get("max_correspondences"),
            "normal_k": d.get("normal_k"),
            "max_nn_dist": d.get("max_nn_dist"),
            "voxel_size": d.get("voxel_size"),
            "min_quality": d.get("min_quality"),
        }

    def _degeneracy_is_fresh(self, config: dict) -> bool:
        if not self._is_fresh("odometry", config):
            return False
        meta = self._load_metadata()
        entry = meta.get(self._DEGENERACY_META_KEY)
        if entry is None:
            return False
        own_hash = compute_hash(self._degeneracy_analyzer_subtree(config))
        if entry.get("config_hash") != own_hash:
            return False
        if entry.get("odometry_hash") != self.hash_for("odometry", config):
            return False
        return (self.root / self._DEGENERACY_FILE).exists()

    def load_degeneracy(self, config: dict) -> np.ndarray | None:
        """Return ``(N, 7)`` per-frame degeneracy matrix or ``None``.

        Columns: ``[cond_number, lambda_min, lambda_max, n_corr,
        eig_dx, eig_dy, eig_dz]``. Null placeholder rows contain NaN.
        """
        if not self._degeneracy_is_fresh(config):
            return None
        data = np.load(self.root / self._DEGENERACY_FILE)
        return data["scores"]

    def save_degeneracy(self, scores: np.ndarray, config: dict) -> None:
        """Write the per-frame degeneracy scores.

        ``scores`` must be ``(N, 7)`` matching :meth:`load_degeneracy`.
        """
        self.root.mkdir(parents=True, exist_ok=True)
        arr = np.asarray(scores, dtype=np.float64)
        np.savez_compressed(self.root / self._DEGENERACY_FILE, scores=arr)
        meta = self._load_metadata()
        meta[self._DEGENERACY_META_KEY] = {
            "config_hash": compute_hash(self._degeneracy_analyzer_subtree(config)),
            "odometry_hash": self.hash_for("odometry", config),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "params": self._degeneracy_analyzer_subtree(config),
            "frame_count": int(arr.shape[0]),
        }
        self._save_metadata(meta)

    def load_optimized(self, config: dict) -> np.ndarray | None:
        if not self._is_fresh("optimized", config):
            return None
        data = np.load(self.root / "optimized.npz")
        return data["poses"]

    def save_optimized(self, poses: np.ndarray, config: dict, metrics: dict | None = None) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.root / "optimized.npz", poses=np.asarray(poses))
        self._write_metadata_entry("optimized", config, metrics)

    def load_map(self, config: dict) -> o3d.geometry.PointCloud | None:
        if not self._is_fresh("map", config):
            return None
        return self._load_pcd_npz(self.root / "map.npz")

    def save_map(
        self,
        pcd: o3d.geometry.PointCloud,
        config: dict,
        metrics: dict | None = None,
    ) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self._save_pcd_npz(pcd, self.root / "map.npz")
        self._write_metadata_entry("map", config, metrics)

    def load_curbs(self, config: dict) -> tuple[o3d.geometry.PointCloud, list[np.ndarray]] | None:
        """``(working-resolution map, curb clusters)``, or ``None`` when stale."""
        if not self._is_fresh("curbs", config):
            return None
        pcd = self._load_pcd_npz(self.root / "curbs_map.npz")
        curbs = self._load_clusters_geojson(self.root / "curbs.geojson")
        return pcd, curbs

    def save_curbs(
        self,
        pcd: o3d.geometry.PointCloud,
        curb_clusters: list[np.ndarray],
        config: dict,
        metrics: dict | None = None,
    ) -> None:
        """Persist the working-resolution map and its curb clusters."""
        self.root.mkdir(parents=True, exist_ok=True)
        self._save_pcd_npz(pcd, self.root / "curbs_map.npz")
        self._save_clusters_geojson(curb_clusters, self.root / "curbs.geojson")
        self._write_metadata_entry("curbs", config, metrics)

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
    def _save_clusters_geojson(
        clusters: list[np.ndarray],
        path: Path,
        feature_type: str = "curb",
    ) -> None:
        """Minimal GeoJSON dump of cluster points (schema of ``save_features_geojson``)."""
        features: list[dict[str, Any]] = []
        for idx, cluster in enumerate(clusters):
            coords = [[float(x), float(y), float(z)] for x, y, z in cluster]
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "MultiPoint", "coordinates": coords},
                    "properties": {
                        "id": idx,
                        "type": feature_type,
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
