"""Scan Context place recognition descriptor.

Implements the original Scan Context (Kim & Kim, 2018 RAL) for
appearance-based loop closure detection without pose priors.

A scan context is a 2D polar-bin image (rings × sectors) encoding the
maximum height in each bin.  Ring keys (mean height per ring) enable
fast pre-filtering before full column-cosine distance computation.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import KDTree


def make_scan_context(
    points_xyz: np.ndarray,
    num_rings: int = 20,
    num_sectors: int = 60,
    max_range: float = 80.0,
) -> np.ndarray:
    """Build a Scan Context descriptor from a 3D point cloud.

    Args:
        points_xyz: ``(N, 3)`` point cloud in the sensor frame.
        num_rings: Number of radial bins.
        num_sectors: Number of angular bins.
        max_range: Maximum radial distance in metres.

    Returns:
        ``(num_rings, num_sectors)`` float32 array. Each cell holds the
        max z-value of points falling in that bin (0 if empty).
    """
    x, y, z = points_xyz[:, 0], points_xyz[:, 1], points_xyz[:, 2]
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x) + np.pi  # [0, 2π)

    # Mask out-of-range points
    valid = r < max_range
    r, theta, z = r[valid], theta[valid], z[valid]

    ring_step = max_range / num_rings
    sector_step = 2.0 * np.pi / num_sectors

    ring_idx = np.clip((r / ring_step).astype(np.int32), 0, num_rings - 1)
    sector_idx = np.clip((theta / sector_step).astype(np.int32), 0, num_sectors - 1)

    sc = np.zeros((num_rings, num_sectors), dtype=np.float32)
    # Use np.maximum.at for scatter-max
    flat_idx = ring_idx * num_sectors + sector_idx
    np.maximum.at(sc.ravel(), flat_idx, z.astype(np.float32))

    return sc


def compute_ring_key(sc: np.ndarray) -> np.ndarray:
    """Compute ring key from a scan context (mean height per ring).

    Returns:
        ``(num_rings,)`` float32 vector.
    """
    return sc.mean(axis=1).astype(np.float32)


def _column_cosine_distance(sc_a: np.ndarray, sc_b: np.ndarray) -> tuple[float, int]:
    """Column-wise cosine distance between two scan contexts, and its best shift.

    Optimized: pre-compute column norms once, then rotate with
    index arithmetic instead of np.roll for each shift.

    Returns:
        ``(distance, shift)``: the smallest distance over all sector shifts
        and the shift ``k`` that attains it (column ``c`` of ``sc_a`` matches
        column ``(c + k) % S`` of ``sc_b``).
    """
    num_sectors = sc_a.shape[1]
    norms_a = np.linalg.norm(sc_a, axis=0)  # (S,)
    norms_b = np.linalg.norm(sc_b, axis=0)  # (S,)
    # Precompute dot products via matrix multiply: (R, S).T @ (R, S) → (S, S)
    # dot_matrix[i, j] = sum over rings of sc_a[:, i] * sc_b[:, j]
    dot_matrix = sc_a.T @ sc_b  # (S, S)

    best_dist, best_shift = 1.0, 0
    for shift in range(num_sectors):
        # For shift k, column i of sc_a aligns with column (i+k)%S of sc_b
        # cos_sim[i] = dot_matrix[i, (i+shift)%S] / (norms_a[i] * norms_b[(i+shift)%S])
        j_indices = (np.arange(num_sectors) + shift) % num_sectors
        dots = dot_matrix[np.arange(num_sectors), j_indices]
        na = norms_a
        nb = norms_b[j_indices]
        valid = (na > 1e-8) & (nb > 1e-8)
        if valid.sum() == 0:
            continue
        cos_sim = dots[valid] / (na[valid] * nb[valid])
        dist = 1.0 - cos_sim.mean()
        if dist < best_dist:
            best_dist, best_shift = dist, shift
    return float(best_dist), best_shift


def sc_distance(sc_a: np.ndarray, sc_b: np.ndarray) -> float:
    """Compute Scan Context distance with rotation-invariant alignment.

    The distance is the minimum column-wise cosine distance over all
    possible sector shifts (rotation invariance).

    Returns:
        Distance in [0, 1]. Lower = more similar.
    """
    return _column_cosine_distance(sc_a, sc_b)[0]


def sc_relative_yaw(sc_query: np.ndarray, sc_match: np.ndarray) -> float:
    """Yaw [rad] of the query scan's frame relative to the matched scan's frame.

    A point at azimuth ``a`` in the query scan lies at ``a + yaw`` in the
    matched scan, so ``Rz(yaw)`` rotates query-frame points into the matched
    frame. The resolution is one sector (6 deg for 60 sectors).
    """
    num_sectors = sc_query.shape[1]
    shift = _column_cosine_distance(sc_query, sc_match)[1]
    return 2.0 * np.pi * shift / num_sectors


class ScanContextDatabase:
    """Database of scan contexts for efficient place retrieval.

    Uses ring-key KD-tree for fast candidate pre-filtering, then
    re-ranks with full SC distance.
    """

    def __init__(self, num_rings: int = 20, num_sectors: int = 60) -> None:
        self.num_rings = num_rings
        self.num_sectors = num_sectors
        self._scs: list[np.ndarray] = []
        self._ring_keys: list[np.ndarray] = []
        self._frame_indices: list[int] = []
        self._tree: KDTree | None = None
        self._tree_dirty = True

    def add(self, sc: np.ndarray, ring_key: np.ndarray, frame_idx: int) -> None:
        """Add a scan context to the database."""
        self._scs.append(sc)
        self._ring_keys.append(ring_key)
        self._frame_indices.append(frame_idx)
        self._tree_dirty = True

    def _rebuild_tree(self) -> None:
        if self._ring_keys:
            self._tree = KDTree(np.array(self._ring_keys))
        else:
            self._tree = None
        self._tree_dirty = False

    def query(
        self,
        sc: np.ndarray,
        ring_key: np.ndarray,
        top_k: int = 10,
        min_frame_gap: int = 100,
        current_frame: int = -1,
    ) -> list[tuple[int, float]]:
        if not self._scs:
            return []

        if self._tree_dirty:
            self._rebuild_tree()
        if self._tree is None:  # Guard: empty DB or rebuild produced no tree
            return []

        n_query = min(len(self._scs), top_k * 3)
        _, indices = self._tree.query(ring_key.reshape(1, -1), k=n_query)
        indices = np.asarray(indices).flatten()  # Ensure ndarray for downstream indexing

        results: list[tuple[int, float]] = []
        for idx in indices:
            fid = self._frame_indices[int(idx)]  # Cast np.intp -> int for list indexing
            if current_frame >= 0 and abs(fid - current_frame) < min_frame_gap:
                continue
            dist = sc_distance(sc, self._scs[int(idx)])
            results.append((fid, dist))

        results.sort(key=lambda x: x[1])
        return results[:top_k]
