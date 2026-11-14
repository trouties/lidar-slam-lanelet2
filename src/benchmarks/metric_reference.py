"""A metric version of the KITTI odometry ground truth.

The KITTI devkit turns the OxTS positions into poses with a spherical
Mercator projection (radius 6378137 m, scaled by the cosine of the start
latitude) applied to WGS84 geodetic coordinates (``latlonToMercator`` and
``convertOxtsToPose``). On the ellipsoid this projection is not conformal:
around Karlsruhe (49 deg N) it lengthens north-south distances by about
0.10 % and shortens east-west distances by about 0.19 %. The ground-truth
positions therefore carry a direction-dependent scale error, while the
orientations come straight from the inertial unit and do not.

:func:`metric_ground_truth` removes the projection: it maps each
ground-truth position back onto the devkit's Mercator plane, inverts the
projection to latitude and longitude, projects those with a conformal
projection of unit scale centred at the start (true horizontal distances),
and expresses the result in the frame of the original ground truth again.
Heights and orientations are kept, so the frame stays flattened like the
original and differs from it only by the devkit projection. The only external
input is the OxTS log of the drive, used to locate the ground-truth frame on
the Mercator plane (start latitude, heading); the positions themselves are
the ground truth's own.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from evo.core.geometry import umeyama_alignment
from pyproj import Transformer

from src.export.georef import georeference_from_gnss, gnss_at, load_oxts_monotonic

# Radius the devkit's latlonToMercator uses (the WGS84 equatorial radius).
DEVKIT_EARTH_RADIUS_M = 6378137.0


def devkit_mercator(lat: np.ndarray, lon: np.ndarray, lat0: float) -> tuple[np.ndarray, np.ndarray]:
    """KITTI devkit ``latlonToMercator`` (degrees in, metres out)."""
    s = np.cos(np.radians(lat0))
    mx = s * DEVKIT_EARTH_RADIUS_M * np.radians(lon)
    my = s * DEVKIT_EARTH_RADIUS_M * np.log(np.tan(np.radians(90.0 + np.asarray(lat)) / 2.0))
    return mx, my


def devkit_mercator_inverse(
    mx: np.ndarray, my: np.ndarray, lat0: float
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of :func:`devkit_mercator` (metres in, degrees out)."""
    sr = np.cos(np.radians(lat0)) * DEVKIT_EARTH_RADIUS_M
    lon = np.degrees(np.asarray(mx) / sr)
    lat = np.degrees(2.0 * np.arctan(np.exp(np.asarray(my) / sr))) - 90.0
    return lat, lon


def local_conformal(
    lat: np.ndarray, lon: np.ndarray, lat0: float, lon0: float
) -> tuple[np.ndarray, np.ndarray]:
    """Transverse Mercator on WGS84 centred at (lat0, lon0) with scale 1.

    Conformal, and within a few kilometres of the centre its scale differs
    from 1 by less than 1e-7, so distances in it are true distances. Heights
    stay as they are: like the devkit's poses, the frame is flattened.
    """
    proj = Transformer.from_crs(
        "EPSG:4326",
        f"+proj=tmerc +lat_0={lat0} +lon_0={lon0} +k=1 +x_0=0 +y_0=0 +ellps=WGS84",
        always_xy=True,
    )
    x, y = proj.transform(np.asarray(lon).tolist(), np.asarray(lat).tolist())
    return np.asarray(x), np.asarray(y)


def _rigid(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r, t, _ = umeyama_alignment(src.T, dst.T, with_scale=False)
    return r, t


@dataclass
class MetricReference:
    """Ground truth with the projection removed, and what the correction did.

    Attributes:
        poses: ``(N, 4, 4)`` corrected Velodyne-frame poses (same frame and
            orientations as the input ground truth).
        time_offset_s: GNSS log time offset used to locate the trajectory.
        devkit_fit_rms_m: Horizontal RMS between the ground truth and the
            OxTS positions in the devkit projection after a rigid fit; small
            values confirm that the ground truth is that projection.
        correction_rms_m, correction_max_m: Size of the position correction
            (horizontal), i.e. the non-rigid part of the projection error.
    """

    poses: np.ndarray
    time_offset_s: float
    devkit_fit_rms_m: float
    correction_rms_m: float
    correction_max_m: float


def metric_ground_truth(
    gt_velo: np.ndarray, timestamps: np.ndarray, oxts_dir: Path, lever_arm: np.ndarray
) -> MetricReference:
    """Remove the devkit projection from Velodyne-frame ground-truth poses.

    Args:
        gt_velo: ``(N, 4, 4)`` ground truth in the Velodyne frame.
        timestamps: ``(N,)`` scan timestamps [s].
        oxts_dir: KITTI raw ``oxts`` directory of the drive.
        lever_arm: GNSS/INS unit origin in Velodyne coordinates.
    """
    gt_velo = np.asarray(gt_velo, dtype=np.float64)
    # Time offset of the log, from the ground truth itself (fit in UTM).
    offset = georeference_from_gnss(gt_velo, timestamps, oxts_dir, lever_arm).time_offset_s
    data, t_ox = load_oxts_monotonic(oxts_dir)
    lat, lon, alt, _ = gnss_at(data, t_ox, timestamps, offset)
    lat0 = float(lat[0])

    # Positions of the GNSS/INS unit in the ground-truth frame.
    p = np.einsum("nij,j->ni", gt_velo[:, :3, :3], lever_arm) + gt_velo[:, :3, 3]

    # Locate the ground-truth frame on the devkit's Mercator plane.
    mx, my = devkit_mercator(lat, lon, lat0)
    m = np.column_stack([mx, my, alt])
    r, t = _rigid(p, m)
    res = p @ r.T + t - m
    devkit_fit_rms = float(np.sqrt(np.mean(res[:, 0] ** 2 + res[:, 1] ** 2)))

    # Ground-truth positions -> Mercator plane -> geodetic -> local conformal frame.
    pm = p @ r.T + t
    glat, glon = devkit_mercator_inverse(pm[:, 0], pm[:, 1], lat0)
    ex, ny = local_conformal(glat, glon, float(glat[0]), float(glon[0]))
    metric = np.column_stack([ex, ny, pm[:, 2]])

    # Back into the ground-truth frame with a rigid fit; the remainder is the
    # projection's non-rigid part, which is what the correction changes.
    r2, t2 = _rigid(metric, p)
    q = metric @ r2.T + t2
    corr = np.hypot(q[:, 0] - p[:, 0], q[:, 1] - p[:, 1])

    out = gt_velo.copy()
    out[:, :3, 3] = q - np.einsum("nij,j->ni", gt_velo[:, :3, :3], lever_arm)
    return MetricReference(
        poses=out,
        time_offset_s=float(offset),
        devkit_fit_rms_m=devkit_fit_rms,
        correction_rms_m=float(np.sqrt(np.mean(corr**2))),
        correction_max_m=float(corr.max()),
    )
