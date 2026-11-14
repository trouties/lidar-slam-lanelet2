"""Georeferencing of the map frame with the vehicle's GNSS/INS log.

The map frame is the sensor frame of the first scan. To place the map on the
earth, a rigid transformation (three rotations, three translations, no scale)
from the map frame to UTM is estimated with the trajectory as control points:
the positions of the GNSS/INS unit in the map frame, obtained from the SLAM
poses and the unit's lever arm, are fitted to the unit's own positions
projected to UTM. In surveying terms this is a six-parameter Helmert
transformation; its residuals are the distortion of the SLAM trajectory
relative to the GNSS positions, plus the GNSS error.

For KITTI, the GNSS/INS log is the OxTS file of the KITTI Raw drive that the
odometry sequence was cut from. The KITTI ground truth is derived from the
same unit, so a map georeferenced this way does not use the ground truth but
is not independent of it either.

Without a GNSS log, :func:`nominal_georeference` places the map frame at a
given latitude/longitude with its axes on the UTM grid. Lanelet2's
``UtmProjector`` with that origin then returns the map coordinates exactly,
but the map is not georeferenced.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from evo.core.geometry import umeyama_alignment
from pyproj import Transformer

from src.data.kitti_raw import ODOM_TO_RAW, load_oxts_sequence

# OxTS log columns (KITTI Raw dataformat.txt).
_LAT, _LON, _ALT, _VN, _VE, _POSMODE = 0, 1, 2, 6, 7, 27
# OxTS position modes with an RTK solution (5 = float, 6 = integer).
_RTK_MODES = (5, 6)


def utm_epsg(lat: float, lon: float) -> int:
    """EPSG code of the WGS84 / UTM zone containing ``(lat, lon)``."""
    zone = int((lon + 180.0) // 6.0) + 1
    return (32600 if lat >= 0 else 32700) + zone


@dataclass
class GeoReference:
    """A rigid map-to-UTM transformation and how well it fits.

    Attributes:
        map_to_utm: ``(4, 4)`` transformation from map coordinates to
            (easting, northing, altitude) in ``epsg``.
        epsg: EPSG code of the UTM zone.
        source: ``"gnss"`` when fitted to a GNSS log, ``"nominal"`` otherwise.
        time_offset_s: Offset added to the scan timestamps to read the GNSS
            log (``gnss`` only).
        n_control: Number of trajectory points used in the fit.
        residual_horizontal_rms_m, residual_horizontal_max_m,
        residual_vertical_rms_m: Fit residuals over the control points.
    """

    map_to_utm: np.ndarray
    epsg: int
    source: str
    time_offset_s: float | None = None
    n_control: int = 0
    residual_horizontal_rms_m: float | None = None
    residual_horizontal_max_m: float | None = None
    residual_vertical_rms_m: float | None = None

    def to_latlon(self, xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Map coordinates ``(N, 3)`` -> (lat, lon, altitude) arrays."""
        xyz = np.atleast_2d(np.asarray(xyz, dtype=np.float64))
        utm = xyz @ self.map_to_utm[:3, :3].T + self.map_to_utm[:3, 3]
        to_wgs84 = Transformer.from_crs(f"EPSG:{self.epsg}", "EPSG:4326", always_xy=True)
        # Plain lists: pyproj treats a length-1 array as a scalar and warns.
        lon, lat = to_wgs84.transform(utm[:, 0].tolist(), utm[:, 1].tolist())
        return np.asarray(lat), np.asarray(lon), utm[:, 2]

    def summary(self) -> dict:
        return {
            "source": self.source,
            "epsg": self.epsg,
            "map_to_utm": np.round(self.map_to_utm, 6).tolist(),
            "time_offset_s": self.time_offset_s,
            "n_control": self.n_control,
            "residual_horizontal_rms_m": self.residual_horizontal_rms_m,
            "residual_horizontal_max_m": self.residual_horizontal_max_m,
            "residual_vertical_rms_m": self.residual_vertical_rms_m,
        }


def nominal_georeference(lat0: float, lon0: float) -> GeoReference:
    """Map origin at ``(lat0, lon0)``, x east and y north on the UTM grid."""
    epsg = utm_epsg(lat0, lon0)
    e0, n0 = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True).transform(lon0, lat0)
    T = np.eye(4)
    T[0, 3], T[1, 3] = e0, n0
    return GeoReference(map_to_utm=T, epsg=epsg, source="nominal")


def load_imu_lever_arm(raw_root: Path, date: str) -> np.ndarray:
    """Position of the GNSS/INS unit in the Velodyne frame (KITTI Raw calib).

    ``calib_imu_to_velo.txt`` holds R, T with ``p_velo = R p_imu + T``, so
    the unit's origin sits at ``T`` in Velodyne coordinates.
    """
    for line in (Path(raw_root) / date / "calib_imu_to_velo.txt").read_text().splitlines():
        if line.startswith("T:"):
            return np.array([float(v) for v in line.split()[1:4]])
    raise ValueError(f"no T in calib_imu_to_velo.txt for {date}")


def oxts_dir_for_sequence(raw_root: Path, sequence: str) -> Path | None:
    """OxTS directory of the KITTI Raw drive behind an odometry sequence."""
    if sequence not in ODOM_TO_RAW:
        return None
    date, drive = ODOM_TO_RAW[sequence]
    d = Path(raw_root).expanduser() / date / f"{date}_drive_{drive}_extract" / "oxts"
    return d if (d / "data").is_dir() else None


def load_oxts_monotonic(oxts_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """OxTS log with samples whose timestamp goes backwards dropped.

    The unsynchronised 100 Hz logs contain a few such samples; linear
    interpolation needs an increasing time axis.
    """
    data, t = load_oxts_sequence(oxts_dir)
    keep = np.concatenate([[True], t[1:] > np.maximum.accumulate(t)[:-1]])
    return data[keep], t[keep]


def gnss_at(
    data: np.ndarray, t_ox: np.ndarray, timestamps: np.ndarray, offset: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """GNSS latitude, longitude, altitude at ``timestamps + offset``.

    Positions are interpolated linearly in the 100 Hz log. The fourth array
    tells whether the nearest log sample has an RTK solution.
    """
    ts = np.asarray(timestamps, dtype=np.float64) + offset
    lat = np.interp(ts, t_ox, data[:, _LAT])
    lon = np.interp(ts, t_ox, data[:, _LON])
    alt = np.interp(ts, t_ox, data[:, _ALT])
    mode = data[np.searchsorted(t_ox, ts).clip(0, len(t_ox) - 1), _POSMODE].astype(int)
    return lat, lon, alt, np.isin(mode, _RTK_MODES)


def _rigid_fit(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    r, t, _ = umeyama_alignment(src.T, dst.T, with_scale=False)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = r, t
    return T


def georeference_from_gnss(
    poses: np.ndarray,
    timestamps: np.ndarray,
    oxts_dir: Path,
    lever_arm: np.ndarray,
    coarse_window_s: tuple[float, float] | None = None,
    fine_halfwidth_s: float = 0.5,
    fine_step_s: float = 0.005,
) -> GeoReference:
    """Fit the map-to-UTM transformation to a KITTI Raw OxTS log.

    The log is not synchronised with the scans. A coarse time offset comes
    from matching ground speed (GNSS velocity vs. trajectory speed); it is
    refined by minimising the horizontal residual of the rigid fit, which is
    much sharper: a 0.1 s error is about 1 m along track at urban speeds,
    and the speed profile cannot resolve it.

    Only epochs with an RTK solution (float or integer) are control points.

    Args:
        poses: ``(N, 4, 4)`` Velodyne-to-map poses.
        timestamps: ``(N,)`` scan timestamps [s].
        oxts_dir: KITTI Raw ``oxts`` directory.
        lever_arm: GNSS/INS unit origin in Velodyne coordinates.
    """
    poses = np.asarray(poses, dtype=np.float64)
    t = np.asarray(timestamps, dtype=np.float64)
    data, t_ox = load_oxts_monotonic(oxts_dir)
    epsg = utm_epsg(float(data[0, _LAT]), float(data[0, _LON]))
    to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)

    p_unit = np.einsum("nij,j->ni", poses[:, :3, :3], lever_arm) + poses[:, :3, 3]

    speed_traj = np.linalg.norm(np.diff(p_unit, axis=0), axis=1) / np.diff(t)
    speed_gnss = np.hypot(data[:, _VN], data[:, _VE])
    if coarse_window_s is None:
        # An odometry sequence can start well into its raw drive (sequence 08
        # starts at frame 1100), so search every offset the log allows.
        coarse_window_s = (-5.0, float(t_ox[-1] - t[-1]) + 5.0)
    coarse = np.arange(*coarse_window_s, 0.02)
    mse = [np.mean((np.interp(t[:-1] + o, t_ox, speed_gnss) - speed_traj) ** 2) for o in coarse]
    o_coarse = float(coarse[int(np.argmin(mse))])

    def sample(offset: float) -> tuple[np.ndarray, np.ndarray]:
        lat, lon, alt, rtk = gnss_at(data, t_ox, t, offset)
        e, n = to_utm.transform(lon, lat)
        return np.column_stack([e, n, alt]), rtk

    best = None
    for o in o_coarse + np.arange(-fine_halfwidth_s, fine_halfwidth_s + 1e-9, fine_step_s):
        utm, rtk = sample(o)
        T = _rigid_fit(p_unit[rtk], utm[rtk])
        r = p_unit[rtk] @ T[:3, :3].T + T[:3, 3] - utm[rtk]
        h_rms = float(np.sqrt(np.mean(r[:, 0] ** 2 + r[:, 1] ** 2)))
        if best is None or h_rms < best[0]:
            best = (h_rms, float(o), T, r)

    h_rms, offset, T, r = best
    return GeoReference(
        map_to_utm=T,
        epsg=epsg,
        source="gnss",
        time_offset_s=round(offset, 3),
        n_control=int(r.shape[0]),
        residual_horizontal_rms_m=round(h_rms, 3),
        residual_horizontal_max_m=round(float(np.hypot(r[:, 0], r[:, 1]).max()), 3),
        residual_vertical_rms_m=round(float(np.sqrt(np.mean(r[:, 2] ** 2))), 3),
    )


def georeference_kitti(
    raw_root: str | Path, sequence: str, poses: np.ndarray, timestamps: np.ndarray
) -> GeoReference | None:
    """:func:`georeference_from_gnss` for a KITTI odometry sequence.

    Returns ``None`` when the OxTS log of the sequence's KITTI Raw drive is
    not under ``raw_root``.
    """
    raw_root = Path(raw_root).expanduser()
    oxts_dir = oxts_dir_for_sequence(raw_root, sequence)
    if oxts_dir is None:
        return None
    date = ODOM_TO_RAW[sequence][0]
    return georeference_from_gnss(poses, timestamps, oxts_dir, load_imu_lever_arm(raw_root, date))
