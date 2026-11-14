"""Tests for the metric version of the KITTI ground truth."""

from __future__ import annotations

import numpy as np
from pyproj import Geod

from src.benchmarks.metric_reference import (
    devkit_mercator,
    devkit_mercator_inverse,
    local_conformal,
)

LAT0, LON0 = 49.0, 8.4


def test_devkit_mercator_round_trip():
    lat = np.array([49.0, 49.003, 48.996])
    lon = np.array([8.4, 8.41, 8.392])
    mx, my = devkit_mercator(lat, lon, LAT0)
    lat2, lon2 = devkit_mercator_inverse(mx, my, LAT0)
    assert np.allclose(lat2, lat, atol=1e-12) and np.allclose(lon2, lon, atol=1e-12)


def test_devkit_mercator_is_anisotropic_at_49_degrees():
    """North-south distances come out about 0.10 % long, east-west about 0.19 % short."""
    geod = Geod(ellps="WGS84")
    d = 0.01  # degrees, about 1 km
    lat = np.array([LAT0, LAT0 + d, LAT0])
    lon = np.array([LON0, LON0, LON0 + d])
    mx, my = devkit_mercator(lat, lon, LAT0)
    _, _, true_n = geod.inv(LON0, LAT0, LON0, LAT0 + d)
    _, _, true_e = geod.inv(LON0, LAT0, LON0 + d, LAT0)
    k_n = (my[1] - my[0]) / true_n
    k_e = (mx[2] - mx[0]) / true_e
    assert abs(100 * (k_n - 1) - 0.099) < 0.01
    assert abs(100 * (k_e - 1) + 0.191) < 0.01


def test_local_conformal_has_true_distances():
    geod = Geod(ellps="WGS84")
    lat = np.array([LAT0, LAT0 + 0.01, LAT0 - 0.004])
    lon = np.array([LON0, LON0 + 0.02, LON0 - 0.015])
    x, y = local_conformal(lat, lon, LAT0, LON0)
    for a, b in ((0, 1), (0, 2), (1, 2)):
        _, _, true = geod.inv(lon[a], lat[a], lon[b], lat[b])
        assert abs(np.hypot(x[a] - x[b], y[a] - y[b]) - true) < 1e-3
