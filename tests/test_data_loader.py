"""Placeholder tests for data loading module."""

import numpy as np

from src.data.transforms import apply_transform, latlon_to_mercator


def test_latlon_to_mercator_returns_floats():
    x, y = latlon_to_mercator(48.1351, 11.5820)  # Munich
    assert isinstance(x, float)
    assert isinstance(y, float)


def test_apply_transform_identity():
    points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    T = np.eye(4)
    result = apply_transform(points, T)
    np.testing.assert_array_almost_equal(result, points)
