import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from custom_knn_distance.distance import D, r, M, create_custom_distance

def test_density_function():
    X = np.array([[1, 2], [2, 3], [3, 4]])
    y = np.array([2, 3])
    result = D(y, X, r=1.0)
    assert isinstance(result, float), "D(y, X, r) should return a float."
    assert result >= 0, "Density value must be non-negative."

def test_adaptive_radius():
    X = np.array([[1, 2], [2, 3], [3, 4]])
    y = np.array([2, 3])
    result = r(y, X)
    assert isinstance(result, float), "r(y, X) should return a float."
    assert result > 0, "Adaptive radius must be positive."

def test_custom_distance_value():
    X = np.array([[1, 2], [2, 3], [3, 4]])
    x = np.array([2, 2])
    y = np.array([1, 3])
    result = M(x, y, X)
    assert isinstance(result, float), "M(x, y, X) should return a float."
    assert result >= 0, "Distance must be non-negative."

def test_custom_distance_callable():
    X = np.array([[1, 2], [2, 3], [3, 4]])
    metric = create_custom_distance(X)
    assert callable(metric), "create_custom_distance(X) should return a callable function."
    distance = metric(np.array([1, 2]), np.array([2, 3]))
    assert isinstance(distance, float), "Returned distance should be float."
    assert distance >= 0, "Returned distance should be non-negative."

def test_symmetry_violation_expected():
    """
    Özel metrik simetrik olmayabilir, bu test bunu doğrular.
    """
    X = np.array([[1, 2], [2, 3], [3, 4]])
    metric = create_custom_distance(X)
    d1 = metric(np.array([1, 2]), np.array([2, 3]))
    d2 = metric(np.array([2, 3]), np.array([1, 2]))
    assert d1 != d2, "Custom metric is expected to be non-symmetric."

if __name__ == "__main__":
    test_density_function()
    test_adaptive_radius()
    test_custom_distance_value()
    test_custom_distance_callable()
    test_symmetry_violation_expected()
    print("✅ All distance function tests passed successfully.")