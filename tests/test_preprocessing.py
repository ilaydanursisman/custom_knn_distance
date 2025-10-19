import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd

from custom_knn_distance.preprocessing import load_and_preprocess_car_data

def test_load_and_preprocess_car_data():

    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(base_dir, 'datasets', 'car.data')

    X_encoded, y, classes = load_and_preprocess_car_data(data_path)

    assert isinstance(X_encoded, np.ndarray), "X_encoded numpy array olmalı"
    assert isinstance(y, pd.Series), "y bir pandas Series olmalı"
    assert isinstance(classes, list), "classes bir liste olmalı"
    assert len(X_encoded) == len(y), "Özellik ve hedef uzunluğu eşit olmalı"
    assert set(y.unique()) == set(classes), "Sınıflar doğru tanımlanmalı"

    print("✅ Preprocessing function test passed.")

if __name__ == "__main__":
    test_load_and_preprocess_car_data()