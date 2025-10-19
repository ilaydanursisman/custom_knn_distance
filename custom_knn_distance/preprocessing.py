import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

def load_and_preprocess_car_data(filepath=None):
    """
    Car Evaluation veri setini yükleyip ön işler. 
    Parametre verilmezse varsayılan olarak datasets/car.data dosyasını kullanır.

    Returns:
        - X_encoded: np.ndarray, one-hot encode edilmiş özellikler
        - y: pd.Series, hedef değişken
        - classes: list, sınıf isimleri
    """
    import os

    if filepath is None:
        base_dir = os.path.dirname(os.path.dirname(__file__))
        filepath = os.path.join(base_dir, 'datasets', 'car.data')

    column_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
    data = pd.read_csv(filepath, names=column_names)

    y = data['class']
    classes = y.unique().tolist()

    X = data.drop('class', axis=1)

    encoder = OneHotEncoder(sparse_output=False, dtype=float)
    X_encoded = encoder.fit_transform(X)

    return X_encoded, y, classes