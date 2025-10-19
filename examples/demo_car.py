import sys
import os

base_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(base_dir)

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from custom_knn_distance import (
    create_custom_distance,
    load_and_preprocess_car_data,
    calculate_specificity,
    calculate_average_reports,
    reports_to_dataframe
)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", None)

X_encoded, y, classes = load_and_preprocess_car_data()
X = X_encoded 

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores_euclidean = []
cv_reports_euclidean = []
cv_specificities_euclidean = {label: [] for label in classes}

cv_scores_custom = []
cv_reports_custom = []
cv_specificities_custom = {label: [] for label in classes}

for train_index, test_index in kf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    knn_euclidean = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
    knn_euclidean.fit(X_train, y_train)
    preds_euclidean = knn_euclidean.predict(X_test)
    cv_scores_euclidean.append(accuracy_score(y_test, preds_euclidean))
    cv_reports_euclidean.append(classification_report(y_test, preds_euclidean, output_dict=True, zero_division=0))
    for label, spec in calculate_specificity(y_test, preds_euclidean, classes).items():
        cv_specificities_euclidean[label].append(spec)

    custom_metric = create_custom_distance(X_train)
    knn_custom = KNeighborsClassifier(n_neighbors=3, metric=custom_metric)
    knn_custom.fit(X_train, y_train)
    preds_custom = knn_custom.predict(X_test)
    cv_scores_custom.append(accuracy_score(y_test, preds_custom))
    cv_reports_custom.append(classification_report(y_test, preds_custom, output_dict=True, zero_division=0))
    for label, spec in calculate_specificity(y_test, preds_custom, classes).items():
        cv_specificities_custom[label].append(spec)

print(f"\nOrtalama doğruluk (Euclidean): {np.mean(cv_scores_euclidean)}")
print(f"Ortalama doğruluk (Custom): {np.mean(cv_scores_custom)}")

print("\nEuclidean Modeli İçin Ortalama Raporlar:")
df_euclidean = reports_to_dataframe(calculate_average_reports(cv_reports_euclidean))
print(df_euclidean)

print("\nEuclidean Modeli İçin Ortalama Özgüllükler:")
df_spec_euc = pd.DataFrame({k: [np.mean(v)] for k, v in cv_specificities_euclidean.items()}).T
df_spec_euc.columns = ["Specificity"]
print(df_spec_euc)

print("\nCustom Modeli İçin Ortalama Raporlar:")
df_custom = reports_to_dataframe(calculate_average_reports(cv_reports_custom))
print(df_custom)

print("\nCustom Modeli İçin Ortalama Özgüllükler:")
df_spec_custom = pd.DataFrame({k: [np.mean(v)] for k, v in cv_specificities_custom.items()}).T
df_spec_custom.columns = ["Specificity"]
print(df_spec_custom)