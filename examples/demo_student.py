import sys
import os

base_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(base_dir)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder

from custom_knn_distance import create_custom_distance

base_dir = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(base_dir, 'datasets', 'student.csv')
data = pd.read_csv(data_path, sep=';')

y_grades = data['grades']
X_features = data.drop(['grades', 'total_grades'], axis=1)

categorical_cols = X_features.select_dtypes(include=['object']).columns
encoder = OneHotEncoder(sparse_output=False, drop='first')
X_encoded = encoder.fit_transform(X_features[categorical_cols])
encoded_columns = encoder.get_feature_names_out(categorical_cols)
X_encoded_df = pd.DataFrame(X_encoded, columns=encoded_columns)

X_features_encoded = X_features.drop(categorical_cols, axis=1)
X_features_encoded = pd.concat([X_features_encoded, X_encoded_df], axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X_features_encoded, y_grades, test_size=0.20, random_state=42
)

knn_euclidean = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn_euclidean.fit(X_train, y_train)
predictions_euclidean = knn_euclidean.predict(X_test)

custom_dist = create_custom_distance(X_train)
knn_custom = KNeighborsClassifier(n_neighbors=3, metric=custom_dist)
knn_custom.fit(X_train, y_train)
predictions_custom = knn_custom.predict(X_test)

report_euclidean = classification_report(y_test, predictions_euclidean, output_dict=True)
report_custom = classification_report(y_test, predictions_custom, output_dict=True)

cm_euclidean = confusion_matrix(y_test, predictions_euclidean)
cm_custom = confusion_matrix(y_test, predictions_custom)

def calculate_specificities(cm):
    specificities = []
    for class_index in range(len(cm)):
        true_negatives = sum(cm[i, i] for i in range(len(cm))) - cm[class_index, class_index]
        false_positives = sum(cm[i, class_index] for i in range(len(cm))) - cm[class_index, class_index]
        specificity = true_negatives / (true_negatives + false_positives) if true_negatives + false_positives > 0 else 0
        specificities.append(specificity)
    return specificities

specificities_euclidean = calculate_specificities(cm_euclidean)
specificities_custom = calculate_specificities(cm_custom)

def report_confusion_matrix_metrics(cm):
    TN = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    TP = cm[1][1]
    return {"DN": TN, "YP": FP, "YN": FN, "DP": TP}

cm_metrics_knn_euclidean = report_confusion_matrix_metrics(cm_euclidean)
cm_metrics_knn_custom = report_confusion_matrix_metrics(cm_custom)

accuracy_knn_euclidean = (cm_metrics_knn_euclidean['DP'] + cm_metrics_knn_euclidean['DN']) / np.sum(cm_euclidean)
accuracy_knn_custom = (cm_metrics_knn_custom['DP'] + cm_metrics_knn_custom['DN']) / np.sum(cm_custom)

print("Confusion Matrix (Euclidean):")
print(cm_euclidean)
print("Confusion Matrix (Custom):")
print(cm_custom)

def build_summary_row(report_dict, specificities, accuracy, model_name):
    weighted_avg = report_dict['weighted avg']
    row = {
        'model': model_name,
        'precision': weighted_avg['precision'],
        'recall': weighted_avg['recall'],
        'f1-score': weighted_avg['f1-score'],
        'support': weighted_avg['support'],
        'avg_specificity': np.mean(specificities),
        'accuracy': accuracy
    }
    return row

summary_rows = [
    build_summary_row(report_euclidean, specificities_euclidean, accuracy_knn_euclidean, 'KNN_Euclidean'),
    build_summary_row(report_custom, specificities_custom, accuracy_knn_custom, 'KNN_Custom')
]

df_comparison = pd.DataFrame(summary_rows)

print("\nKNN Model Performance Comparison:\n")
print(df_comparison.to_string(index=False))