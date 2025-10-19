import sys
import os
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from custom_knn_distance.evaluation import (
    calculate_specificity,
    calculate_average_reports,
    reports_to_dataframe
)

def test_calculate_specificity():
    y_true = pd.Series(["a", "b", "a", "b", "c", "c"])
    y_pred = pd.Series(["a", "b", "c", "b", "c", "a"])
    classes = ["a", "b", "c"]
    
    specificities = calculate_specificity(y_true, y_pred, classes)
    assert isinstance(specificities, dict), "Sonuç bir sözlük olmalı"
    assert all(cls in specificities for cls in classes), "Tüm sınıflar sözlükte bulunmalı"
    print("✅ calculate_specificity fonksiyonu başarılı.")

def test_calculate_average_reports():
    y_true_1 = ["a", "b", "a", "c"]
    y_pred_1 = ["a", "b", "b", "c"]
    y_true_2 = ["a", "a", "b", "c"]
    y_pred_2 = ["a", "a", "b", "b"]

    report1 = classification_report(y_true_1, y_pred_1, output_dict=True)
    report2 = classification_report(y_true_2, y_pred_2, output_dict=True)

    avg_report = calculate_average_reports([report1, report2])
    assert isinstance(avg_report, dict), "Ortalama rapor sözlük olmalı"
    assert "accuracy" in avg_report, "'accuracy' metrikte bulunmalı"
    print("✅ calculate_average_reports fonksiyonu başarılı.")

def test_reports_to_dataframe():
    mock_report = {
        "accuracy": 0.8,
        "weighted avg": {"precision": 0.75, "recall": 0.8, "f1-score": 0.76, "support": 100}
    }
    df = reports_to_dataframe(mock_report)
    assert isinstance(df, pd.DataFrame), "Sonuç bir DataFrame olmalı"
    assert "precision" in df.columns, "precision kolonu olmalı"
    print("✅ reports_to_dataframe fonksiyonu başarılı.")

if __name__ == "__main__":
    test_calculate_specificity()
    test_calculate_average_reports()
    test_reports_to_dataframe()