import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

def calculate_specificity(y_true, y_pred, classes):
    """
    Her bir sınıf için özgüllüğü (specificity) hesaplar.

    Args:
        y_true (pd.Series or array): Gerçek etiketler
        y_pred (array): Tahmin edilen etiketler
        classes (list): Sınıf isimleri

    Returns:
        dict: Her sınıf için özgüllük değerleri
    """
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    specificities = {}
    for i, label in enumerate(classes):
        TN = sum(cm[k, k] for k in range(len(classes)) if k != i)
        FP = sum(cm[k, i] for k in range(len(classes)) if k != i)
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        specificities[label] = specificity
    return specificities

def calculate_average_reports(reports):
    """
    Birden fazla classification_report çıktısının ortalamasını alır.

    Args:
        reports (list): Her fold için classification_report sözlükleri

    Returns:
        dict: Ortalama precision, recall, f1-score ve accuracy
    """
    consolidated = {}
    accuracies = []

    for report in reports:
        for label, metrics in report.items():
            if label == 'accuracy':
                accuracies.append(metrics)
                continue

            if label not in consolidated:
                consolidated[label] = {metric: [] for metric in metrics if metric != 'support'}

            for metric, value in metrics.items():
                if metric != 'support':
                    consolidated[label][metric].append(value)

    avg_report = {}
    for label, metrics in consolidated.items():
        avg_report[label] = {metric: np.mean(vals) for metric, vals in metrics.items()}

    if accuracies:
        avg_report['accuracy'] = np.mean(accuracies)

    return avg_report

def reports_to_dataframe(average_reports):
    """
    Classification report sözlüğünü pandas DataFrame'e dönüştürür.

    Args:
        average_reports (dict): Ortalama metrik sözlüğü

    Returns:
        pd.DataFrame: DataFrame formatında metrikler
    """
    rows = []
    for label, metrics in average_reports.items():
        if isinstance(metrics, dict):
            row = {'Label': label}
            row.update(metrics)
        else:
            row = {'Label': label, 'Value': metrics}
        rows.append(row)
    return pd.DataFrame(rows)