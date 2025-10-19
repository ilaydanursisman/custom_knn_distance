import numpy as np

def calculate_specificities(confusion_matrix):
    """
    Tüm sınıflar için özgüllük (specificity) hesaplar.
    Özgüllük = TN / (TN + FP)
    """
    specificities = []
    num_classes = confusion_matrix.shape[0]

    for class_idx in range(num_classes):
        TP = confusion_matrix[class_idx, class_idx]
        FP = sum(confusion_matrix[:, class_idx]) - TP
        FN = sum(confusion_matrix[class_idx, :]) - TP
        TN = confusion_matrix.sum() - (TP + FP + FN)

        specificity = TN / (TN + FP) if (TN + FP) != 0 else 0
        specificities.append(specificity)

    return specificities


def report_confusion_matrix_metrics(confusion_matrix):
    """
    Sadece ikili sınıflar (binary) için DN, YP, YN, DP değerlerini verir.
    """
    if confusion_matrix.shape != (2, 2):
        raise ValueError("Bu fonksiyon yalnızca 2x2'lik (binary) karışıklık matrisi için geçerlidir.")

    TN = confusion_matrix[0, 0]
    FP = confusion_matrix[0, 1]
    FN = confusion_matrix[1, 0]
    TP = confusion_matrix[1, 1]

    return {"DN": TN, "YP": FP, "YN": FN, "DP": TP}