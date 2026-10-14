from __future__ import annotations

import numpy as np


def confusion_matrix(targets: list[int], predictions: list[int], num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for target, prediction in zip(targets, predictions):
        matrix[target, prediction] += 1
    return matrix


def metrics_from_confusion(matrix: np.ndarray) -> dict[str, float]:
    total = int(matrix.sum())
    accuracy = float(np.trace(matrix) / total) if total else 0.0
    f1_scores = []
    recalls = []
    for i in range(matrix.shape[0]):
        tp = int(matrix[i, i])
        fp = int(matrix[:, i].sum() - tp)
        fn = int(matrix[i, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_scores.append(f1)
        recalls.append(recall)
    return {
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1_scores)),
        "balanced_accuracy": float(np.mean(recalls)),
    }

