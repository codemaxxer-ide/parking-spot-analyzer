"""Binary metrics with OCCUPIED (1) positive; zero denominators return zero."""

import numpy as np


def classification_metrics(truth, predictions) -> dict:
    truth, predictions = np.asarray(truth), np.asarray(predictions)
    if truth.ndim != 1 or predictions.shape != truth.shape or len(truth) == 0:
        raise ValueError("Metrics need non-empty, equally sized one-dimensional labels.")
    if not np.isin(truth, (0, 1)).all() or not np.isin(predictions, (0, 1)).all():
        raise ValueError("Only 0=VACANT and 1=OCCUPIED labels are supported.")
    matrix = np.zeros((2, 2), dtype=int)
    np.add.at(matrix, (truth.astype(int), predictions.astype(int)), 1)
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"sample_count": len(truth), "vacant_count": int((truth == 0).sum()),
            "occupied_count": int((truth == 1).sum()), "accuracy": (tp + tn) / len(truth),
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "confusion_matrix": matrix.tolist(), "positive_class": "OCCUPIED (1)",
            "matrix_order": "rows=true, columns=predicted; class order [VACANT, OCCUPIED]"}
