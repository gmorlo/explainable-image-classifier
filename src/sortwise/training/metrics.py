"""Classification metrics used during validation."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_fscore_support,
)


def classification_metrics(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    *,
    class_names: tuple[str, ...] | list[str],
) -> dict[str, Any]:
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_arr,
        y_pred_arr,
        labels=list(range(len(class_names))),
        zero_division=0,
    )
    per_class = {
        name: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, name in enumerate(class_names)
    }
    return {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "macro_f1": float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)),
        "per_class": per_class,
        "report": classification_report(
            y_true_arr,
            y_pred_arr,
            target_names=list(class_names),
            zero_division=0,
        ),
    }
