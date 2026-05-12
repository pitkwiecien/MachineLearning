from dataclasses import dataclass
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: np.ndarray

    def __str__(self) -> str:
        return (
            f"Accuracy:  {self.accuracy:.4f}\n"
            f"Precision: {self.precision:.4f}\n"
            f"Recall:    {self.recall:.4f}\n"
            f"F1-score:  {self.f1:.4f}\n"
            f"Confusion matrix:\n{self.confusion_matrix}"
        )


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred_logits: np.ndarray,
    average: str = "macro",
) -> ClassificationMetrics:

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.argmax(y_pred_logits, axis=1)

    return ClassificationMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        confusion_matrix=confusion_matrix(y_true, y_pred),
    )