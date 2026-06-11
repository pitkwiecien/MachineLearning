import numpy.typing as npt
import numpy as np
from typing import Self
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class ClassifierResult:
    def __init__(self, accuracy: float, precision: float, recall: float, f1: float):
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.f1 = f1

    @classmethod
    def from_predictions(cls, y_true: npt.NDArray[np.int64], y_pred: npt.NDArray[np.int64]) -> Self:
        accuracy = float(accuracy_score(y_true, y_pred))
        precision = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
        recall = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))

        return cls(accuracy, precision, recall, f1)
