import numpy as np


def mse(y_pred, y_true):
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    return float(np.mean((y_pred - y_true) ** 2))


def mse_derivative(y_pred, y_true):
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)

    return (2 / y_pred.size) * (y_pred - y_true)