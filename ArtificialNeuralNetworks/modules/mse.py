import numpy as np
def to_one_hot(y, num_classes):
    y = y.astype(int)
    one_hot = np.zeros((y.size, num_classes))
    one_hot[np.arange(y.size), y] = 1
    return one_hot

def mse(y_pred, y_true):
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)

    if y_true.ndim == 1:
        y_true = to_one_hot(y_true, y_pred.shape[1])

    return float(np.mean((y_pred - y_true) ** 2))


def mse_derivative(y_pred, y_true):
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)

    if y_true.ndim == 1:
            y_true = to_one_hot(y_true, y_pred.shape[1])
            
    return (2 / y_pred.size) * (y_pred - y_true)