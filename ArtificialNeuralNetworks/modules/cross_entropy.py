from __future__ import annotations

import numpy as np
import numpy.typing as npt


def _ensure_batch(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    return x


def softmax(logits: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    z = _ensure_batch(logits)
    z_max = np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z - z_max)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def softmax_cross_entropy_with_logits(
    logits: npt.NDArray[np.float64],
    y_true: npt.NDArray[np.int_] | npt.NDArray[np.float64],
) -> float:
    """Mean softmax + cross-entropy computed from logits (numerically stable).

    Args:
        logits: shape (N, C) or (C,) for single sample.
        y_true: integer class indices shape (N,) or one-hot shape (N, C).

    Returns:
        Mean cross-entropy loss as Python float.
    """
    logits = _ensure_batch(logits)
    N, C = logits.shape

    y = np.asarray(y_true)
    if y.ndim == 1 and np.issubdtype(y.dtype, np.integer):
        labels = y
        z_max = np.max(logits, axis=1)
        lse = np.log(np.sum(np.exp(logits - z_max[:, None]), axis=1)) + z_max
        loss = np.mean(lse - logits[np.arange(N), labels])
    else:
        y_onehot = _ensure_batch(y.astype(np.float64))
        z_max = np.max(logits, axis=1, keepdims=True)
        lse = np.log(np.sum(np.exp(logits - z_max), axis=1, keepdims=True)) + z_max
        loss = -np.sum(y_onehot * (logits - lse)) / N

    return float(loss)


def grad_softmax_cross_entropy_logits(
    logits: npt.NDArray[np.float64],
    y_true: npt.NDArray[np.int_] | npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Gradient of mean softmax cross-entropy with respect to logits.

    Args:
        logits: shape (N, C) or (C,) for single sample.
        y_true: integer class indices shape (N,) or one-hot shape (N, C).

    Returns:
        Gradient array with same shape as logits (after batch normalization).
    """
    logits = _ensure_batch(logits)
    probs = softmax(logits)
    N, C = probs.shape

    y = np.asarray(y_true)
    if y.ndim == 1 and np.issubdtype(y.dtype, np.integer):
        y_onehot = np.eye(C, dtype=np.float64)[y]
    else:
        y_onehot = _ensure_batch(y.astype(np.float64))

    return (probs - y_onehot) / N

