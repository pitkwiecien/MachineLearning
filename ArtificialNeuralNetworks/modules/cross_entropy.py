import numpy as np


def softmax(logits):
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=1, keepdims=True)


def softmax_cross_entropy_with_logits(logits, y):
    probs = softmax(logits)
    N = logits.shape[0]
    eps = 1e-12

    if y.ndim == 1:
        correct = probs[np.arange(N), y.astype(int)]
        return float(-np.mean(np.log(correct + eps)))

    return float(-np.mean(np.sum(y * np.log(probs + eps), axis=1)))


def grad_softmax_cross_entropy_logits(logits, y):
    probs = softmax(logits)
    N = logits.shape[0]

    if y.ndim == 1:
        y_onehot = np.zeros_like(probs)
        y_onehot[np.arange(N), y.astype(int)] = 1
    else:
        y_onehot = y

    return (probs - y_onehot) / N