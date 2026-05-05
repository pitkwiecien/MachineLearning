import numpy as np


def mse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
	"""Compute mean squared error between predictions and targets.

	Args:
		y_pred: Predicted values.
		y_true: Ground-truth values.

	Returns:
		Mean squared error as a Python float.

	Raises:
		ValueError: If the input shapes do not match.
	"""
	y_pred_arr = np.asarray(y_pred)
	y_true_arr = np.asarray(y_true)

	if y_pred_arr.shape != y_true_arr.shape:
		raise ValueError(
			f"Shapes of y_pred {y_pred_arr.shape} and y_true {y_true_arr.shape} must match"
		)

	return float(np.mean((y_pred_arr - y_true_arr) ** 2))
