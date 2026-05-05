import numpy as np


def mse_derivative(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
	"""Derivative of mean squared error with respect to predictions.

	Args:
		y_pred: Predicted values.
		y_true: Ground-truth values.

	Returns:
		Gradient array with the same shape as `y_pred`.

	Raises:
		ValueError: If the input shapes do not match.
	"""
	y_pred_arr = np.asarray(y_pred, dtype=np.float64)
	y_true_arr = np.asarray(y_true, dtype=np.float64)

	if y_pred_arr.shape != y_true_arr.shape:
		raise ValueError(
			f"Shapes of y_pred {y_pred_arr.shape} and y_true {y_true_arr.shape} must match"
		)

	return (2.0 / y_pred_arr.size) * (y_pred_arr - y_true_arr)


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
