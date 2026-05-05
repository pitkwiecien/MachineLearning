import pathlib
from typing import Iterator

import numpy as np


def load_csv(path: str = "./data/data_scaled.csv", delimiter: str = ",") -> np.ndarray:
	"""Load a CSV file into a NumPy array.

	Args:
		path: Path to CSV file.
		delimiter: Field delimiter.

	Returns:
		Array of shape (n_rows, n_features).

	Raises:
		FileNotFoundError: If the file does not exist.
		ValueError: If the file cannot be parsed as numeric CSV.
	"""
	p = pathlib.Path(path)
	if not p.exists():
		raise FileNotFoundError(f"CSV file not found: {path}")
	try:
		return np.loadtxt(p, delimiter=delimiter)
	except ValueError as exc:
		raise ValueError(f"Failed to parse CSV '{path}': {exc}") from exc


def csv_batch_generator(
	batch_size: int,
	path: str = "../data/data_scaled.csv",
	shuffle: bool = True,
	drop_last: bool = False,
	delimiter: str = ",",
) -> Iterator[np.ndarray]:
	"""Yield successive batches (rows) from the CSV file as NumPy arrays.

	Args:
		batch_size: Number of rows per batch.
		path: Path to CSV file. Default: `./data/data_scaled.csv`.
		shuffle: Whether to shuffle rows before batching.
		drop_last: If True, drop the final batch when its size < batch_size.
		delimiter: CSV delimiter.

	Yields:
		NumPy arrays with shape (k, n_features), where k == batch_size except
		possibly the final batch (unless `drop_last` is True).
	"""
	if batch_size <= 0:
		raise ValueError("batch_size must be a positive integer")

	data = load_csv(path, delimiter=delimiter)

	if data.ndim == 1:
		data = data.reshape(-1, 1)

	n = data.shape[0]
	indices = np.arange(n)
	if shuffle:
		np.random.shuffle(indices)
		data = data[indices]

	for start in range(0, n, batch_size):
		end = start + batch_size
		batch = data[start:end]
		if batch.shape[0] < batch_size and drop_last:
			break
		yield batch


def load_batches_as_list(batch_size: int, **kwargs) -> list:
	"""Return all batches as a list (convenience wrapper around the generator)."""
	return list(csv_batch_generator(batch_size, **kwargs))
