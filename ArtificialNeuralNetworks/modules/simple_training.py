from dataclasses import dataclass
from typing import Callable

import numpy as np
import numpy.typing as npt

from .ActivationFunction import ActivationFunction
from .MLP import MLP
from .cross_entropy import grad_softmax_cross_entropy_logits, softmax_cross_entropy_with_logits
from .csv_read import load_train_validation_test_split
from .mse import mse, mse_derivative


FloatArray = npt.NDArray[np.float64]


ACTIVATION_NAMES: dict[str, ActivationFunction] = {
    "relu": ActivationFunction.RELU,
    "sigmoid": ActivationFunction.SIGMOID,
    "tanh": ActivationFunction.TANH,
    "linear": ActivationFunction.LINEAR,
    "softmax": ActivationFunction.SOFTMAX,
}


@dataclass
class TrainingConfig:
    hidden_sizes: list[int]
    hidden_activation: str = "relu"
    output_activation: str = "linear"
    loss_name: str = "cross_entropy"
    learning_rate: float = 0.01
    epochs: int = 50
    batch_size: int | None = 32
    test_size: float = 0.2
    val_size: float = 0.2
    random_state: int | None = 42
    shuffle: bool = True
    round_output: bool = False


def _loss_pair(loss_name: str) -> tuple[
    Callable[[FloatArray, FloatArray], float],
    Callable[[FloatArray, FloatArray], FloatArray],
]:
    normalized = loss_name.strip().lower()

    if normalized == "mse":
        return mse, mse_derivative

    if normalized in {"cross_entropy", "ce", "softmax_cross_entropy"}:
        def loss_function(y_pred: FloatArray, y_true: FloatArray) -> float:
            return softmax_cross_entropy_with_logits(y_pred.T, y_true.T)

        def loss_derivative(y_pred: FloatArray, y_true: FloatArray) -> FloatArray:
            return grad_softmax_cross_entropy_logits(y_pred.T, y_true.T).T

        return loss_function, loss_derivative

    raise ValueError(f"Unknown loss function: {loss_name}")


def _activation(name: str) -> ActivationFunction:
    try:
        return ACTIVATION_NAMES[name.strip().lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown activation function: {name}") from exc


def _encode_targets(y_train: FloatArray, y_val: FloatArray, y_test: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray, np.ndarray]:
    all_labels = np.concatenate([y_train, y_val, y_test]).astype(int)
    classes = np.unique(all_labels)
    class_to_index = {int(label): index for index, label in enumerate(classes)}

    def encode(y: FloatArray) -> FloatArray:
        y_int = np.asarray(y, dtype=int)
        indices = np.vectorize(class_to_index.get)(y_int)
        return np.eye(len(classes), dtype=np.float64)[indices]

    return encode(y_train), encode(y_val), encode(y_test), classes


def _minibatches(X: FloatArray, y: FloatArray, batch_size: int | None, shuffle: bool, rng: np.random.Generator):
    effective_batch_size = int(X.shape[0]) if batch_size is None else int(batch_size)
    if effective_batch_size <= 0:
        raise ValueError("batch_size must be a positive integer or None")

    indices = np.arange(int(X.shape[0]))
    if shuffle:
        rng.shuffle(indices)

    for start in range(0, len(indices), effective_batch_size):
        batch_indices = indices[start:start + effective_batch_size]
        yield X[batch_indices].T, y[batch_indices].T


def build_model(input_size: int, output_size: int, config: TrainingConfig) -> MLP:
    activations = [_activation(config.hidden_activation)] * len(config.hidden_sizes)
    if config.loss_name.strip().lower() in {"cross_entropy", "ce", "softmax_cross_entropy"}:
        activations.append(ActivationFunction.LINEAR)
    else:
        activations.append(_activation(config.output_activation))
    layer_sizes = [input_size, *config.hidden_sizes, output_size]
    return MLP(layer_sizes, activations, round_output=config.round_output)


def evaluate_model(
    model: MLP,
    X: FloatArray,
    y: FloatArray,
    loss_name: str,
    batch_size: int | None = 32,
) -> tuple[float, float]:
    loss_function, _ = _loss_pair(loss_name)
    batch_losses: list[float] = []
    predictions: list[FloatArray] = []

    effective_batch_size = int(X.shape[0]) if batch_size is None else int(batch_size)

    for start in range(0, X.shape[0], effective_batch_size):
        batch_indices = slice(start, start + effective_batch_size)
        X_batch = X[batch_indices].T
        y_batch = y[batch_indices].T
        y_pred = model.forward(X_batch)
        batch_losses.append(loss_function(y_pred, y_batch))
        predictions.append(y_pred)

    predicted_matrix = np.concatenate(predictions, axis=1)
    predicted_labels = np.argmax(predicted_matrix, axis=0)
    true_labels = np.argmax(y, axis=1)
    accuracy = float(np.mean(predicted_labels == true_labels))

    return float(np.mean(batch_losses)), accuracy


def train_validate_test(
    path: str,
    config: TrainingConfig,
) -> tuple[MLP, dict[str, list[float]], dict[str, float], np.ndarray]:
    X_train, X_val, X_test, y_train, y_val, y_test = load_train_validation_test_split(
        path=path,
        val_size=config.val_size,
        test_size=config.test_size,
        random_state=config.random_state,
        shuffle=config.shuffle,
    )

    y_train_encoded, y_val_encoded, y_test_encoded, classes = _encode_targets(y_train, y_val, y_test)
    model = build_model(X_train.shape[1], len(classes), config)
    loss_function, loss_derivative = _loss_pair(config.loss_name)

    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    rng = np.random.default_rng(config.random_state)

    for _ in range(config.epochs):
        batch_losses: list[float] = []
        for X_batch, y_batch in _minibatches(X_train, y_train_encoded, config.batch_size, config.shuffle, rng):
            batch_losses.append(model.train_step(X_batch, y_batch, loss_function, loss_derivative, config.learning_rate))

        train_loss, train_accuracy = evaluate_model(model, X_train, y_train_encoded, config.loss_name, config.batch_size)
        val_loss, val_accuracy = evaluate_model(model, X_val, y_val_encoded, config.loss_name, config.batch_size)

        history["train_loss"].append(float(np.mean(batch_losses)))
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

    test_loss, test_accuracy = evaluate_model(model, X_test, y_test_encoded, config.loss_name, config.batch_size)
    metrics = {
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "classes": classes,
    }

    return model, history, metrics, classes