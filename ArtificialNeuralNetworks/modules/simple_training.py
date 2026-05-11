from dataclasses import dataclass
from typing import Callable, Any

import numpy as np
import numpy.typing as npt

from .ActivationFunction import ActivationFunction
from .MLP import MLP
from .cross_entropy import grad_softmax_cross_entropy_logits, softmax_cross_entropy_with_logits
from .csv_read import load_train_validation_test_split
from .mse import mse, mse_derivative
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix


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


def train_model(
    path: str,
    config: TrainingConfig,
) -> tuple[MLP, dict[str, list[float]], np.ndarray]:
    """Train a model using train/validation splits and return the trained model,
    training history, class labels, and test scores.
    """
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

    history: dict[str, list[float] | Any] = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_f1": None,
        "val_precision": None,
        "val_recall": None,
        "val_confusion_matrix": None,
    }

    rng = np.random.default_rng(config.random_state)

    for _ in range(config.epochs):
        batch_losses: list[float] = []
        for X_batch, y_batch in _minibatches(X_train, y_train_encoded, config.batch_size, config.shuffle, rng):
            batch_losses.append(model.train_step(X_batch, y_batch, loss_function, loss_derivative, config.learning_rate))

        train_loss, train_accuracy = evaluate_model(model, X_train, y_train_encoded, config.loss_name, config.batch_size)
        val_loss, val_accuracy = evaluate_model(model, X_val, y_val_encoded, config.loss_name, config.batch_size)
        test_scores = test_model(model, X_test=X_test, y_test=y_test, config=config)

        history["train_loss"].append(float(np.mean(batch_losses)))
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        history["val_f1"] = test_scores["test_f1"]
        history["val_precision"] = test_scores["test_precision"]
        history["val_recall"] = test_scores["test_recall"]
        history["val_confusion_matrix"] = test_scores["test_confusion_matrix"]

    # Evaluate on test set


    return model, history, classes


def test_model(
    model: MLP,
    path: str | None = None,
    config: TrainingConfig | None = None,
    X_test: FloatArray | None = None,
    y_test: FloatArray | None = None,
) -> dict[str, Any]:
    """Evaluate a trained `model` on a test set.

    Provide either `path` and `config` (to load the dataset splits) or pass
    `X_test` and `y_test` arrays directly. Returns a metrics dict.
    """
    if path is not None and config is not None:
        _, _, X_test_loaded, _, _, y_test_loaded = load_train_validation_test_split(
            path=path,
            val_size=config.val_size,
            test_size=config.test_size,
            random_state=config.random_state,
            shuffle=config.shuffle,
        )
        X_test = X_test_loaded
        y_test = y_test_loaded

    if X_test is None or y_test is None:
        raise ValueError("Provide either (path and config) or (X_test and y_test)")

    # Prepare integer labels
    y_test_int = np.asarray(y_test, dtype=int)

    # Run model on test set in batches and collect logits/predictions
    batch_size = config.batch_size if config is not None else 32
    effective_batch_size = int(X_test.shape[0]) if batch_size is None else int(batch_size)
    preds_list: list[np.ndarray] = []
    losses: list[float] = []
    loss_function, _ = _loss_pair(config.loss_name if config is not None else "cross_entropy")

    for start in range(0, X_test.shape[0], effective_batch_size):
        sl = slice(start, start + effective_batch_size)
        X_batch = X_test[sl].T
        y_batch_int = y_test_int[sl]
        # one-hot for loss computation
        num_classes = model.layers[-1].output_size
        y_batch_onehot = np.eye(num_classes, dtype=np.float64)[y_batch_int].T
        y_pred = model.forward(X_batch)
        preds_list.append(y_pred)
        losses.append(loss_function(y_pred, y_batch_onehot))

    predicted_matrix = np.concatenate(preds_list, axis=1)
    predicted_labels = np.argmax(predicted_matrix, axis=0)

    test_loss = float(np.mean(losses))
    test_accuracy = float(accuracy_score(y_test_int, predicted_labels))
    test_f1 = float(f1_score(y_test_int, predicted_labels, average="macro", zero_division=0))
    test_precision = float(precision_score(y_test_int, predicted_labels, average="macro", zero_division=0))
    test_recall = float(recall_score(y_test_int, predicted_labels, average="macro", zero_division=0))
    test_confusion = confusion_matrix(y_test_int, predicted_labels)

    return {
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "test_f1": test_f1,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "test_confusion_matrix": test_confusion,
    }