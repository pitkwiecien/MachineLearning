import numpy as np
from dataclasses import dataclass

from .MLP import MLP
from .ActivationFunction import ActivationFunction
from .mse import mse, mse_derivative
from .cross_entropy import softmax_cross_entropy_with_logits, grad_softmax_cross_entropy_logits
from .csv_read import load_train_validation_test_split
from .metrics import compute_classification_metrics, ClassificationMetrics


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


ACTIVATIONS = {
    "relu": ActivationFunction.RELU,
    "sigmoid": ActivationFunction.SIGMOID,
    "tanh": ActivationFunction.TANH,
    "linear": ActivationFunction.LINEAR,
    "softmax": ActivationFunction.SOFTMAX,
}


def _activation(name: str):
    return ACTIVATIONS[name.lower()]


def _loss_pair(name: str):

    name = name.lower()

    if name == "mse":
        return mse, mse_derivative

    if name in {"cross_entropy", "ce"}:
        return softmax_cross_entropy_with_logits, grad_softmax_cross_entropy_logits

    raise ValueError(f"Unknown loss: {name}")


def build_model(input_size: int, output_size: int, config: TrainingConfig):

    hidden = [_activation(config.hidden_activation)] * len(config.hidden_sizes)

    output_act = (
        ActivationFunction.LINEAR
        if config.loss_name in {"cross_entropy", "ce"}
        else _activation(config.output_activation)
    )

    return MLP(
        [input_size] + config.hidden_sizes + [output_size],
        hidden + [output_act],
        round_output=config.round_output
    )


def minibatches(X, y, batch_size, shuffle=True):

    idx = np.arange(len(X))
    if shuffle:
        np.random.shuffle(idx)

    batch_size = len(X) if batch_size is None else batch_size

    for i in range(0, len(X), batch_size):
        b = idx[i:i + batch_size]
        yield X[b], y[b]


def train_validate_test(path: str, config: TrainingConfig):

    X_train, X_val, X_test, y_train, y_val, y_test = load_train_validation_test_split(
        path,
        config.val_size,
        config.test_size,
        config.random_state,
        config.shuffle,
    )

    model = build_model(
        X_train.shape[1],
        len(np.unique(y_train)),
        config
    )

    loss_fn, loss_grad_fn = _loss_pair(config.loss_name)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
    }

    for _ in range(config.epochs):

        losses = []

        for Xb, yb in minibatches(
            X_train,
            y_train,
            config.batch_size,
            config.shuffle
        ):
            losses.append(
                model.train_step(Xb, yb, loss_fn, loss_grad_fn, config.learning_rate)
            )

        train_pred = model.forward(X_train)
        val_pred = model.forward(X_val)

        train_loss = loss_fn(train_pred, y_train)
        val_loss = loss_fn(val_pred, y_val)

        train_acc = np.mean(np.argmax(train_pred, axis=1) == y_train.astype(int))
        val_acc = np.mean(np.argmax(val_pred, axis=1) == y_val.astype(int))

        history["train_loss"].append(float(np.mean(losses)))
        history["val_loss"].append(float(val_loss))
        history["train_accuracy"].append(float(train_acc))
        history["val_accuracy"].append(float(val_acc))

    test_pred = model.forward(X_test)

    metrics: ClassificationMetrics = compute_classification_metrics(
        y_test,
        test_pred
    )

    return model, history, metrics, np.unique(y_train)