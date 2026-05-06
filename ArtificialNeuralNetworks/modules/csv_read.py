
import numpy as np
import numpy.typing as npt
from sklearn.model_selection import train_test_split


FloatArray = npt.NDArray[np.float64]
DEFAULT_PATH = "./data/data_scaled.csv"


def load_csv(path: str = DEFAULT_PATH) -> FloatArray:
    return np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)


def load_xy(path: str = DEFAULT_PATH) -> tuple[FloatArray, FloatArray]:
    data = load_csv(path)
    return data[:, :-1], data[:, -1]


def load_train_validation_test_split(
    path: str = DEFAULT_PATH,
    val_size: float = 0.2,
    test_size: float = 0.2,
    random_state: int | None = 42,
    shuffle: bool = True,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    X, y = load_xy(path)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
    )

    val_relative_size = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_relative_size,
        random_state=random_state,
        shuffle=shuffle,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
