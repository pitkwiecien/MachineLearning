import numpy as np
from sklearn.model_selection import train_test_split

DEFAULT_PATH = "./data/data_scaled.csv"


def load_csv(path=DEFAULT_PATH):
    return np.loadtxt(path, delimiter=",", skiprows=1)


def load_xy(path=DEFAULT_PATH):
    data = load_csv(path)
    return data[:, :-1], data[:, -1]


def load_train_validation_test_split(
    path=DEFAULT_PATH,
    val_size=0.2,
    test_size=0.2,
    random_state=42,
    shuffle=True,
):

    X, y = load_xy(path)

    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle
    )

    val_ratio = val_size / (1 - test_size)

    X_train, X_val, y_train, y_val = train_test_split(
        X_tv,
        y_tv,
        test_size=val_ratio,
        random_state=random_state,
        shuffle=shuffle
    )

    return X_train, X_val, X_test, y_train, y_val, y_test