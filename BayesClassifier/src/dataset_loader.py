import numpy as np
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo
from src.typing import FeatureMatrix, LabelVector


class DatasetLoader:
    def __init__(
        self,
        ucirepo_id: int,
        test_ratio: float = 0.2,
        random_seed: int = 42,
    ) -> None:
        self.ucirepo_id: int = ucirepo_id
        self.test_ratio: float = test_ratio
        self.random_seed: int = random_seed

    def load_data(self) -> tuple[FeatureMatrix, LabelVector]:
        spambase = fetch_ucirepo(id=self.ucirepo_id)
        features: FeatureMatrix = np.asarray(spambase.data.features, dtype=np.float64)
        targets: LabelVector = np.asarray(spambase.data.targets).ravel().astype(np.int64)

        return features, targets

    def train_test_split(
        self,
        features: FeatureMatrix,
        labels: LabelVector,
    ) -> tuple[FeatureMatrix, FeatureMatrix, LabelVector, LabelVector]:
        X_train, X_test, y_train, y_test = train_test_split(
            features,
            labels,
            test_size=self.test_ratio,
            random_state=self.random_seed,
            stratify=labels,
        )

        return (
            X_train,
            X_test,
            y_train,
            y_test,
        )