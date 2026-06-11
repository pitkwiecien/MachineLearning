from sklearn.model_selection import StratifiedKFold

from src.naive_bayes_classifier import NaiveBayes
from src.classifier_result import ClassifierResult
from src.typing import FeatureMatrix, LabelVector
from src.dataset_loader import DatasetLoader


class Validator:
    def __init__(
        self,
        n_splits: int = 5,
        n_repeats: int = 20,
        random_seed: int = 42,
        variance_smoothing: float = 1e-9
        ) -> None:
        self.variance_smoothing = variance_smoothing
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_seed = random_seed

    def repeated_experiment(
        self,
        features: FeatureMatrix,
        labels: LabelVector,
        ucirepo_id: int,
        test_ratio: float = 0.2,
    ) -> list[ClassifierResult]:

        loader = DatasetLoader(
            ucirepo_id=ucirepo_id,
            test_ratio=test_ratio,
            random_seed=self.random_seed,
        )

        results: list[ClassifierResult] = []

        for i in range(self.n_repeats):
            seed = self.random_seed + i
            loader.random_seed = seed
            X_train, X_test, y_train, y_test = loader.train_test_split(features, labels)
            model = NaiveBayes(variance_smoothing=self.variance_smoothing)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            results.append(ClassifierResult.from_predictions(y_test, y_pred))

        return results

    def k_fold_cross_validation(
        self,
        features: FeatureMatrix,
        labels: LabelVector,
    ) -> list[ClassifierResult]:

        kf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_seed,
        )

        results: list[ClassifierResult] = []
        for train_index, test_index in kf.split(features, labels):

            X_train, X_test = features[train_index], features[test_index]
            y_train, y_test = labels[train_index], labels[test_index]

            model = NaiveBayes(variance_smoothing=self.variance_smoothing)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            results.append(ClassifierResult.from_predictions(y_test, y_pred))

        return results
