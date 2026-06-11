from src.dataset_loader import DatasetLoader
from src.validation import Validator
from src.classifier_result import ClassifierResult
from src.naive_bayes_classifier import NaiveBayes

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score


class ExperimentRunner:

    def run(
        self,
        ucirepo_id: int = 94,
        variance_smoothing: float = 1e-9,
    ) -> None:

        loader = DatasetLoader(
            ucirepo_id=ucirepo_id
        )

        features, labels = loader.load_data()

        validator = Validator(
            n_splits=5,
            n_repeats=20,
            variance_smoothing=variance_smoothing
        )

        repeated_results = validator.repeated_experiment(
            features,
            labels,
            ucirepo_id=ucirepo_id
        )

        cv_results = validator.k_fold_cross_validation(
            features,
            labels
        )

        print("\nREPEATED EXPERIMENT")
        self._print_summary(repeated_results)

        print("\nK-FOLD CV")
        self._print_summary(cv_results)

        X_train, X_test, y_train, y_test = (
            loader.train_test_split(
                features,
                labels
            )
        )

        model = NaiveBayes(
            variance_smoothing=variance_smoothing
        )

        model.fit(
            X_train,
            y_train
        )

        y_pred = model.predict(X_test)

        final_result = ClassifierResult.from_predictions(
            y_test,
            y_pred
        )

        print("\nFINAL MODEL (TEST SET)")
        print(final_result.__dict__)

    def grid_search_variance_smoothing(
        self,
        ucirepo_id: int = 94,
    ) -> pd.DataFrame:

        loader = DatasetLoader(
            ucirepo_id=ucirepo_id
        )

        features, labels = loader.load_data()

        smoothing_values = [
            1e-12,
            1e-10,
            1e-8,
            1e-6,
            1e-4,
            1e-2
        ]

        rows = []

        for smooth in smoothing_values:

            validator = Validator(
                n_splits=5,
                n_repeats=20,
                variance_smoothing=smooth
            )

            results = validator.k_fold_cross_validation(
                features,
                labels
            )

            rows.append({
                "variance_smoothing": smooth,
                "accuracy": np.mean(
                    [r.accuracy for r in results]
                ),
                "precision": np.mean(
                    [r.precision for r in results]
                ),
                "f1": np.mean(
                    [r.f1 for r in results]
                ),
                "recall": np.mean(
                    [r.recall for r in results]
                )
            })

        return pd.DataFrame(rows)

    def plot_grid_search(
        self,
        grid_df: pd.DataFrame,
    ) -> None:

        plt.figure(figsize=(8, 5))

        plt.plot(
            grid_df["variance_smoothing"],
            grid_df["accuracy"],
            marker="o",
            label="Accuracy"
        )

        plt.plot(
            grid_df["variance_smoothing"],
            grid_df["precision"],
            marker="s",
            label="precision"
        )

        plt.plot(
            grid_df["variance_smoothing"],
            grid_df["f1"],
            marker="s",
            label="F1"
        )

        plt.plot(
            grid_df["variance_smoothing"],
            grid_df["recall"],
            marker="s",
            label="recall"
        )

        plt.xscale("log")

        plt.xlabel("variance_smoothing")
        plt.ylabel("score")

        plt.title("Grid Search")

        plt.legend()
        plt.grid()

        plt.show()

    def repeated_metrics_dataframe(
        self,
        ucirepo_id: int = 94,
    ) -> pd.DataFrame:
        loader = DatasetLoader(ucirepo_id=ucirepo_id)
        features, labels = loader.load_data()
        validator = Validator(
            n_splits=5,
            n_repeats=20
        )

        results = validator.repeated_experiment(
            features,
            labels,
            ucirepo_id=ucirepo_id
        )

        return pd.DataFrame({
            "accuracy": [r.accuracy for r in results],
            "precision": [r.precision for r in results],
            "recall": [r.recall for r in results],
            "f1": [r.f1 for r in results],
        })

    def plot_accuracy_histogram(
        self,
        metrics_df: pd.DataFrame,
    ) -> None:
        plt.figure(figsize=(8, 5))
        plt.hist(metrics_df["accuracy"], bins=10)
        plt.title("Accuracy Distribution")
        plt.xlabel("Accuracy")
        plt.ylabel("Count")
        plt.show()

    def plot_metrics_boxplot(
        self,
        metrics_df: pd.DataFrame,
    ) -> None:
        plt.figure(figsize=(8, 5))
        metrics_df.boxplot()
        plt.title("Metric Distribution")
        plt.show()

    def compare_with_sklearn(self, ucirepo_id: int = 94) -> pd.DataFrame:
        loader = DatasetLoader(
            ucirepo_id=ucirepo_id
        )

        features, labels = loader.load_data()
        X_train, X_test, y_train, y_test = (
            loader.train_test_split(
                features,
                labels
            )
        )

        custom_model = NaiveBayes()
        custom_model.fit(
            X_train,
            y_train
        )
        custom_pred = custom_model.predict(
            X_test
        )

        sklearn_model = GaussianNB()
        sklearn_model.fit(
            X_train,
            y_train
        )

        sklearn_pred = sklearn_model.predict(X_test)

        return pd.DataFrame({
            "model": [
                "Custom NB",
                "Sklearn NB"
            ],
            "accuracy": [
                accuracy_score(y_test, custom_pred),
                accuracy_score(y_test, sklearn_pred)
            ],
            "precision": [
                precision_score(y_test, custom_pred),
                precision_score(y_test, sklearn_pred)
            ],
            "f1": [
                f1_score(y_test, custom_pred),
                f1_score(y_test, sklearn_pred)
            ],
            "recall": [
                recall_score(y_test, custom_pred),
                recall_score(y_test, sklearn_pred)
            ]
        })

    def plot_model_comparison(
        self,
        comparison_df: pd.DataFrame,
    ) -> None:
        comparison_df.plot(
            x="model",
            y=["accuracy", "precision", "f1", "recall"],
            kind="bar",
            figsize=(8, 5)
        )
        plt.title("Custom vs Sklearn")
        plt.ylabel("Score")
        plt.show()

    def _print_summary(
        self,
        results: list[ClassifierResult]
    ) -> None:
        def stats(values):
            arr = np.array(values)

            return {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
            }

        print("accuracy:", stats([r.accuracy for r in results]))
        print("precision:", stats([r.precision for r in results]))
        print("recall:", stats([r.recall for r in results]))
        print("f1:", stats([r.f1 for r in results]))