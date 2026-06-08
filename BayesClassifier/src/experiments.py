from src.dataset_loader import DatasetLoader
from src.validation import Validator
from src.classifier_result import ClassifierResult
from src.naive_bayes_classifier import NaiveBayes


class ExperimentRunner:
    def run(self, ucirepo_id: int = 94) -> None:
        loader = DatasetLoader(ucirepo_id=ucirepo_id)
        features, labels = loader.load_data()
        validator = Validator(n_splits=5, n_repeats=20)
        repeated_results = validator.repeated_experiment(features, labels, ucirepo_id=ucirepo_id)
        cv_results = validator.k_fold_cross_validation(features, labels)

        print("\nREPEATED EXPERIMENT")
        self._print_summary(repeated_results)

        print("\nK-FOLD CV")
        self._print_summary(cv_results)

        X_train, X_test, y_train, y_test = loader.train_test_split(features, labels)

        model = NaiveBayes()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        final_result = ClassifierResult.from_predictions(y_test, y_pred)

        print("\nFINAL MODEL (TEST SET)")
        print(final_result.__dict__)

    def _print_summary(self, results: list[ClassifierResult]) -> None:
        def stats(values: list[float]) -> dict[str, float]:
            import numpy as np
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