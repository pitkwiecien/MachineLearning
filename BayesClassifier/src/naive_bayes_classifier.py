from src.typing import FeatureMatrix, FeatureVector, LabelVector
import numpy as np


class NaiveBayes:
    def __init__(self) -> None:
        self.classes: LabelVector | None = None
        self.class_priors: dict[int, float] = {}
        self.feature_means: dict[int, FeatureVector] = {}
        self.feature_variances: dict[int, FeatureVector] = {}

    def fit(
        self,
        features: FeatureMatrix,
        labels: LabelVector,
    ) -> None:
        self.classes = np.unique(labels)

        for current_class in self.classes:
            class_samples: FeatureMatrix = features[labels == current_class]

            self.class_priors[int(current_class)] = (len(class_samples) / len(features))
            self.feature_means[int(current_class)] = np.mean(class_samples, axis=0)
            self.feature_variances[int(current_class)] = np.var(class_samples, axis=0) + 1e-9

    def predict(
        self,
        features: FeatureMatrix,
    ) -> LabelVector:
        predictions: list[int] = [self._predict_single(sample) for sample in features]
        return np.asarray(predictions, dtype=np.int64)

    def _predict_single(
        self,
        sample: FeatureVector,
    ) -> int:
        if self.classes is None:
            raise RuntimeError("The classifier must be trained before making predictions")

        class_scores: list[float] = []
        for current_class in self.classes:
            class_label: int = int(current_class)
            log_prior: float = np.log(self.class_priors[class_label])

            log_likelihood = log_likelihood = np.sum(
                - 0.5 * np.log(2 * np.pi * self.feature_variances[class_label])
                - ((sample - self.feature_means[class_label]) ** 2)
                / (2 * self.feature_variances[class_label]))
            posterior_probability = log_prior + log_likelihood
            class_scores.append(posterior_probability)

        predicted_index: int = int(np.argmax(class_scores))
        return int(self.classes[predicted_index])
