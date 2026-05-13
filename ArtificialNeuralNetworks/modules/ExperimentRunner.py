import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import product
from IPython.display import display

from .simple_training import TrainingConfig, train_validate_test


class ExperimentRunner:

    def __init__(self):
        self.results = []

    def _architectures(self):

        return [
            [32],
            [64],
            [128],
            [256],

            [32, 32],
            [64, 64],
            [128, 64],
            [256, 128],

            [64, 32],
            [128, 64, 32],
            [64, 32, 16],
            [128, 64, 32, 16],

            [64, 64, 32],
            [32, 64, 32],
        ]

    def _configs(self):

        architectures = self._architectures()

        activations = [
            "relu",
            "tanh",
            "sigmoid"
        ]

        learning_rates = [
            0.0005,
            0.001,
            0.003,
            0.01
        ]

        batch_sizes = [
            16,
            32,
            64,
            128
        ]

        losses = [
            "cross_entropy",
            "mse"
        ]

        epochs = [
            100,
            200,
            400
        ]

        configs = []

        for arch, act, lr, bs, loss, ep in product(
            architectures,
            activations,
            learning_rates,
            batch_sizes,
            losses,
            epochs
        ):

            configs.append(
                TrainingConfig(
                    hidden_sizes=arch,
                    hidden_activation=act,
                    output_activation="linear",
                    loss_name=loss,
                    learning_rate=lr,
                    batch_size=bs,
                    epochs=ep,
                    random_state=42
                )
            )

        return configs

    def run(self, data_path, top_k=10, sample_size=None):

        configs = self._configs()

        # Obsługa sample_size
        if sample_size is not None:
            random.seed(42)
            configs = random.sample(configs, min(sample_size, len(configs)))

        print(f"Experiments: {len(configs)}")

        for cfg in tqdm(configs, desc="Training"):

            try:
                _, history, metrics, _ = train_validate_test(data_path, cfg)

                self.results.append({
                    "architecture": cfg.hidden_sizes,
                    "activation": cfg.hidden_activation,
                    "loss": cfg.loss_name,
                    "lr": cfg.learning_rate,
                    "batch_size": cfg.batch_size,
                    "epochs": cfg.epochs,
                    "accuracy": metrics.accuracy,
                    "f1": metrics.f1,
                    "precision": metrics.precision,
                    "recall": metrics.recall,
                    "train_loss": history["train_loss"][-1], # Dodana kolumna
                    "val_loss": history["val_loss"][-1],
                })

            except Exception as e:
                print(f"[ERROR] {e}")

        return self._show_results(top_k)

    def _show_results(self, top_k):

        df = pd.DataFrame(self.results)

        df = df.sort_values(
            by=["accuracy", "f1", "val_loss"],
            ascending=[False, False, True]
        ).reset_index(drop=True)

        cols = ["accuracy", "f1", "precision", "recall", "train_loss", "val_loss", "lr"]
        df[cols] = df[cols].round(4)

        print("\nTOP MODELS:\n")
        display(df.head(top_k))

        print("\nFULL RESULTS (sorted):\n")
        display(df)

        best = df.iloc[0]

        print("\nBEST MODEL:\n")
        print(best.to_string())

        return df