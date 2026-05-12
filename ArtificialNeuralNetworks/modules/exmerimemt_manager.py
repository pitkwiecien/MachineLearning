import pandas as pd
import numpy as np

class ExperimentManager:
    def __init__(self):
        self.results = []

    def add_result(self, config, metrics, history):
        res = {
            "hidden_sizes": str(config.hidden_sizes),
            "hidden_activation": config.hidden_activation,
            "loss_name": config.loss_name,
            "learning_rate": config.learning_rate,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "accuracy": metrics.accuracy,
            "f1": metrics.f1,
            "final_loss": history["val_loss"][-1]
        }
        self.results.append(res)

    def get_best_param(self, param_name) -> str | None:
        if not self.results:
            return None
        best_run = max(self.results, key=lambda x: x["f1"])
        return best_run[param_name]

    def to_markdown(self):
        df = pd.DataFrame(self.results)
        df = df.round(3)
        top_5_f1 = sorted(list(set(df["f1"].values)), reverse=True)[:5]
        def highlight_f1(val):
            if val in top_5_f1 :
                return f"*** {val:.3f} ***"
            return f"{val:.3f}"
        display_df = df.copy()
        display_df["f1"] = df["f1"].apply(highlight_f1)
        return display_df.to_markdown(index=False)