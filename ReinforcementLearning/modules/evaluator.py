import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes

from modules.results import EpisodeStats, TrainingResult, BenchmarkResult


class Evaluator:
    def stability_analysis(self, results: list[TrainingResult]) -> BenchmarkResult:
        final_rewards = [r.mean_reward for r in results]
        return BenchmarkResult(
            n_runs=len(results),
            runs=results,
            mean_reward=float(np.mean(final_rewards)),
            std_reward=float(np.std(final_rewards)),
            min_reward=float(np.min(final_rewards)),
            max_reward=float(np.max(final_rewards)),
        )

    def _rolling_mean(self, values: list[float], window: int = 100) -> np.ndarray:
        return np.convolve(values, np.ones(window) / window, mode="valid")

    def plot_learning_curve(
        self,
        result: TrainingResult,
        window: int = 100,
        ax: matplotlib.axes.Axes | None = None,
    ) -> matplotlib.axes.Axes:
        rewards = [s.total_reward for s in result.episode_stats]
        smoothed = self._rolling_mean(rewards, window)

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 4))

        ax.plot(rewards, alpha=0.25, color="steelblue", label="per-episode reward")
        ax.plot(
            range(window - 1, len(rewards)),
            smoothed,
            color="steelblue",
            label=f"mean (w={window})",
        )
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total reward")
        ax.set_title("Learning curve — Q-Learning")
        ax.legend()
        return ax

    def plot_comparison(
        self,
        ql_stats: list[EpisodeStats],
        random_stats: list[EpisodeStats],
        ax: matplotlib.axes.Axes | None = None,
    ) -> matplotlib.axes.Axes:
        ql_rewards = [s.total_reward for s in ql_stats]
        random_rewards = [s.total_reward for s in random_stats]

        if ax is None:
            _, ax = plt.subplots(figsize=(7, 5))

        ax.boxplot(
            [ql_rewards, random_rewards],
            labels=["Q-Learning", "Random"],
            patch_artist=True,
            boxprops=dict(facecolor="steelblue", alpha=0.6),
        )
        ax.set_ylabel("Total reward")
        ax.set_title("Q-Learning vs random agent")
        return ax

    def plot_stability(
        self,
        benchmark: BenchmarkResult,
        window: int = 100,
        ax: matplotlib.axes.Axes | None = None,
    ) -> matplotlib.axes.Axes:
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 4))

        all_smoothed = np.array([
            self._rolling_mean(
                [s.total_reward for s in r.episode_stats], window
            )
            for r in benchmark.runs
        ])

        mean_curve = all_smoothed.mean(axis=0)
        std_curve = all_smoothed.std(axis=0)
        xs = np.arange(len(mean_curve))

        ax.fill_between(xs, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2, color="steelblue")
        ax.plot(xs, mean_curve, color="steelblue", label=f"mean ± std ({benchmark.n_runs} runs)")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Mean reward")
        ax.set_title("Training stability across runs")
        ax.legend()
        return ax
