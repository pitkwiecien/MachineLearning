import numpy as np
import gymnasium as gym
from tqdm import tqdm
from itertools import product
import pandas as pd

from modules.config import TrainingConfig
from modules.agent import QLearningAgent
from modules.runner import ExperimentRunner
from modules.evaluator import Evaluator
from modules.results import TrainingResult, BenchmarkResult, EpisodeStats
from modules._typing import Environment
from IPython.display import display


class Experiment:
    def __init__(
        self,
        env_id: str,
        config: TrainingConfig | None = None,
        n_runs: int = 10,
        seed: int = 1234,
        use_sarsa: bool = False,
    ) -> None:
        self.env_id = env_id
        self.config = config or TrainingConfig()
        self.n_runs = n_runs
        self.seed = seed
        self.use_sarsa = use_sarsa
        self.evaluator = Evaluator()
        self.benchmark: BenchmarkResult | None = None
        self.random_stats: list[EpisodeStats] | None = None
        self.eval_stats: list[EpisodeStats] | None = None

    def _make_env(self) -> Environment:
        return gym.make(self.env_id)

    def _make_agent(self, env: Environment, rng: np.random.Generator | None = None) -> QLearningAgent:
        return QLearningAgent(
            n_states=int(env.observation_space.n),
            n_actions=int(env.action_space.n),
            config=self.config,
            rng=rng,
            use_sarsa=self.use_sarsa,
        )

    def run(self, eval_episodes: int = 500) -> None:
        training_results: list[TrainingResult] = []

        for run_idx in tqdm(range(self.n_runs)):
            env = self._make_env()
            rng = np.random.default_rng(self.seed + run_idx)
            agent = self._make_agent(env, rng)
            runner = ExperimentRunner(env, agent)
            training_results.append(runner.train())
            env.close()

        self.benchmark = self.evaluator.stability_analysis(training_results)
        self._best_agent = self._prepare_best_agent()
        self.eval_stats, self.random_stats = self._evaluate(eval_episodes)

    def _prepare_best_agent(self) -> QLearningAgent:
        assert self.benchmark is not None
        best_result = max(self.benchmark.runs, key=lambda r: r.mean_reward)
        env = self._make_env()
        agent = self._make_agent(env)
        agent.q_table = best_result.q_table.copy()
        env.close()
        return agent

    def _evaluate(self, n_episodes: int) -> tuple[list[EpisodeStats], list[EpisodeStats]]:
        env = self._make_env()
        runner = ExperimentRunner(env, self._best_agent)
        ql_stats = [runner.run_episode(training=False) for _ in range(n_episodes)]
        random_stats = runner.run_random(n_episodes)
        env.close()
        return ql_stats, random_stats

    def print_summary(self) -> None:
        assert self.benchmark and self.eval_stats and self.random_stats
        algo = "SARSA" if self.use_sarsa else "Q-Learning"
        b = self.benchmark
        print(f"Algorithm: {algo}")
        print(f"Stability ({b.n_runs} runs):  mean={b.mean_reward:.2f}  std={b.std_reward:.2f}  min={b.min_reward:.2f}  max={b.max_reward:.2f}")
        ql_mean = float(np.mean([s.total_reward for s in self.eval_stats]))
        rnd_mean = float(np.mean([s.total_reward for s in self.random_stats]))
        print(f"{algo} eval:     {ql_mean:.2f}")
        print(f"Random agent:     {rnd_mean:.2f}")

    def plot_all(self) -> None:
        assert self.benchmark and self.eval_stats and self.random_stats
        import matplotlib.pyplot as plt
        algo = "SARSA" if self.use_sarsa else "Q-Learning"
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(algo)
        self.evaluator.plot_learning_curve(self.benchmark.runs[0], ax=axes[0], use_sarsa=self.use_sarsa)
        self.evaluator.plot_comparison(self.eval_stats, self.random_stats, ax=axes[1], use_sarsa=self.use_sarsa)
        self.evaluator.plot_stability(self.benchmark, ax=axes[2])
        plt.tight_layout()
        plt.show()

    def save(self, path: str) -> None:
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "Experiment":
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)


def grid_search(
    env_id: str,
    param_grid: dict[str, list[float | int]],
    n_episodes: int = 10_000,
    eval_episodes: int = 500,
    seed: int = 1234,
    save_path: str | None = None,
    use_sarsa: bool = False,
) -> pd.DataFrame:
    keys = list(param_grid.keys())
    combinations = list(product(*param_grid.values()))

    records = []
    best_eval = -np.inf
    best_qtable = None
    best_config = None
    best_result = None
    best_eval_stats = None

    for combo in tqdm(combinations, desc="Grid search"):
        params = dict(zip(keys, combo))
        config = TrainingConfig(**params, n_episodes=n_episodes)

        env = gym.make(env_id)
        rng = np.random.default_rng(seed)
        agent = QLearningAgent(
            n_states=int(env.observation_space.n),
            n_actions=int(env.action_space.n),
            config=config,
            rng=rng,
            use_sarsa=use_sarsa,
        )
        runner = ExperimentRunner(env, agent)
        result = runner.train()

        eval_stats = [runner.run_episode(training=False) for _ in range(eval_episodes)]
        env.close()

        eval_rewards = [s.total_reward for s in eval_stats]
        eval_mean = float(np.mean(eval_rewards))

        records.append({
            **params,
            "eval_mean": round(eval_mean, 2),
            "eval_std": round(float(np.std(eval_rewards)), 2),
            "eval_min": round(float(np.min(eval_rewards)), 2),
            "eval_max": round(float(np.max(eval_rewards)), 2),
            "train_mean": round(result.mean_reward, 2),
        })

        if eval_mean > best_eval:
            best_eval = eval_mean
            best_qtable = agent.q_table.copy()
            best_config = config
            best_result = result
            best_eval_stats = eval_stats

    df = pd.DataFrame(records).sort_values("eval_mean", ascending=False).reset_index(drop=True)

    if save_path is not None:
        df.to_csv(save_path, index=False)

    best_exp = Experiment(env_id, config=best_config, n_runs=1, seed=seed, use_sarsa=use_sarsa)
    env = gym.make(env_id)
    best_agent = QLearningAgent(
        n_states=int(env.observation_space.n),
        n_actions=int(env.action_space.n),
        config=best_config,
        use_sarsa=use_sarsa,
    )
    best_agent.q_table = best_qtable
    env.close()

    best_exp._best_agent = best_agent
    best_exp.benchmark = best_exp.evaluator.stability_analysis([best_result])
    best_exp.eval_stats = best_eval_stats
    _, best_exp.random_stats = best_exp._evaluate(eval_episodes)
    best_exp.print_summary()
    best_exp.plot_all()
    display(df)

    return df
