from dataclasses import dataclass
import numpy.typing as npt
import numpy as np


@dataclass
class EpisodeStats:
    total_reward: float
    steps: int
    terminated: bool
    truncated: bool


@dataclass
class TrainingResult:
    episode_stats: list[EpisodeStats]
    q_table: npt.NDArray[np.int_]
    mean_reward: float
    std_reward: float
    min_reward: float
    max_reward: float


@dataclass
class BenchmarkResult:
    n_runs: int
    runs: list[TrainingResult]
    mean_reward: float
    std_reward: float
    min_reward: float
    max_reward: float