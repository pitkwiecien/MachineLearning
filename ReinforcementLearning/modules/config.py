from dataclasses import dataclass


@dataclass
class TrainingConfig:
    alpha: float = 0.1
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    n_episodes: int = 10_000
    max_steps: int = 200
