import numpy as np

from modules.config import TrainingConfig


class QLearningAgent:
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        config: TrainingConfig,
        rng: np.random.Generator | None = None
    ) -> None:
        self.n_states = n_states
        self.n_actions = n_actions
        self.config = config
        self.epsilon = config.epsilon_start
        self._rng = rng or np.random.default_rng()
        self.q_table: np.ndarray = np.zeros((n_states, n_actions))

    def select_action(self, state: int, *, explore: bool = True) -> int:
        if explore and self._rng.random() < self.epsilon:
            return int(self._rng.integers(self.n_actions))
        return int(np.argmax(self.q_table[state]))

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        terminated: bool,
        truncated: bool,
    ) -> None:
        if terminated:
            target = reward
        else:
            target = reward + self.config.gamma * float(np.max(self.q_table[next_state]))

        td_error = target - self.q_table[state, action]
        self.q_table[state, action] += self.config.alpha * td_error

    def decay_epsilon(self) -> None:
        self.epsilon = max(
            self.config.epsilon_end,
            self.epsilon * self.config.epsilon_decay,
        )

    def reset_epsilon(self) -> None:
        self.epsilon = self.config.epsilon_start
