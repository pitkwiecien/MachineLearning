import numpy as np
import gymnasium as gym

from modules._typing import Environment
from modules.agent import QLearningAgent
from modules.results import EpisodeStats, TrainingResult


class ExperimentRunner:
    def __init__(self, env: Environment, agent: QLearningAgent) -> None:
        self.env = env
        self.agent = agent

    def run_episode(self, *, training: bool = True) -> EpisodeStats:
        state, _ = self.env.reset()
        total_reward = 0.0
        terminated = False
        truncated = False

        step = 0
        for step in range(self.agent.config.max_steps):
            action = self.agent.select_action(state, explore=training)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            reward = float(reward)
            total_reward += reward

            if training:
                self.agent.update(state, action, reward, next_state, terminated, truncated)

            state = next_state
            if terminated or truncated:
                break

        if training:
            self.agent.decay_epsilon()

        return EpisodeStats(
            total_reward=total_reward,
            steps=step+1,
            terminated=terminated,
            truncated=truncated,
        )

    def train(self) -> TrainingResult:
        episode_stats: list[EpisodeStats] = [
            self.run_episode(training=True)
            for _ in range(self.agent.config.n_episodes)
        ]
        rewards = [s.total_reward for s in episode_stats]
        return TrainingResult(
            episode_stats=episode_stats,
            q_table=self.agent.q_table.copy(),
            mean_reward=float(np.mean(rewards)),
            std_reward=float(np.std(rewards)),
            min_reward=float(np.min(rewards)),
            max_reward=float(np.max(rewards)),
        )

    def run_random(self, n_episodes: int) -> list[EpisodeStats]:
        stats: list[EpisodeStats] = []
        for _ in range(n_episodes):
            state, _ = self.env.reset()
            total_reward = 0.0
            terminated = False
            truncated = False

            step = 0
            for step in range(self.agent.config.max_steps):
                action = int(self.env.action_space.sample())
                state, reward, terminated, truncated, _ = self.env.step(action)
                total_reward += float(reward)
                if terminated or truncated:
                    break
            stats.append(EpisodeStats(
                total_reward=total_reward,
                steps=step + 1,
                terminated=terminated,
                truncated=truncated,
            ))
        return stats