"""Off-policy Q-learning controller for the same navigation environment as SARSA.

This is deliberately separate from qlearning_lab.py: that lab models gripper
wear and grasp choice, whereas this controller models warehouse navigation.
"""
from __future__ import annotations
import random
import numpy as np
from rl_utils import linear_epsilon_schedule
from warehouse_rl_env import WarehouseRLEnv


class WarehouseQLearning:
    def __init__(self, warehouse_environment, alpha=0.1, gamma=0.95, seed=0):
        self.env = WarehouseRLEnv(warehouse_environment)
        self.alpha = alpha
        self.gamma = gamma
        self.rng = random.Random(seed)
        self.q_table = {}
        self.epsilon = 0.0

    @staticmethod
    def _key(state):
        return (state.dx, state.dy, state.blocked_north, state.blocked_south,
                state.blocked_east, state.blocked_west)

    def _q(self, state):
        key = self._key(state)
        if key not in self.q_table:
            self.q_table[key] = np.zeros(len(self.env.ACTIONS), dtype=float)
        return self.q_table[key]

    def choose_action(self, state):
        values = self._q(state)
        if self.rng.random() < self.epsilon:
            return self.rng.randrange(len(values))
        return int(np.argmax(values))

    def train(self, start_position, target, battery=100.0, wear=0.0,
              episodes=1000, eps_start=1.0, eps_end=0.05, max_steps=50):
        returns, successes = [], 0
        for episode in range(episodes):
            state = self.env.reset(start_position, target, battery, wear)
            self.epsilon = linear_epsilon_schedule(episode, episodes, eps_start, eps_end)
            total, done = 0.0, False
            for step in range(max_steps):
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                timed_out = not done and step == max_steps - 1
                if timed_out:
                    reward -= 10.0
                q = self._q(state)
                target_value = reward if (done or timed_out) else reward + self.gamma * float(np.max(self._q(next_state)))
                q[action] += self.alpha * (target_value - q[action])
                total += reward
                state = next_state
                if done or timed_out:
                    break
            successes += int(done)
            returns.append(total)
        self.epsilon = 0.0
        return returns, successes
