"""Small UCB1 bandit for learning among warehouse route/zone strategies.

Each arm can represent a candidate zone, aisle, or route strategy. Call
select() before a trial and update(arm, reward) after its outcome. This module
only ranks choices; hard safety and feasibility checks remain elsewhere.
"""
from __future__ import annotations
import math
import random


class UCB1Bandit:
    def __init__(self, arms, exploration: float = math.sqrt(2.0), seed: int = 0):
        self.arms = list(arms)
        if not self.arms:
            raise ValueError("At least one bandit arm is required.")
        if len(set(self.arms)) != len(self.arms):
            raise ValueError("Bandit arm identifiers must be unique.")
        self.exploration = float(exploration)
        self.counts = {arm: 0 for arm in self.arms}
        self.total_rewards = {arm: 0.0 for arm in self.arms}
        self.rng = random.Random(seed)

    def select(self):
        # Ensure every arm is sampled at least once.
        untried = [arm for arm in self.arms if self.counts[arm] == 0]
        if untried:
            return untried[0]
        total = sum(self.counts.values())
        scores = {
            arm: self.total_rewards[arm] / self.counts[arm]
                 + self.exploration * math.sqrt(math.log(total) / self.counts[arm])
            for arm in self.arms
        }
        best = max(scores.values())
        ties = [arm for arm, score in scores.items() if math.isclose(score, best)]
        return self.rng.choice(ties)

    def update(self, arm, reward: float):
        if arm not in self.counts:
            raise KeyError(f"Unknown bandit arm: {arm}")
        self.counts[arm] += 1
        self.total_rewards[arm] += float(reward)

    def summary(self):
        return {
            arm: {"trials": self.counts[arm],
                  "mean_reward": (self.total_rewards[arm] / self.counts[arm]
                                  if self.counts[arm] else None)}
            for arm in self.arms
        }
