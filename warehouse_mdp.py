"""Stochastic warehouse MDP solved with tabular value iteration.

The planner is advisory: extracted routes are checked for legal adjacent moves,
and the orchestrator retains A* as a safe fallback. Coordinates are (x, y).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

Position = tuple[int, int]


@dataclass(frozen=True)
class MDPPlan:
    path: list[Position]
    iterations: int
    residual: float
    expected_return: float


class WarehouseMDP:
    ACTIONS = {
        0: (0, -1),   # north
        1: (0, 1),    # south
        2: (1, 0),    # east
        3: (-1, 0),   # west
    }
    # Intended direction, then the two perpendicular slip directions.
    OUTCOMES = {
        0: ((0, -1), (1, 0), (-1, 0)),
        1: ((0, 1), (1, 0), (-1, 0)),
        2: ((1, 0), (0, -1), (0, 1)),
        3: ((-1, 0), (0, -1), (0, 1)),
    }

    def __init__(self, environment, intended_probability: float = 0.8,
                 gamma: float = 0.95, step_cost: float = -1.0,
                 goal_reward: float = 20.0, blocked_cost: float = -3.0):
        if not 0.0 < intended_probability <= 1.0:
            raise ValueError("intended_probability must be in (0, 1].")
        if not 0.0 <= gamma < 1.0:
            raise ValueError("gamma must be in [0, 1).")
        self.env = environment
        self.p_intended = intended_probability
        self.p_slip = (1.0 - intended_probability) / 2.0
        self.gamma = gamma
        self.step_cost = step_cost
        self.goal_reward = goal_reward
        self.blocked_cost = blocked_cost

    def states(self) -> Iterable[Position]:
        for y in range(self.env.height):
            for x in range(self.env.width):
                if self.env.is_valid(x, y):
                    yield (x, y)

    def _transition(self, state: Position, delta: tuple[int, int]):
        candidate = (state[0] + delta[0], state[1] + delta[1])
        if not self.env.is_valid(*candidate):
            return state, self.blocked_cost
        return candidate, self.step_cost

    def _action_value(self, state, action, values, goal):
        deltas = self.OUTCOMES[action]
        probabilities = (self.p_intended, self.p_slip, self.p_slip)
        total = 0.0
        for probability, delta in zip(probabilities, deltas):
            next_state, reward = self._transition(state, delta)
            if next_state == goal:
                reward = self.goal_reward
            future = 0.0 if next_state == goal else values[next_state]
            total += probability * (reward + self.gamma * future)
        return total

    def solve(self, goal: Position, theta: float = 1e-7,
              max_iterations: int = 10_000) -> tuple[dict, dict, int, float]:
        if not self.env.is_valid(*goal):
            raise ValueError(f"Goal {goal} is not a valid warehouse cell.")
        states = list(self.states())
        values = {state: 0.0 for state in states}
        values[goal] = 0.0
        residual = float("inf")
        policy = {state: 0 for state in states}
        for iteration in range(1, max_iterations + 1):
            residual = 0.0
            updated = values.copy()
            for state in states:
                if state == goal:
                    continue
                action_values = [self._action_value(state, a, values, goal)
                                 for a in self.ACTIONS]
                updated[state] = max(action_values)
                policy[state] = int(max(range(len(action_values)),
                                        key=action_values.__getitem__))
                residual = max(residual, abs(updated[state] - values[state]))
            values = updated
            if residual < theta:
                break
        # Recompute policy against the converged value function.
        for state in states:
            if state == goal:
                continue
            action_values = [self._action_value(state, a, values, goal)
                             for a in self.ACTIONS]
            policy[state] = int(max(range(len(action_values)),
                                    key=action_values.__getitem__))
        return values, policy, iteration, residual

    def plan(self, start: Position, goal: Position, max_steps: int | None = None) -> MDPPlan:
        if not self.env.is_valid(*start):
            raise ValueError(f"Start {start} is not a valid warehouse cell.")
        values, policy, iterations, residual = self.solve(goal)
        limit = max_steps or self.env.width * self.env.height * 2
        path = [start]
        current = start
        visited = {start}
        for _ in range(limit):
            if current == goal:
                break
            action = policy.get(current)
            if action is None:
                break
            dx, dy = self.ACTIONS[action]
            nxt = (current[0] + dx, current[1] + dy)
            if not self.env.is_valid(*nxt) or nxt in visited:
                break
            path.append(nxt)
            current = nxt
            visited.add(current)
        if current != goal:
            raise RuntimeError("Value-iteration policy did not produce a loop-free route.")
        return MDPPlan(path, iterations, residual, values[start])
