"""
rl_utils.py
============
Small shared utilities for the SARSA / Q-learning solutions and labs.
Provided — students don't need to modify this.
"""

import random
import numpy as np
from packbot_env import N_STATES, N_ACTIONS, state_index


def epsilon_greedy(Q, state, epsilon, rng):
    """state is (item_type, wear) tuple."""
    s = state_index(*state)
    if rng.random() < epsilon:
        return rng.randrange(N_ACTIONS)
    return int(np.argmax(Q[s]))


def greedy_action(Q, state):
    s = state_index(*state)
    return int(np.argmax(Q[s]))


def linear_epsilon_schedule(episode, total_episodes, eps_start=1.0, eps_end=0.05):
    frac = min(1.0, episode / max(1, total_episodes * 0.8))
    return eps_start + frac * (eps_end - eps_start)


def wear_threshold(Q, item_type, wear_max=10):
    """
    The smallest wear level at which the greedy policy switches from
    FastGrasp (0) to CarefulGrasp (1), for a given item_type.
    Returns wear_max + 1 if it never switches (always FastGrasp).
    """
    for wear in range(wear_max + 1):
        s = state_index(item_type, wear)
        if int(np.argmax(Q[s])) == 1:
            return wear
    return wear_max + 1


def init_Q():
    return np.zeros((N_STATES, N_ACTIONS))


def fast_advantage_curve(Q, item_type=0, wear_max=10):
    """
    Q(wear, FastGrasp) - Q(wear, CarefulGrasp), for each wear level.
    This is the main comparison metric for the lab: a large negative value
    near max wear means the algorithm has learned to be conservative
    (build a safety margin) as it approaches the breakdown risk; a value
    near zero or positive means it is riding wear right up to the edge.
    """
    curve = []
    for wear in range(wear_max + 1):
        s = state_index(item_type, wear)
        curve.append(Q[s, 0] - Q[s, 1])
    return curve
