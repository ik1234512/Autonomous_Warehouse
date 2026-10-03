"""
sarsa_lab.py
=============
YOUR TASK: implement SARSA control on the PackBot environment.

Run test_correctness.py first -- it checks sarsa_update() against a
hand-computed example before you spend any time on the full training loop.

Do not import anything from sarsa_solution.py.
"""

import random
import numpy as np
from packbot_env import PackBotEnv, state_index
from rl_utils import epsilon_greedy, linear_epsilon_schedule, init_Q, wear_threshold


def sarsa_update(Q, s, a, r, s_next, a_next, alpha, gamma):
    """
    TODO: implement the SARSA update rule.

    Inputs:
        Q       : the (N_STATES, N_ACTIONS) value table
        s, a    : the state index and action just taken
        r       : the reward received
        s_next  : the resulting state index, or None if the episode ended
        a_next  : the SAMPLED next action (from the same behavior policy
                  you used to pick `a`), or None if the episode ended
        alpha, gamma : learning rate and discount factor

    Returns:
        The NEW value that Q[s, a] should be updated to. Do not mutate Q
        inside this function -- just return the new scalar.

    Reminder from lecture: SARSA bootstraps using the VALUE OF THE ACTION
    THAT WAS ACTUALLY SAMPLED next (a_next) -- not the best available
    action at s_next. That is the entire difference between this file
    and qlearning_lab.py.

    Special case: if s_next is None (terminal transition), there is
    nothing to bootstrap from -- the target is just r.
    """
    if s_next is None:
        target = r
    else:
        target = r + gamma * Q[s_next, a_next]
    
    new_value = Q[s, a] + alpha * (target - Q[s, a])
    return new_value


def train_sarsa(num_episodes=4000, alpha=0.1, gamma=0.95, seed=0,
                 eps_start=1.0, eps_end=0.2, log_every=50):
    """
    TODO: implement the SARSA training loop using sarsa_update() above.

    Structure to follow (this is the standard SARSA control loop from
    lecture -- note that, unlike Q-learning, you pick action A' for the
    NEXT state *before* you loop back around, because you need it both
    to act next step AND to bootstrap this step's update):

        Initialize S
        Choose A from S using epsilon-greedy(Q)
        Repeat until S is terminal:
            Take action A, observe R, S'
            If S' is not terminal:
                Choose A' from S' using epsilon-greedy(Q)
                Q(S, A) <- sarsa_update(Q, S, A, R, S', A', alpha, gamma)
            Else:
                A' <- None
                Q(S, A) <- sarsa_update(Q, S, A, R, None, None, alpha, gamma)
            S, A <- S', A'

    Return (Q, episode_returns, threshold_log) to match sarsa_solution.py's
    signature -- compare_policies.py and test scripts expect this shape.
    threshold_log should be a list of (episode, {item_type: wear_threshold(...)})
    tuples, logged every `log_every` episodes.
    """
    env = PackBotEnv(seed=seed)
    rng = random.Random(seed)
    Q = init_Q()

    episode_returns = []
    threshold_log = []

    for episode in range(num_episodes):
        epsilon = linear_epsilon_schedule(episode, num_episodes, eps_start, eps_end)
        
        state = env.reset()
        action = epsilon_greedy(Q, state, epsilon, rng)
        
        episode_return = 0
        done = False
        
        while not done:
            next_state, reward, done, info = env.step(action)
            episode_return += reward
            
            if not done:
                a_next = epsilon_greedy(Q, next_state, epsilon, rng)
            else:
                a_next = None
            
            Q[state, action] = sarsa_update(Q, state, action, reward, 
                                             next_state if not done else None, 
                                             a_next, alpha, gamma)
            
            state = next_state
            action = a_next
        
        episode_returns.append(episode_return)
        
        if (episode + 1) % log_every == 0:
            thresholds = {}
            for item_type in range(3):
                thresholds[item_type] = wear_threshold(Q, item_type)
            threshold_log.append((episode + 1, thresholds))

    return Q, episode_returns, threshold_log


if __name__ == "__main__":
    Q, returns, thresholds = train_sarsa()
    print("Final wear thresholds by item type:", thresholds[-1][1])
    print("Mean return, last 200 episodes:", np.mean(returns[-200:]))
