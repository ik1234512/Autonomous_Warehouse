import random

import numpy as np

from rl_utils import linear_epsilon_schedule
from sarsa_lab import sarsa_update
from warehouse_rl_env import WarehouseRLEnv


class WarehouseSARSA:
    def __init__(
        self,
        warehouse_environment,
        alpha=0.1,
        gamma=0.95,
        epsilon=0.2,
    ):
        self.env = WarehouseRLEnv(warehouse_environment)

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # 5 actions:
        # North, South, East, West, Wait
        self.num_actions = len(self.env.ACTIONS)

        # State -> Q values.
        self.q_table = {}

        self.rng = random.Random(0)

    def _state_key(self, state):
        return (
            state.dx,
            state.dy,
            state.blocked_north,
            state.blocked_south,
            state.blocked_east,
            state.blocked_west,
        )

    def _get_q_values(self, state):
        key = self._state_key(state)

        if key not in self.q_table:
            self.q_table[key] = np.zeros(
                self.num_actions,
                dtype=float,
            )

        return self.q_table[key]

    def choose_action(self, state):
        q_values = self._get_q_values(state)

        if self.rng.random() < self.epsilon:
            return self.rng.randrange(self.num_actions)

        return int(np.argmax(q_values))

    def train(
        self,
        start_position,
        target,
        battery=100.0,
        wear=0.0,
        episodes=2000,
        eps_start=1.0,
        eps_end=0.05,
        max_steps=50,
    ):
        episode_returns = []
        successful_episodes = 0

        for episode in range(episodes):
            state = self.env.reset(
                position=start_position,
                target=target,
                battery=battery,
                wear=wear,
            )

            self.epsilon = linear_epsilon_schedule(
                episode,
                episodes,
                eps_start,
                eps_end,
            )

            action = self.choose_action(state)

            total_reward = 0.0
            done = False
            steps = 0

            while not done and steps < max_steps:
                next_state, reward, done, _ = self.env.step(action)
                steps += 1

                # Treat hitting max_steps as a terminal timeout for learning.
                timed_out = not done and steps >= max_steps
                transition_done = done or timed_out

                if timed_out:
                    reward -= 10.0

                total_reward += reward

                q_values = self._get_q_values(state)

                if transition_done:
                    next_action = None
                    new_q = sarsa_update(
                        q_values.reshape(1, -1),
                        0,
                        action,
                        reward,
                        None,
                        None,
                        self.alpha,
                        self.gamma,
                    )
                else:
                    next_action = self.choose_action(next_state)
                    next_q_values = self._get_q_values(next_state)

                    combined = np.vstack(
                        [q_values, next_q_values]
                    )

                    new_q = sarsa_update(
                        combined,
                        0,
                        action,
                        reward,
                        1,
                        next_action,
                        self.alpha,
                        self.gamma,
                    )

                q_values[action] = new_q

                state = next_state

                if not transition_done:
                    action = next_action

                if timed_out:
                    break

            if done:
                successful_episodes += 1

            episode_returns.append(total_reward)

        return episode_returns, successful_episodes


if __name__ == "__main__":
    from environment import SurgicalLabEnvironment
    from planner import ForkliftPlanner

    env = SurgicalLabEnvironment(map_type="custom")

    print("Target valid:", env.is_valid(8, 7))

    planner = ForkliftPlanner(env)

    path, expanded, planning_time = planner.a_star(
        (4, 4),
        (8, 7),
        "manhattan",
    )

    print("A* path:", path)
    print("A* path length:", len(path))

    controller = WarehouseSARSA(env)

    returns, successes = controller.train(
        start_position=(4, 4),
        target=(8, 7),
        episodes=2000,
    )

    print("Episodes:", len(returns))
    print("Successful episodes:", successes)
    print("Success rate:", successes / len(returns))
    print("Last 10 returns:", returns[-10:])
    print("Mean return:", np.mean(returns))

    # --------------------------------------------------
    # Greedy policy test
    # --------------------------------------------------

    controller.epsilon = 0.0

    state = controller.env.reset(
        position=(4, 4),
        target=(8, 7),
    )

    greedy_path = [controller.env.position]
    done = False

    for _ in range(50):
        action = controller.choose_action(state)

        state, reward, done, info = controller.env.step(action)

        greedy_path.append(controller.env.position)

        if done:
            break

    print("\n=== GREEDY POLICY TEST ===")
    print("Path:", greedy_path)
    print("Reached target:", done)
    print("Steps:", len(greedy_path) - 1)
    print("Final position:", controller.env.position)