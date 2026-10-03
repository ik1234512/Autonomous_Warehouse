import numpy as np

from hmm_environment import WarehouseHMMEnvironment
from hmm_filter import HMMStateEstimator
import hmm_filter
print("HMM FILTER LOADED FROM:", hmm_filter.__file__)

class RobotPerception:
    def __init__(self, warehouse_environment):
        self.warehouse_environment = warehouse_environment

        grid_map = "\n".join(
            "".join("#" if cell == 1 else "." for cell in row)
            for row in warehouse_environment.grid
        )
        self.hmm_env = WarehouseHMMEnvironment(grid_map)

        self.estimators = {}

    def initialize_robot(self, robot):
        estimator = HMMStateEstimator(
            self.hmm_env.num_states,
            self.hmm_env.T,
            self.hmm_env.E,
        )

        # The robot knows its starting position.
        x, y = robot.position
        state = self.hmm_env.coord_to_state(y, x)

        # Initialize belief externally.
        estimator.belief_state = np.zeros(
            estimator.num_states,
            dtype=float
        )
        estimator.belief_state[state] = 1.0

        self.estimators[robot.robot_id] = estimator

        robot.estimated_position = robot.position
        robot.belief = estimator.belief_state.copy()
        robot.last_observation = None

    def get_observation(self, position):
        x, y = position

        # HMM uses (row, column); warehouse uses (x, y).
        state = self.hmm_env.coord_to_state(y, x)

        # Let the HMM's existing emission model generate
        # the noisy sensor reading.
        probabilities = self.hmm_env.E[:, state]

        return int(
            np.random.choice(
                len(probabilities),
                p=probabilities
            )
        )

    def update(self, robot, action):
        estimator = self.estimators[robot.robot_id]

        observation = self.get_observation(robot.position)

        belief = estimator.bayesian_filter_step(
            action,
            observation
        )

        most_likely_state = estimator.get_most_likely_state()
        row, col = self.hmm_env.state_to_coord(most_likely_state)

        # Convert back from HMM (row, column)
        # to warehouse (x, y).
        robot.estimated_position = (col, row)
        robot.belief = belief.copy()
        robot.last_observation = observation

        return observation