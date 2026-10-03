from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class WarehouseRLState:
    dx: int
    dy: int
    blocked_north: int
    blocked_south: int
    blocked_east: int
    blocked_west: int


class WarehouseRLEnv:
    """
    Warehouse environment for the capstone SARSA controller.

    A* is responsible for global planning.
    SARSA learns local movement decisions.
    """

    ACTIONS = {
        0: (0, -1),   # North
        1: (0, 1),    # South
        2: (1, 0),    # East
        3: (-1, 0),   # West
        4: (0, 0),    # Wait
    }

    def __init__(self, warehouse_environment):
        self.environment = warehouse_environment

        self.position = None
        self.target = None
        self.battery = 100.0
        self.wear = 0.0
        self.distance_map = {}

    def reset(self, position, target, battery=100.0, wear=0.0):
        self.position = position
        self.target = target
        self.battery = battery
        self.wear = wear

        # Precompute true shortest-path distance to the target.
        self.distance_map = self._build_distance_map(target)

        return self.get_state()

    def get_state(self):
        dx = self.target[0] - self.position[0]
        dy = self.target[1] - self.position[1]

        blocked_north = int(
            not self.environment.is_valid(
                self.position[0],
                self.position[1] - 1,
            )
        )

        blocked_south = int(
            not self.environment.is_valid(
                self.position[0],
                self.position[1] + 1,
            )
        )

        blocked_east = int(
            not self.environment.is_valid(
                self.position[0] + 1,
                self.position[1],
            )
        )

        blocked_west = int(
            not self.environment.is_valid(
                self.position[0] - 1,
                self.position[1],
            )
        )

        return WarehouseRLState(
            dx=dx,
            dy=dy,
            blocked_north=blocked_north,
            blocked_south=blocked_south,
            blocked_east=blocked_east,
            blocked_west=blocked_west,
        )

    def step(self, action):
        if action not in self.ACTIONS:
            raise ValueError(f"Invalid action: {action}")

        dx, dy = self.ACTIONS[action]

        old_distance = self.distance_map.get(self.position, float("inf"))

        candidate = (
            self.position[0] + dx,
            self.position[1] + dy,
        )

        # WAIT
        if action == 4:
            reward = -0.5
            self.wear += 0.1
            self.battery -= 0.5

        # Blocked movement
        elif not self.environment.is_valid(*candidate):
            reward = -3.0
            self.wear += 0.2
            self.battery -= 1.0

        # Valid movement
        else:
            self.position = candidate

            new_distance = self.distance_map.get(
                self.position,
                float("inf"),
            )

            # Reward actual progress through the warehouse,
            # not just Manhattan-distance progress.
            distance_progress = old_distance - new_distance

            reward = -0.1 + distance_progress

            self.wear += 0.1
            self.battery -= 1.0

        done = self.position == self.target

        if done:
            reward += 10.0

        if self.battery <= 0:
            self.battery = 0.0
            done = True
            reward -= 10.0

        return self.get_state(), reward, done, {
            "position": self.position,
            "target": self.target,
            "battery": self.battery,
            "wear": self.wear,
            "distance_to_target": self.distance_map.get(
                self.position,
                float("inf"),
            ),
        }

    def _build_distance_map(self, target):
        """
        Compute shortest-path distance from every reachable cell
        to the target using BFS.
        """

        distances = {target: 0}
        queue = deque([target])

        moves = [
            (0, -1),
            (0, 1),
            (1, 0),
            (-1, 0),
        ]

        while queue:
            current = queue.popleft()
            current_distance = distances[current]

            for dx, dy in moves:
                neighbor = (
                    current[0] + dx,
                    current[1] + dy,
                )

                if neighbor in distances:
                    continue

                if not self.environment.is_valid(*neighbor):
                    continue

                distances[neighbor] = current_distance + 1
                queue.append(neighbor)

        return distances