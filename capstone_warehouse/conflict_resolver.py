from dataclasses import dataclass
from math import inf


@dataclass(frozen=True)
class Move:
    action: int
    position: tuple[int, int]


class ConflictResolver:
    """Resolve head-to-head robot conflicts with a small minimax search."""

    ACTIONS = {
        0: (0, -1),   # North
        1: (0, 1),    # South
        2: (1, 0),    # East
        3: (-1, 0),   # West
        4: (0, 0),    # Wait
    }

    def __init__(self, warehouse_environment, depth=2):
        self.environment = warehouse_environment
        self.depth = depth

    def valid_moves(self, position):
        moves = []

        for action, (dx, dy) in self.ACTIONS.items():
            candidate = (
                position[0] + dx,
                position[1] + dy,
            )

            if action == 4 or self.environment.is_valid(*candidate):
                moves.append(
                    Move(
                        action=action,
                        position=candidate if action != 4 else position,
                    )
                )

        return moves

    def resolve(
        self,
        robot_position,
        robot_target,
        other_position,
        other_target=None,
    ):
        """Pick the safest move for the active robot against the other robot's best response."""

        best_value = -inf
        best_move = None
        alpha = -inf
        beta = inf

        for move in self.valid_moves(robot_position):
            value = self._min_value(
                max_position=move.position,
                min_position=other_position,
                max_target=robot_target,
                min_target=other_target or other_position,
                depth=self.depth - 1,
                alpha=alpha,
                beta=beta,
                max_previous=robot_position,
                min_previous=other_position,
            )

            if value > best_value:
                best_value = value
                best_move = move

            alpha = max(alpha, best_value)

        if best_move is None:
            return Move(action=4, position=robot_position)

        return best_move

    def _max_value(
        self,
        max_position,
        min_position,
        max_target,
        min_target,
        depth,
        alpha,
        beta,
        max_previous,
        min_previous,
    ):
        if depth <= 0:
            return self._utility(
                max_position,
                min_position,
                max_target,
                min_target,
                max_previous,
                min_previous,
            )

        value = -inf

        for move in self.valid_moves(max_position):
            value = max(
                value,
                self._min_value(
                    max_position=move.position,
                    min_position=min_position,
                    max_target=max_target,
                    min_target=min_target,
                    depth=depth - 1,
                    alpha=alpha,
                    beta=beta,
                    max_previous=max_previous,
                    min_previous=min_previous,
                ),
            )

            if value >= beta:
                break

            alpha = max(alpha, value)

        return value

    def _min_value(
        self,
        max_position,
        min_position,
        max_target,
        min_target,
        depth,
        alpha,
        beta,
        max_previous,
        min_previous,
    ):
        if depth <= 0:
            return self._utility(
                max_position,
                min_position,
                max_target,
                min_target,
                max_previous,
                min_previous,
            )

        value = inf

        for move in self.valid_moves(min_position):
            value = min(
                value,
                self._max_value(
                    max_position=max_position,
                    min_position=move.position,
                    max_target=max_target,
                    min_target=min_target,
                    depth=depth - 1,
                    alpha=alpha,
                    beta=beta,
                    max_previous=max_previous,
                    min_previous=min_previous,
                ),
            )

            if value <= alpha:
                break

            beta = min(beta, value)

        return value

    @staticmethod
    def _distance(position, target):
        return abs(position[0] - target[0]) + abs(position[1] - target[1])

    def _utility(
        self,
        max_position,
        min_position,
        max_target,
        min_target,
        max_previous,
        min_previous,
    ):
        """Higher values are better for the active robot.

        The score rewards progress toward the target and penalizes collisions,
        head-on swaps, proximity, and waiting.
        """

        max_progress = (
            self._distance(max_previous, max_target)
            - self._distance(max_position, max_target)
        )

        utility = float(max_progress)

        # Same-cell collision.
        if max_position == min_position:
            utility -= 100.0

        # Head-on swap: A and B exchange cells in one time step.
        if (
            max_previous == min_position
            and min_previous == max_position
            and max_position != min_position
        ):
            utility -= 100.0

        # Being adjacent to the competing robot is mildly risky.
        proximity = self._distance(max_position, min_position)

        if proximity == 1:
            utility -= 2.0

        # Waiting is represented by no position change.
        if max_position == max_previous:
            utility -= 1.0

        # Give a small preference to keeping the competing robot
        # farther from its own target; this acts as a weak blocking
        # pressure term for the MIN model.
        if min_target is not None:
            utility += 0.05 * self._distance(
                min_position,
                min_target,
            )

        return utility
