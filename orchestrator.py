from warehouse_state import WarehouseState
from robot import Robot
from task import Task
from planner import ForkliftPlanner
from warehouse_sarsa import WarehouseSARSA
from warehouse_mdp import WarehouseMDP
from warehouse_qlearning import WarehouseQLearning
from route_bandit import UCB1Bandit
from conflict_resolver import ConflictResolver
import os

class WarehouseOrchestrator:
    def __init__(self, state):
        self.state = state
        self.planner = ForkliftPlanner(state.environment)

        self.sarsa = WarehouseSARSA(state.environment)
        self.sarsa_target = None
        self.qlearning = None
        self.qlearning_target = None
        self.mdp = WarehouseMDP(state.environment)
        self.route_bandit = UCB1Bandit(["astar", "value_iteration"], seed=0)
        self.route_strategy = os.getenv("WAREHOUSE_PLANNER", "astar").strip().lower()

        self.conflict_resolver = ConflictResolver(
            state.environment,
            depth=2,
        )

    def assign_task(self, task_id: str, robot_id: int) -> None:
        if task_id not in self.state.tasks:
            raise ValueError(f"Unknown task: {task_id}")

        if robot_id not in self.state.robots:
            raise ValueError(f"Unknown robot: {robot_id}")

        task = self.state.tasks[task_id]
        robot = self.state.robots[robot_id]
        item = self.state.inventory[task.item_id]

        if not robot.can_carry(item.weight):
            raise ValueError(
                f"Robot {robot_id} cannot carry item {item.item_id} "
                f"({item.weight}kg > {robot.capacity}kg capacity)."
            )

        task.assign(robot_id)
        robot.set_task(task_id)

        self.state.log_event(
            "TASK_ASSIGNED",
            task_id=task_id,
            robot_id=robot_id,
        )

    def move_robot(
            self,
            robot_id: int,
            position: tuple[int, int],
            battery_cost: float = 1.0,
        ) -> None:

            robot = self.state.robots[robot_id]

            if not self.state.environment.is_valid(*position):
                raise ValueError(
                    f"Cannot move robot to blocked cell {position}."
                )

            old_position = robot.position

            # Move the actual robot in the simulator.
            robot.move_to(position, battery_cost)

            # Determine which cardinal action was taken.
            dx = position[0] - old_position[0]
            dy = position[1] - old_position[1]

            action_map = {
                (0, -1): 0,   # North
                (0, 1): 1,    # South
                (1, 0): 2,    # East
                (-1, 0): 3,   # West
            }

            action = action_map.get((dx, dy))

            if action is not None:
                observation = self.state.perception.update(
                    robot,
                    action,
                )
                estimated_position = robot.estimated_position
            else:
                observation = None
                estimated_position = robot.estimated_position

            self.state.log_event(
                "ROBOT_MOVED",
                robot_id=robot_id,
                position=robot.position,
                estimated_position=estimated_position,
                observation=observation,
                battery=robot.battery,
            )

            self.state.advance_time()

    def pick_item(self, robot_id: int, item_id: str) -> None:
        robot = self.state.robots[robot_id]
        item = self.state.inventory[item_id]

        if robot.position != item.location:
            raise ValueError(
                f"Robot {robot_id} is at {robot.position}, "
                f"but item {item_id} is at {item.location}."
            )

        if not robot.can_carry(item.weight):
            raise ValueError(
                f"Robot {robot_id} cannot carry item {item_id}."
            )

        robot.pick_item(item_id)
        item.status = "IN_TRANSIT"

        if robot.current_task:
            self.state.tasks[robot.current_task].mark_in_progress()

        self.state.log_event(
            "ITEM_PICKED",
            robot_id=robot_id,
            item_id=item_id,
        )

    def deliver_item(self, robot_id: int) -> None:
        robot = self.state.robots[robot_id]

        if robot.carried_item is None:
            raise ValueError(f"Robot {robot_id} is not carrying an item.")

        item = self.state.inventory[robot.carried_item]

        if robot.current_task is None:
            raise ValueError(f"Robot {robot_id} has no active task.")

        task = self.state.tasks[robot.current_task]

        if robot.position != task.destination:
            raise ValueError(
                f"Robot {robot_id} is at {robot.position}, "
                f"but destination is {task.destination}."
            )

        item.location = robot.position
        item.status = "DELIVERED"

        delivered_item = robot.drop_item()
        task.mark_completed()

        robot.status = "IDLE"
        robot.current_task = None

        self.state.log_event(
            "ITEM_DELIVERED",
            robot_id=robot_id,
            item_id=delivered_item,
            task_id=task.task_id,
        )

        self.state.advance_time()
    def plan_route(
        self,
        robot_id: int,
        destination: tuple[int, int],
        strategy: str | None = None,
    ) -> list[tuple[int, int]]:
        """Plan with A*, value iteration, or a UCB-selected strategy.

        A* remains the default. Value-iteration routes are checked for valid
        adjacent steps; if policy extraction loops or fails, A* takes over.
        """
        robot = self.state.robots[robot_id]
        selected = (strategy or self.route_strategy).lower()
        if selected not in {"astar", "value_iteration", "bandit"}:
            raise ValueError("strategy must be astar, value_iteration, or bandit")

        bandit_arm = None
        if selected == "bandit":
            bandit_arm = self.route_bandit.select()
            selected = bandit_arm

        expanded, elapsed = 0, 0.0
        fallback_reason = None
        if selected == "value_iteration":
            try:
                mdp_plan = self.mdp.plan(robot.position, destination)
                route = mdp_plan.path
                # Defensive route validation before handing it to execution.
                for a, b in zip(route, route[1:]):
                    if (abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1
                            or not self.state.environment.is_valid(*b)):
                        raise RuntimeError("MDP produced an invalid transition")
                expanded = mdp_plan.iterations
                elapsed = 0.0
                value_estimate = mdp_plan.expected_return
            except (RuntimeError, ValueError) as exc:
                fallback_reason = str(exc)
                route, expanded, elapsed = self.planner.a_star(
                    robot.position, destination, "manhattan")
                selected = "astar_fallback"
                value_estimate = None
        else:
            route, expanded, elapsed = self.planner.a_star(
                robot.position, destination, "manhattan")
            value_estimate = None

        if not route:
            raise RuntimeError(
                f"No route found for Robot {robot_id} "
                f"from {robot.position} to {destination}."
            )

        bandit_reward = None
        if bandit_arm is not None:
            # A simple online objective: prefer successful shorter routes.
            bandit_reward = 1.0 / max(1, len(route) - 1)
            if selected == "astar_fallback":
                bandit_reward *= 0.25
            self.route_bandit.update(bandit_arm, bandit_reward)

        robot.set_route(route)
        self.state.log_event(
            "ROUTE_PLANNED",
            robot_id=robot_id,
            destination=destination,
            route_length=len(route),
            planner=selected,
            bandit_arm=bandit_arm,
            bandit_reward=bandit_reward,
            value_estimate=value_estimate,
            fallback_reason=fallback_reason,
            nodes_expanded=expanded,
            planning_time_ms=elapsed * 1000,
        )
        return route

    def execute_next_move(
        self,
        robot_id: int,
        use_sarsa: bool = False,
        conflict_robot_id: int | None = None,
        conflict_target: tuple[int, int] | None = None,
    ) -> bool:
        robot = self.state.robots[robot_id]

        if use_sarsa and self.sarsa_target is not None:
            if robot.position == self.sarsa_target:
                return False

            return self.execute_sarsa_move(
                robot_id,
                self.sarsa_target,
                conflict_robot_id=conflict_robot_id,
                conflict_target=conflict_target,
            )

        if not robot.has_next_step():
            return False

        next_position = robot.next_step()

        self.move_robot(
            robot_id,
            next_position,
            battery_cost=1.0,
        )

        return True

    def train_sarsa_for_target(
        self,
        robot_id: int,
        target: tuple[int, int],
        episodes: int = 2000,
    ) -> None:
        robot = self.state.robots[robot_id]

        self.sarsa = WarehouseSARSA(self.state.environment)

        configured_episodes = int(os.getenv("SARSA_EPISODES", episodes))
        self.sarsa.train(
            start_position=robot.position,
            target=target,
            battery=robot.battery,
            wear=robot.wear,
            episodes=configured_episodes,
        )

        # Training is over; execution should be greedy.
        self.sarsa.epsilon = 0.0
        self.sarsa_target = target

    def train_qlearning_for_target(
        self,
        robot_id: int,
        target: tuple[int, int],
        episodes: int = 1000,
    ) -> None:
        """Train an off-policy navigation controller for comparison with SARSA."""
        robot = self.state.robots[robot_id]
        self.qlearning = WarehouseQLearning(self.state.environment)
        configured_episodes = int(os.getenv("QLEARNING_EPISODES", episodes))
        self.qlearning.train(
            start_position=robot.position,
            target=target,
            battery=robot.battery,
            wear=robot.wear,
            episodes=configured_episodes,
        )
        self.qlearning.epsilon = 0.0
        self.qlearning_target = target
        self.state.log_event(
            "QLEARNING_TRAINED",
            robot_id=robot_id,
            target=target,
            episodes=configured_episodes,
            learned_states=len(self.qlearning.q_table),
        )

    def execute_qlearning_move(
        self,
        robot_id: int,
        target: tuple[int, int],
        battery_cost: float = 1.0,
    ) -> bool:
        """Execute one Q-learning proposal, constrained to the current A* route."""
        if self.qlearning is None or self.qlearning_target != target:
            raise RuntimeError("Train Q-learning for this target before execution.")
        robot = self.state.robots[robot_id]
        state = self.qlearning.env.reset(
            position=robot.position, target=target,
            battery=robot.battery, wear=robot.wear)
        self.qlearning.epsilon = 0.0
        action = self.qlearning.choose_action(state)
        dx, dy = self.qlearning.env.ACTIONS[action]
        candidate = (robot.position[0] + dx, robot.position[1] + dy)

        # The global route is still the authority: do not let Q-learning
        # introduce shortcuts, reversals, or moves into reserved/blocked cells.
        try:
            current_index = robot.route.index(robot.position)
            expected_next = robot.route[current_index + 1]
        except (ValueError, IndexError):
            expected_next = None
        if (candidate != expected_next
                or not self.state.environment.is_valid(*candidate)):
            self.state.log_event(
                "QLEARNING_MOVE_REJECTED", robot_id=robot_id,
                proposed_position=candidate, reason="OUTSIDE_ASTAR_ROUTE")
            return False
        self.move_robot(robot_id, candidate, battery_cost=battery_cost)
        self.state.log_event(
            "QLEARNING_MOVE_ACCEPTED", robot_id=robot_id,
            action=action, position=candidate)
        return True

    def execute_sarsa_move(
        self,
        robot_id: int,
        target: tuple[int, int],
        battery_cost: float = 1.0,
        conflict_robot_id: int | None = None,
        conflict_target: tuple[int, int] | None = None,
    ) -> bool:
        """
        Let SARSA propose the next move, then run that proposal through the
        Minimax/Alpha-Beta conflict resolver when another robot creates an
        immediate collision risk.
        """
        robot = self.state.robots[robot_id]

        # Keep the RL environment synchronized with the real robot.
        state = self.sarsa.env.reset(
            position=robot.position,
            target=target,
            battery=robot.battery,
            wear=robot.wear,
        )

        # Training is complete, so execution is greedy.
        self.sarsa.epsilon = 0.0
        proposed_action = self.sarsa.choose_action(state)

        action_map = {
            0: (0, -1),  # North
            1: (0, 1),   # South
            2: (1, 0),   # East
            3: (-1, 0),  # West
            4: (0, 0),   # Wait
        }

        dx, dy = action_map[proposed_action]
        proposed_position = (
            robot.position[0] + dx,
            robot.position[1] + dy,
        )

        # Never allow SARSA to command a blocked cell.
        if not self.state.environment.is_valid(*proposed_position):
            return False

        final_action = proposed_action
        final_position = proposed_position
        conflict_intervened = False

        # Only invoke adversarial search when another robot can create an
        # immediate collision/congestion risk.
        if conflict_robot_id is not None:
            if conflict_robot_id not in self.state.robots:
                raise ValueError(
                    f"Unknown conflict robot: {conflict_robot_id}"
                )

            other_robot = self.state.robots[conflict_robot_id]

            manhattan_distance = (
                abs(robot.position[0] - other_robot.position[0])
                + abs(robot.position[1] - other_robot.position[1])
            )

            conflict_risk = (
                proposed_position == other_robot.position
                or manhattan_distance <= 1
            )

            if conflict_risk:
                other_goal = (
                    conflict_target
                    if conflict_target is not None
                    else other_robot.position
                )

                decision = self.conflict_resolver.resolve(
                    robot_position=robot.position,
                    robot_target=target,
                    other_position=other_robot.position,
                    other_target=other_goal,
                )

                final_action = decision.action
                final_position = decision.position
                conflict_intervened = (
                    final_position != proposed_position
                    or final_action != proposed_action
                )

                if conflict_intervened:
                    self.state.log_event(
                        "CONFLICT_RESOLVED",
                        robot_id=robot_id,
                        other_robot_id=conflict_robot_id,
                        proposed_action=proposed_action,
                        proposed_position=proposed_position,
                        selected_action=final_action,
                        selected_position=final_position,
                    )

        # Minimax may choose WAIT instead of entering a conflict state.
        if final_position == robot.position:
            self.state.log_event(
                "ROBOT_WAITED",
                robot_id=robot_id,
                reason="CONFLICT"
                if conflict_intervened
                else "SARSA_WAIT",
            )
            self.state.log_event(
                "SARSA_ACTION",
                robot_id=robot_id,
                action=final_action,
                proposed_action=proposed_action,
                position=robot.position,
                target=target,
                controller=(
                    "SARSA+MINIMAX"
                    if conflict_intervened
                    else "SARSA"
                ),
            )
            self.state.advance_time()
            return True

        if not self.state.environment.is_valid(*final_position):
            return False

        # All approved movement still flows through move_robot(), preserving
        # battery handling, HMM perception updates, logging, and simulation
        # time advancement.
        self.move_robot(
            robot_id,
            final_position,
            battery_cost=battery_cost,
        )

        self.state.log_event(
            "SARSA_ACTION",
            robot_id=robot_id,
            action=final_action,
            proposed_action=proposed_action,
            position=robot.position,
            target=target,
            controller=(
                "SARSA+MINIMAX"
                if conflict_intervened
                else "SARSA"
            ),
        )

        return True

    def resolve_conflict(
        self,
        robot_id: int,
        other_robot_id: int,
        robot_target: tuple[int, int],
        other_target: tuple[int, int] | None = None,
    ) -> tuple[int, int]:
        robot = self.state.robots[robot_id]
        other_robot = self.state.robots[other_robot_id]

        move = self.conflict_resolver.resolve(
            robot_position=robot.position,
            robot_target=robot_target,
            other_position=other_robot.position,
            other_target=(
                other_target
                if other_target is not None
                else other_robot.position
            ),
        )

        self.state.log_event(
            "CONFLICT_RESOLVED",
            robot_id=robot_id,
            other_robot_id=other_robot_id,
            action=move.action,
            selected_position=move.position,
        )

        return move.position

    def execute_conflict_aware_move(
        self,
        robot_id: int,
        other_robot_id: int,
        robot_target: tuple[int, int],
        other_target: tuple[int, int],
    ) -> bool:

        robot = self.state.robots[robot_id]

        if robot.position == robot_target:
            return False

        next_position = self.resolve_conflict(
            robot_id=robot_id,
            other_robot_id=other_robot_id,
            robot_target=robot_target,
            other_target=other_target,
        )

        if next_position == robot.position:
            self.state.log_event(
                "ROBOT_WAITED",
                robot_id=robot_id,
                reason="CONFLICT",
            )
            self.state.advance_time()
            return True

        if not self.state.environment.is_valid(*next_position):
            return False

        self.move_robot(
            robot_id,
            next_position,
            battery_cost=1.0,
        )

        return True