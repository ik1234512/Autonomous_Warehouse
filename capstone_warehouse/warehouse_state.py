from typing import Dict

from environment import SurgicalLabEnvironment
from robot import Robot
import robot
from task import InventoryItem, Task
from perception import RobotPerception


class WarehouseState:
    """
    Canonical state of the warehouse simulation.

    The existing SurgicalLabEnvironment remains the source of truth
    for warehouse geometry. This class stores everything on top of it.
    """

    def __init__(self, environment: SurgicalLabEnvironment):
        self.environment = environment

        self.robots: Dict[int, Robot] = {}
        self.inventory: Dict[str, InventoryItem] = {}
        self.tasks: Dict[str, Task] = {}

        self.time_step: int = 0
        self.events: list[dict] = []
        self.perception = RobotPerception(environment)

    # Registration

    def add_robot(self, robot: Robot) -> None:
        if not self.environment.is_valid(*robot.position):
            raise ValueError(
                f"Robot {robot.robot_id} position {robot.position} is invalid."
            )

        if robot.robot_id in self.robots:
            raise ValueError(f"Robot {robot.robot_id} already exists.")

        self.robots[robot.robot_id] = robot
        self.perception.initialize_robot(robot)

    def add_item(self, item: InventoryItem) -> None:
        if not self.environment.is_valid(*item.location):
            raise ValueError(
                f"Item {item.item_id} location {item.location} is invalid."
            )

        if item.item_id in self.inventory:
            raise ValueError(f"Item {item.item_id} already exists.")

        self.inventory[item.item_id] = item

    def add_task(self, task: Task) -> None:
        if task.task_id in self.tasks:
            raise ValueError(f"Task {task.task_id} already exists.")

        if task.item_id not in self.inventory:
            raise ValueError(
                f"Task {task.task_id} references unknown item {task.item_id}."
            )

        self.tasks[task.task_id] = task

    # Events
    def log_event(self, event_type: str, **details) -> None:
        self.events.append(
            {
                "time": self.time_step,
                "type": event_type,
                **details,
            }
        )

    def advance_time(self) -> None:
        self.time_step += 1