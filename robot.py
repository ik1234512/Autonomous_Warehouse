from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Robot:
    robot_id: int
    position: tuple[int, int]
    capacity: float = 10.0
    battery: float = 100.0
    max_battery: float = 100.0
    wear: float = 0.0

    status: str = "IDLE"
    current_task: Optional[str] = None
    carried_item: Optional[str] = None
    route: list[tuple[int, int]] = field(default_factory=list)

    # HMM perception state
    estimated_position: Optional[tuple[int, int]] = None
    belief: Optional[object] = None
    last_observation: Optional[int] = None

    def can_carry(self, weight: float) -> bool:
        return weight <= self.capacity

    def set_task(self, task_id: str) -> None:
        self.current_task = task_id
        self.status = "ASSIGNED"

    def start_task(self) -> None:
        self.status = "MOVING"

    def wait(self) -> None:
        self.status = "WAITING"

    def charge(self, amount: float = 100.0) -> None:
        self.battery = min(self.max_battery, self.battery + amount)
        self.status = "IDLE"

    def drain_battery(self, amount: float) -> None:
        self.battery = max(0.0, self.battery - amount)

    def move_to(
        self,
        position: tuple[int, int],
        movement_cost: float = 1.0,
    ) -> None:
        self.position = position
        self.drain_battery(movement_cost)

    def pick_item(self, item_id: str) -> None:
        self.carried_item = item_id

    def drop_item(self) -> Optional[str]:
        item_id = self.carried_item
        self.carried_item = None
        return item_id

    def set_route(self, route: list[tuple[int, int]]) -> None:
        self.route = list(route)

    def has_next_step(self) -> bool:
        return bool(self.route)

    def next_step(self) -> tuple[int, int]:
        if not self.route:
            raise RuntimeError("Robot has no route.")

        if self.route[0] == self.position:
            self.route.pop(0)

        if not self.route:
            raise RuntimeError(
                "Robot is already at the destination."
            )

        return self.route.pop(0)