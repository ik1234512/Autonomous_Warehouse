from dataclasses import dataclass
from typing import Optional


@dataclass
class InventoryItem:
    item_id: str
    weight: float
    location: tuple[int, int]
    status: str = "AVAILABLE"


@dataclass
class Task:
    task_id: str
    item_id: str
    pickup: tuple[int, int]
    destination: tuple[int, int]
    priority: str = "NORMAL"
    deadline: Optional[int] = None

    assigned_robot: Optional[int] = None
    status: str = "PENDING"

    def assign(self, robot_id: int) -> None:
        self.assigned_robot = robot_id
        self.status = "ASSIGNED"

    def mark_in_progress(self) -> None:
        self.status = "IN_PROGRESS"

    def mark_completed(self) -> None:
        self.status = "COMPLETED"

    def mark_failed(self) -> None:
        self.status = "FAILED"
