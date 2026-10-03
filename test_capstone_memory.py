"""API-level tests for the capstone SQLite memory layer."""
from __future__ import annotations

import tempfile
from pathlib import Path

from capstone_memory import WarehouseMemory


class Robot:
    def __init__(self, robot_id, position, estimated_position, battery, wear, status, current_task, carried_item):
        self.robot_id = robot_id
        self.position = position
        self.estimated_position = estimated_position
        self.battery = battery
        self.wear = wear
        self.status = status
        self.current_task = current_task
        self.carried_item = carried_item


class Task:
    def __init__(self):
        self.task_id = "TASK-001"
        self.item_id = "ITEM-001"
        self.status = "COMPLETED"
        self.priority = "URGENT"
        self.deadline = 20
        self.assigned_robot_id = 1


class Item:
    def __init__(self):
        self.item_id = "ITEM-001"
        self.weight = 6.0
        self.location = (5, 5)
        self.status = "DELIVERED"


class State:
    def __init__(self):
        self.robots = {
            1: Robot(1, (5, 5), (5, 5), 96, 0.4, "IDLE", None, None)
        }
        self.tasks = {"TASK-001": Task()}
        self.inventory = {"ITEM-001": Item()}
        self.logged = []

    def log_event(self, event_type, **kwargs):
        self.logged.append((event_type, kwargs))


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "memory.db"
        memory = WarehouseMemory(db)
        state = State()
        run_id = memory.start_run("memory-test")
        memory.attach_to_state_log(state, run_id)

        state.log_event("ITEM_DELIVERED", robot_id=1, item_id="ITEM-001")

        assert memory.count_events(run_id) == 1
        assert state.logged[0][0] == "ITEM_DELIVERED"
        assert memory.recent_events(run_id, 1)[0]["event_type"] == "ITEM_DELIVERED"
        assert memory.robot_history(run_id, 1)[-1]["x"] == 5
        assert memory.task_history(run_id, "TASK-001")[-1]["status"] == "COMPLETED"
        memory.close()

    print("All capstone SQLite memory tests passed.")


if __name__ == "__main__":
    main()
