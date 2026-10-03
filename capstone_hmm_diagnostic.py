"""Small diagnostic for checking the HMM state during warehouse moves."""

from __future__ import annotations

from environment import SurgicalLabEnvironment
from robot import Robot
from task import Task, InventoryItem
from warehouse_state import WarehouseState
from orchestrator import WarehouseOrchestrator


def print_hmm(robot: Robot, label: str) -> None:
    belief = robot.belief
    print(
        f"{label}: true={robot.position} "
        f"estimated={robot.estimated_position} "
        f"map_prob={(float(belief.max()) if belief is not None else None):.4f} "
        f"belief_sum={(float(belief.sum()) if belief is not None else None):.4f} "
        f"obs={robot.last_observation}"
    )


def run_leg(orchestrator: WarehouseOrchestrator, robot_id: int, target):
    robot = orchestrator.state.robots[robot_id]
    before = robot.position
    orchestrator.train_sarsa_for_target(
        robot_id=robot_id,
        target=target,
        episodes=2000,
    )

    step = 0
    print(f"\n--- LEG: {before} -> {target} ---")
    print_hmm(robot, "START")

    while robot.position != target:
        old = robot.position
        ok = orchestrator.execute_sarsa_move(
            robot_id=robot_id,
            target=target,
        )
        if not ok:
            print("STOP: execute_sarsa_move returned False")
            break
        step += 1
        print_hmm(robot, f"STEP {step} move {old} -> {robot.position}")

    return robot.position == target


def main() -> None:
    env = SurgicalLabEnvironment(map_type="custom")
    state = WarehouseState(env)
    orchestrator = WarehouseOrchestrator(state)

    robot = Robot(
        robot_id=1,
        position=(4, 4),
        capacity=10.0,
        battery=100.0,
    )
    state.add_robot(robot)

    # Reset HMM estimator explicitly so this diagnostic starts clean.
    state.perception.initialize_robot(robot)

    item = InventoryItem(
        item_id="ITEM-001",
        weight=6.0,
        location=(5, 4),
    )
    state.add_item(item)

    task = Task(
        task_id="TASK-001",
        item_id="ITEM-001",
        pickup=(5, 4),
        destination=(5, 5),
        priority="URGENT",
        deadline=20,
    )
    state.add_task(task)
    orchestrator.assign_task("TASK-001", 1)

    print("=== HMM INTEGRATION DIAGNOSTIC ===")
    print_hmm(robot, "INITIAL")

    pickup_ok = run_leg(orchestrator, 1, item.location)
    if not pickup_ok:
        raise RuntimeError("Pickup leg did not reach the target.")

    orchestrator.pick_item(1, item.item_id)

    delivery_ok = run_leg(orchestrator, 1, task.destination)
    if not delivery_ok:
        raise RuntimeError("Delivery leg did not reach the target.")

    orchestrator.deliver_item(1)

    print("\n=== SUMMARY ===")
    print("True final position:", robot.position)
    print("Estimated final position:", robot.estimated_position)
    print("Final MAP probability:", robot.belief.max())
    print("Final belief sum:", robot.belief.sum())
    print("Last observation:", robot.last_observation)
    print("Item status:", item.status)
    print("Task status:", task.status)


if __name__ == "__main__":
    main()
