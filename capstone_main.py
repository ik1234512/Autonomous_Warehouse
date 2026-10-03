import os
"""End-to-end capstone demo.

This script wires the warehouse state, dispatch, route planning, SARSA,
and HMM perception together in one run.
"""

from environment import SurgicalLabEnvironment
from robot import Robot
from task import Task, InventoryItem
from warehouse_state import WarehouseState
from orchestrator import WarehouseOrchestrator

from capstone_dispatch import dispatch_verified_plan
from capstone_memory import WarehouseMemory


def main() -> None:
    env = SurgicalLabEnvironment(map_type="custom")
    state = WarehouseState(env)
    orchestrator = WarehouseOrchestrator(state)

    memory = WarehouseMemory("warehouse_memory.db")
    run_id = memory.start_run("capstone-end-to-end")
    memory.attach_to_state_log(state, run_id)
    memory.snapshot_state(state, run_id=run_id)

    # Robots
    robot = Robot(
        robot_id=1,
        position=(4, 4),
        capacity=10.0,
        battery=100.0,
    )
    state.add_robot(robot)

    robot2 = Robot(
        robot_id=2,
        position=(4, 2),
        capacity=10.0,
        battery=100.0,
    )
    state.add_robot(robot2)

    print("Initial Robot 1:", robot.position)
    print("Initial Robot 2:", robot2.position)

    # Simple conflict pre-flight check before the real task starts.
    original_position = robot.position
    robot.position = (4, 3)

    print("\n=== CONFLICT PRE-FLIGHT TEST ===")

    conflict_target = robot2.position

    orchestrator.train_sarsa_for_target(
        robot_id=1,
        target=conflict_target,
        episodes=int(os.getenv("SARSA_EPISODES", "500")),
    )

    conflict_move_done = orchestrator.execute_sarsa_move(
        robot_id=1,
        target=conflict_target,
        conflict_robot_id=2,
    )

    print("Conflict-aware move executed:", conflict_move_done)
    print("Robot 1 after conflict test:", robot.position)
    print("Robot 2:", robot2.position)

    robot.position = original_position
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

    print("\n=== NEURO-SYMBOLIC DISPATCH ===")

    human_request = (
        "Dispatch ITEM-001 from its pickup location to its task "
        "destination using a suitable available robot. Prefer completing "
        "the urgent task without violating any robot safety constraints."
    )

    dispatch_result = dispatch_verified_plan(
        user_request=human_request,
        state=state,
        orchestrator=orchestrator,
    )

    print("Dispatch status:", dispatch_result["status"])

    if dispatch_result["status"] != "DISPATCH_AUTHORIZED":
        print("Dispatch result:")
        print(dispatch_result["result"])
        return

    print("Verified assignments:")
    for assignment in dispatch_result["task_assignments"]:
        print(" ", assignment)

    # Convert the SMT result to the robot ID format used by the orchestrator.
    assigned_robot_id = int(dispatch_result["task_assignments"][0]["robot_id"])

    pickup_route = orchestrator.plan_route(
        robot_id=assigned_robot_id,
        destination=item.location,
    )

    print("\n=== PICKUP ROUTE ===")
    print(pickup_route)

    orchestrator.train_sarsa_for_target(
        robot_id=assigned_robot_id,
        target=item.location,
        episodes=int(os.getenv("SARSA_EPISODES", "500")),
    )

    while orchestrator.execute_next_move(
        assigned_robot_id,
        use_sarsa=True,
    ):
        pass

    orchestrator.pick_item(
        assigned_robot_id,
        item.item_id,
    )

    delivery_route = orchestrator.plan_route(
        robot_id=assigned_robot_id,
        destination=task.destination,
    )

    print("\n=== DELIVERY ROUTE ===")
    print(delivery_route)

    orchestrator.train_sarsa_for_target(
        robot_id=assigned_robot_id,
        target=task.destination,
        episodes=int(os.getenv("SARSA_EPISODES", "500")),
    )

    while orchestrator.execute_next_move(
        assigned_robot_id,
        use_sarsa=True,
    ):
        pass

    orchestrator.deliver_item(assigned_robot_id)

    print("\n=== FINAL STATE ===")
    assigned_robot = state.robots[assigned_robot_id]
    print("Assigned robot:", assigned_robot_id)
    print("Robot position:", assigned_robot.position)
    print("Estimated position:", assigned_robot.estimated_position)
    print("Belief sum:", assigned_robot.belief.sum())
    print("Most likely probability:", assigned_robot.belief.max())
    print("Last observation:", assigned_robot.last_observation)
    print("Item status:", item.status)
    print("Task status:", state.tasks["TASK-001"].status)

    print("\n=== SQLITE MEMORY ===")
    print("Run ID:", run_id)
    print("Events stored:", memory.count_events(run_id))
    print("Recent event types:")
    for event in reversed(memory.recent_events(run_id, limit=10)):
        print(" ", event["event_type"])

    robot_history = memory.robot_history(run_id, assigned_robot_id)
    print("Robot snapshots stored:", len(robot_history))
    print("Task snapshots stored:", len(memory.task_history(run_id, "TASK-001")))
    print("Database:", memory.db_path)
    memory.close()


if __name__ == "__main__":
    main()
