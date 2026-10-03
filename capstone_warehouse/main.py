from environment import SurgicalLabEnvironment

from robot import Robot
from task import Task, InventoryItem
from warehouse_state import WarehouseState
from orchestrator import WarehouseOrchestrator


def main():
    # Existing A* warehouse environment
    env = SurgicalLabEnvironment(map_type="custom")

    # Shared warehouse state
    state = WarehouseState(env)

    # System controller
    orchestrator = WarehouseOrchestrator(state)

    # --------------------------------------------------
    # Create robots
    # --------------------------------------------------
    robot = Robot(
        robot_id=1,
        position=(4, 4),
        capacity=10.0,
        battery=100.0,
    )
    state.add_robot(robot)

    print("True position:", robot.position)
    print("Estimated position:", robot.estimated_position)

    if robot.belief is not None:
        print("Belief sum:", robot.belief.sum())

    robot2 = Robot(
        robot_id=2,
        position=(4, 2),
        capacity=10.0,
        battery=100.0,
    )
    state.add_robot(robot2)

    # --------------------------------------------------
    # Dedicated SARSA + Minimax/Alpha-Beta conflict test
    # --------------------------------------------------
    # Temporarily place Robot 1 in a head-on conflict setup.
    original_position = robot.position
    robot.position = (4, 3)

    print("\n=== SARSA + CONFLICT INTEGRATION TEST ===")

    # Train SARSA toward the cell occupied by Robot 2.
    # This creates a situation where SARSA has an incentive
    # to propose a conflicting move.
    conflict_target = robot2.position

    orchestrator.train_sarsa_for_target(
        robot_id=1,
        target=conflict_target,
        episodes=2000,
    )

    conflict_move_done = orchestrator.execute_sarsa_move(
        robot_id=1,
        target=conflict_target,
        conflict_robot_id=2,
    )

    print("Conflict-aware move executed:", conflict_move_done)
    print("Robot 1 position after conflict check:", robot.position)
    print("Robot 2 position:", robot2.position)

    # Restore the normal starting position for the real warehouse task.
    robot.position = original_position

    # Reset the HMM belief to the restored true position.
    state.perception.initialize_robot(robot)

    print("Restored Robot 1 position:", robot.position)

    # --------------------------------------------------
    # Create inventory item
    # --------------------------------------------------
    item = InventoryItem(
        item_id="ITEM-001",
        weight=6.0,
        location=(5, 4),
    )
    state.add_item(item)

    # --------------------------------------------------
    # Create task
    # --------------------------------------------------
    task = Task(
        task_id="TASK-001",
        item_id="ITEM-001",
        pickup=(5, 4),
        destination=(5, 5),
        priority="URGENT",
        deadline=20,
    )
    state.add_task(task)

    # --------------------------------------------------
    # Assign task
    # --------------------------------------------------
    orchestrator.assign_task("TASK-001", 1)

    # --------------------------------------------------
    # A* → Pickup
    # --------------------------------------------------
    pickup_route = orchestrator.plan_route(
        robot_id=1,
        destination=item.location,
    )

    print("\n=== PICKUP ROUTE ===")
    print(pickup_route)

    orchestrator.train_sarsa_for_target(
        robot_id=1,
        target=item.location,
        episodes=2000,
    )

    while orchestrator.execute_next_move(
        1,
        use_sarsa=True,
    ):
        pass

    # Pick up package
    orchestrator.pick_item(1, item.item_id)

    # --------------------------------------------------
    # A* → Delivery
    # --------------------------------------------------
    delivery_route = orchestrator.plan_route(
        robot_id=1,
        destination=task.destination,
    )

    print("\n=== DELIVERY ROUTE ===")
    print(delivery_route)

    orchestrator.train_sarsa_for_target(
        robot_id=1,
        target=task.destination,
        episodes=2000,
    )

    while orchestrator.execute_next_move(
        1,
        use_sarsa=True,
    ):
        pass

    # Deliver package
    orchestrator.deliver_item(1)

    # --------------------------------------------------
    # Final state
    # --------------------------------------------------
    print("Estimated position:", robot.estimated_position)
    print("Most likely probability:", robot.belief.max())
    print("Belief sum:", robot.belief.sum())
    print("Last observation:", robot.last_observation)


if __name__ == "__main__":
    main()
