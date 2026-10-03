from environment import SurgicalLabEnvironment
from warehouse_mdp import WarehouseMDP
from warehouse_qlearning import WarehouseQLearning
from route_bandit import UCB1Bandit
from warehouse_state import WarehouseState
from robot import Robot
from orchestrator import WarehouseOrchestrator


def test_value_iteration_route_is_valid():
    env = SurgicalLabEnvironment(map_type="trivial")
    plan = WarehouseMDP(env).plan((0, 0), (9, 9))
    assert plan.path[0] == (0, 0)
    assert plan.path[-1] == (9, 9)
    assert all(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
               and env.is_valid(*b) for a, b in zip(plan.path, plan.path[1:]))


def test_value_iteration_obstacle_avoidance():
    env = SurgicalLabEnvironment(map_type="custom")
    plan = WarehouseMDP(env).plan((2, 2), (8, 7))
    assert plan.path[-1] == (8, 7)
    assert all(env.is_valid(*cell) for cell in plan.path)


def test_bandit_samples_each_arm_and_updates():
    bandit = UCB1Bandit(["left", "right"], seed=0)
    assert bandit.select() == "left"
    bandit.update("left", 0.2)
    assert bandit.select() == "right"
    bandit.update("right", 0.8)
    assert bandit.summary()["right"]["mean_reward"] == 0.8


def test_qlearning_trains_and_learns_q_values():
    env = SurgicalLabEnvironment(map_type="trivial")
    agent = WarehouseQLearning(env, seed=3)
    returns, successes = agent.train((0, 0), (2, 0), episodes=30, max_steps=10)
    assert len(returns) == 30
    assert successes > 0
    assert agent.q_table


def test_orchestrator_can_use_mdp_route_and_bandit():
    env = SurgicalLabEnvironment(map_type="trivial")
    state = WarehouseState(env)
    state.add_robot(Robot(robot_id=1, position=(0, 0), capacity=10, battery=100))
    orchestrator = WarehouseOrchestrator(state)
    route = orchestrator.plan_route(1, (3, 0), strategy="value_iteration")
    assert route[0] == (0, 0) and route[-1] == (3, 0)
    route2 = orchestrator.plan_route(1, (4, 0), strategy="bandit")
    assert route2[-1] == (4, 0)
    assert sum(orchestrator.route_bandit.counts.values()) == 1


def test_orchestrator_qlearning_move_stays_on_planned_route():
    env = SurgicalLabEnvironment(map_type="trivial")
    state = WarehouseState(env)
    robot = Robot(robot_id=1, position=(0, 0), capacity=10, battery=100)
    state.add_robot(robot)
    orchestrator = WarehouseOrchestrator(state)
    route = orchestrator.plan_route(1, (3, 0), strategy="astar")
    orchestrator.train_qlearning_for_target(1, (3, 0), episodes=40)
    moved = orchestrator.execute_qlearning_move(1, (3, 0))
    assert robot.position in route
    assert robot.position == (0, 0) or robot.position == route[1]
    assert isinstance(moved, bool)
