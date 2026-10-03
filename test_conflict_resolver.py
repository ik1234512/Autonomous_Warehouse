from environment import SurgicalLabEnvironment
from conflict_resolver import ConflictResolver


def test_head_on_conflict():
    env = SurgicalLabEnvironment(map_type="custom")
    resolver = ConflictResolver(env, depth=2)

    robot_a = (4, 3)
    robot_b = (4, 2)

    target_a = (4, 1)
    target_b = (4, 4)

    print("=== HEAD-ON CONFLICT TEST ===")
    print("Robot A:", robot_a, "->", target_a)
    print("Robot B:", robot_b, "->", target_b)

    print("A valid moves:", resolver.valid_moves(robot_a))
    print("B valid moves:", resolver.valid_moves(robot_b))

    move = resolver.resolve(
        robot_position=robot_a,
        robot_target=target_a,
        other_position=robot_b,
        other_target=target_b,
    )

    print("Resolver selected:", move)
    print("Collision avoided:", move.position != robot_b)

    assert move.position != robot_b


def test_swap_conflict():
    env = SurgicalLabEnvironment(map_type="custom")
    resolver = ConflictResolver(env, depth=2)

    robot_a = (4, 3)
    robot_b = (4, 4)

    target_a = robot_b
    target_b = robot_a

    print("\n=== POSITION-SWAP CONFLICT TEST ===")
    print("Robot A:", robot_a, "->", target_a)
    print("Robot B:", robot_b, "->", target_b)

    move = resolver.resolve(
        robot_position=robot_a,
        robot_target=target_a,
        other_position=robot_b,
        other_target=target_b,
    )

    print("Resolver selected:", move)

    swap_detected = (
        move.position == robot_b
    )

    print("Immediate entry into B's cell:", swap_detected)

    assert not swap_detected


if __name__ == "__main__":
    test_head_on_conflict()
    test_swap_conflict()
    print("\nAll conflict-resolution tests passed.")
