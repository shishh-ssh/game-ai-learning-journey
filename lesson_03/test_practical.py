from astar import reconstruct_path


def test_reconstruct_path_returns_start_to_goal_order() -> None:
    parents = {
        (0, 0): None,
        (0, 1): (0, 0),
        (1, 1): (0, 1),
    }

    assert reconstruct_path((1, 1), parents) == [
        (0, 0),
        (0, 1),
        (1, 1),
    ]


def test_reconstruct_path_handles_start_only() -> None:
    assert reconstruct_path((2, 2), {(2, 2): None}) == [(2, 2)]
