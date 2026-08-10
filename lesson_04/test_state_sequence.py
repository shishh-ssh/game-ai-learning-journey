from state_sequence import run_state_sequence


def test_empty_observations_return_initial_state() -> None:
    assert run_state_sequence("patrol", []) == ["patrol"]


def test_each_frame_uses_previous_returned_state() -> None:
    observations = [
        (100, True, 5.0),
        (100, True, 1.0),
    ]

    assert run_state_sequence("patrol", observations) == [
        "patrol",
        "chase",
        "attack",
    ]


def test_recovery_does_not_cascade_in_the_same_frame() -> None:
    observations = [
        (50, True, 1.0),
        (50, True, 1.0),
        (50, True, 1.0),
    ]

    assert run_state_sequence("flee", observations) == [
        "flee",
        "patrol",
        "chase",
        "attack",
    ]


def test_dead_state_is_preserved_across_later_frames() -> None:
    observations = [
        (0, True, 1.0),
        (100, True, 1.0),
    ]

    assert run_state_sequence("chase", observations) == [
        "chase",
        "dead",
        "dead",
    ]
