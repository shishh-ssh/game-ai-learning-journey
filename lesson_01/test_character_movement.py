import pytest

from character_movement import execute_commands, move_character


@pytest.mark.parametrize(
    ("position", "command", "expected"),
    [
        ((2, 2), "up", (2, 1)),
        ((2, 2), "down", (2, 3)),
        ((2, 2), "left", (1, 2)),
        ((2, 2), "right", (3, 2)),
    ],
)
def test_move_character_moves_inside_world(
    position: tuple[int, int],
    command: str,
    expected: tuple[int, int],
) -> None:
    assert move_character(position, command, width=5, height=5) == expected


@pytest.mark.parametrize(
    ("position", "command"),
    [
        ((2, 0), "up"),
        ((2, 4), "down"),
        ((0, 2), "left"),
        ((4, 2), "right"),
    ],
)
def test_move_character_stays_at_boundary(
    position: tuple[int, int],
    command: str,
) -> None:
    assert move_character(position, command, width=5, height=5) == position


def test_move_character_rejects_unknown_command() -> None:
    with pytest.raises(ValueError):
        move_character((2, 2), "jump", width=5, height=5)


@pytest.mark.parametrize(
    ("width", "height"),
    [
        (0, 5),
        (-1, 5),
        (5, 0),
        (5, -1),
    ],
)
def test_move_character_rejects_invalid_world_size(
    width: int,
    height: int,
) -> None:
    with pytest.raises(ValueError):
        move_character((2, 2), "up", width=width, height=height)


@pytest.mark.parametrize(
    "position",
    [
        (-1, 2),
        (5, 2),
        (2, -1),
        (2, 5),
    ],
)
def test_move_character_rejects_invalid_position(
    position: tuple[int, int],
) -> None:
    with pytest.raises(ValueError):
        move_character(position, "up", width=5, height=5)


def test_execute_commands_returns_complete_trajectory() -> None:
    assert execute_commands(
        (0, 0),
        ["right", "right", "down", "left"],
        width=5,
        height=5,
    ) == [
        (0, 0),
        (1, 0),
        (2, 0),
        (2, 1),
        (1, 1),
    ]


def test_execute_commands_empty_sequence_includes_start() -> None:
    assert execute_commands((2, 3), [], width=5, height=5) == [(2, 3)]


def test_execute_commands_records_blocked_move() -> None:
    assert execute_commands(
        (0, 0),
        ["up", "right"],
        width=5,
        height=5,
    ) == [
        (0, 0),
        (0, 0),
        (1, 0),
    ]


def test_execute_commands_rejects_unknown_command() -> None:
    with pytest.raises(ValueError):
        execute_commands((0, 0), ["right", "jump"], width=5, height=5)


def test_execute_commands_empty_sequence_rejects_invalid_position() -> None:
    with pytest.raises(ValueError):
        execute_commands((-1, 0), [], width=5, height=5)


def test_execute_commands_empty_sequence_rejects_invalid_world_size() -> None:
    with pytest.raises(ValueError):
        execute_commands((0, 0), [], width=0, height=5)


def test_move_character_stays_for_stay_command() -> None:
    assert move_character((2, 2), "stay", width=5, height=5) == (2, 2)


def test_execute_commands_records_stay_command() -> None:
    assert execute_commands(
        (1, 1),
        ["stay", "right"],
        width=5,
        height=5,
    ) == [
        (1, 1),
        (1, 1),
        (2, 1),
    ]
