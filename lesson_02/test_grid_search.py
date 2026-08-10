import pytest

from grid_search import find_shortest_path, get_neighbors


def test_get_neighbors_inside_world() -> None:
    assert get_neighbors((1, 1), width=3, height=3, blocked=set()) == [
        (1, 0),
        (1, 2),
        (0, 1),
        (2, 1),
    ]


def test_get_neighbors_at_corner() -> None:
    assert get_neighbors((0, 0), width=3, height=3, blocked=set()) == [
        (0, 1),
        (1, 0),
    ]


def test_get_neighbors_filters_blocked_cells() -> None:
    assert get_neighbors(
        (1, 1),
        width=3,
        height=3,
        blocked={(1, 0), (0, 1)},
    ) == [
        (1, 2),
        (2, 1),
    ]


def test_get_neighbors_returns_empty_when_surrounded() -> None:
    assert get_neighbors(
        (1, 1),
        width=3,
        height=3,
        blocked={(1, 0), (1, 2), (0, 1), (2, 1)},
    ) == []


@pytest.mark.parametrize(
    ("width", "height"),
    [
        (0, 3),
        (-1, 3),
        (3, 0),
        (3, -1),
    ],
)
def test_get_neighbors_rejects_invalid_world_size(
    width: int,
    height: int,
) -> None:
    with pytest.raises(ValueError):
        get_neighbors((1, 1), width=width, height=height, blocked=set())


@pytest.mark.parametrize(
    "position",
    [
        (-1, 1),
        (3, 1),
        (1, -1),
        (1, 3),
    ],
)
def test_get_neighbors_rejects_invalid_position(
    position: tuple[int, int],
) -> None:
    with pytest.raises(ValueError):
        get_neighbors(position, width=3, height=3, blocked=set())


def test_get_neighbors_rejects_blocked_position() -> None:
    with pytest.raises(ValueError):
        get_neighbors((1, 1), width=3, height=3, blocked={(1, 1)})


def test_find_shortest_path_returns_start_when_already_at_goal() -> None:
    assert find_shortest_path(
        (1, 1),
        (1, 1),
        width=3,
        height=3,
        blocked=set(),
    ) == [(1, 1)]


def test_find_shortest_path_reaches_adjacent_goal() -> None:
    assert find_shortest_path(
        (0, 0),
        (1, 0),
        width=3,
        height=3,
        blocked=set(),
    ) == [
        (0, 0),
        (1, 0),
    ]


def test_find_shortest_path_uses_deterministic_shortest_route() -> None:
    assert find_shortest_path(
        (0, 0),
        (2, 1),
        width=3,
        height=3,
        blocked=set(),
    ) == [
        (0, 0),
        (0, 1),
        (1, 1),
        (2, 1),
    ]


def test_find_shortest_path_detours_around_wall() -> None:
    assert find_shortest_path(
        (0, 1),
        (4, 1),
        width=5,
        height=3,
        blocked={(1, 1), (2, 1), (3, 1)},
    ) == [
        (0, 1),
        (0, 0),
        (1, 0),
        (2, 0),
        (3, 0),
        (4, 0),
        (4, 1),
    ]


def test_find_shortest_path_returns_empty_when_unreachable() -> None:
    assert find_shortest_path(
        (0, 0),
        (2, 2),
        width=3,
        height=3,
        blocked={(1, 0), (0, 1)},
    ) == []


def test_find_shortest_path_rejects_blocked_start() -> None:
    with pytest.raises(ValueError):
        find_shortest_path(
            (0, 0),
            (2, 2),
            width=3,
            height=3,
            blocked={(0, 0)},
        )


def test_find_shortest_path_rejects_blocked_goal() -> None:
    with pytest.raises(ValueError):
        find_shortest_path(
            (0, 0),
            (2, 2),
            width=3,
            height=3,
            blocked={(2, 2)},
        )


def test_find_shortest_path_rejects_out_of_bounds_start() -> None:
    with pytest.raises(ValueError):
        find_shortest_path(
            (-1, 0),
            (2, 2),
            width=3,
            height=3,
            blocked=set(),
        )


def test_find_shortest_path_rejects_out_of_bounds_goal() -> None:
    with pytest.raises(ValueError):
        find_shortest_path(
            (0, 0),
            (3, 2),
            width=3,
            height=3,
            blocked=set(),
        )


@pytest.mark.parametrize(
    ("width", "height"),
    [
        (0, 3),
        (3, 0),
    ],
)
def test_find_shortest_path_rejects_invalid_world_size(
    width: int,
    height: int,
) -> None:
    with pytest.raises(ValueError):
        find_shortest_path(
            (0, 0),
            (2, 2),
            width=width,
            height=height,
            blocked=set(),
        )

