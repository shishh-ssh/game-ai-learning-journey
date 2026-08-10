import pytest

from astar import (
    find_shortest_path_astar,
    manhattan_distance,
    update_neighbor_if_better,
)
from lesson_02.grid_search import find_shortest_path


def assert_valid_path(
    path: list[tuple[int, int]],
    start: tuple[int, int],
    goal: tuple[int, int],
    blocked: set[tuple[int, int]],
) -> None:
    assert path[0] == start
    assert path[-1] == goal
    assert all(position not in blocked for position in path)
    assert all(
        abs(first[0] - second[0]) + abs(first[1] - second[1]) == 1
        for first, second in zip(path, path[1:])
    )


@pytest.mark.parametrize(
    ("position", "goal", "expected"),
    [
        ((0, 0), (0, 0), 0),
        ((0, 0), (3, 0), 3),
        ((2, 5), (2, 1), 4),
        ((1, 2), (4, 6), 7),
        ((-2, 3), (1, -1), 7),
    ],
)
def test_manhattan_distance(
    position: tuple[int, int],
    goal: tuple[int, int],
    expected: int,
) -> None:
    assert manhattan_distance(position, goal) == expected


def test_manhattan_distance_is_symmetric() -> None:
    assert manhattan_distance((1, 5), (4, 1)) == manhattan_distance(
        (4, 1),
        (1, 5),
    )


def test_update_neighbor_records_unseen_position() -> None:
    current = (2, 0)
    neighbor = (2, 1)
    g_score = {current: 2}
    parents = {current: None}

    entry = update_neighbor_if_better(
        current,
        neighbor,
        goal=(4, 1),
        g_score=g_score,
        parents=parents,
    )

    assert entry == (5, neighbor)
    assert g_score[neighbor] == 3
    assert parents[neighbor] == current


def test_update_neighbor_replaces_more_expensive_route() -> None:
    current = (2, 0)
    neighbor = (2, 1)
    g_score = {current: 2, neighbor: 5}
    parents = {current: (1, 0), neighbor: (2, 2)}

    entry = update_neighbor_if_better(
        current,
        neighbor,
        goal=(4, 1),
        g_score=g_score,
        parents=parents,
    )

    assert entry == (5, neighbor)
    assert g_score[neighbor] == 3
    assert parents[neighbor] == current


@pytest.mark.parametrize("known_cost", [3, 2])
def test_update_neighbor_keeps_equal_or_better_known_route(
    known_cost: int,
) -> None:
    current = (2, 0)
    neighbor = (2, 1)
    old_parent = (1, 1)
    g_score = {current: 2, neighbor: known_cost}
    parents = {current: (1, 0), neighbor: old_parent}

    entry = update_neighbor_if_better(
        current,
        neighbor,
        goal=(4, 1),
        g_score=g_score,
        parents=parents,
    )

    assert entry is None
    assert g_score[neighbor] == known_cost
    assert parents[neighbor] == old_parent


def test_astar_returns_start_when_already_at_goal() -> None:
    assert find_shortest_path_astar(
        (1, 1),
        (1, 1),
        width=3,
        height=3,
        blocked=set(),
    ) == [(1, 1)]


def test_astar_reaches_adjacent_goal() -> None:
    assert find_shortest_path_astar(
        (0, 0),
        (1, 0),
        width=3,
        height=3,
        blocked=set(),
    ) == [(0, 0), (1, 0)]


@pytest.mark.parametrize(
    ("start", "goal", "width", "height", "blocked"),
    [
        ((0, 0), (4, 2), 5, 3, set()),
        ((0, 1), (4, 1), 5, 3, {(1, 1), (2, 1), (3, 1)}),
    ],
)
def test_astar_matches_bfs_shortest_path_length(
    start: tuple[int, int],
    goal: tuple[int, int],
    width: int,
    height: int,
    blocked: set[tuple[int, int]],
) -> None:
    astar_path = find_shortest_path_astar(
        start,
        goal,
        width,
        height,
        blocked,
    )
    bfs_path = find_shortest_path(start, goal, width, height, blocked)

    assert_valid_path(astar_path, start, goal, blocked)
    assert len(astar_path) == len(bfs_path)


def test_astar_returns_empty_when_unreachable() -> None:
    assert find_shortest_path_astar(
        (0, 0),
        (2, 2),
        width=3,
        height=3,
        blocked={(1, 0), (0, 1)},
    ) == []


@pytest.mark.parametrize(("width", "height"), [(0, 3), (3, 0)])
def test_astar_rejects_invalid_world_size(width: int, height: int) -> None:
    with pytest.raises(ValueError):
        find_shortest_path_astar(
            (0, 0),
            (2, 2),
            width=width,
            height=height,
            blocked=set(),
        )


@pytest.mark.parametrize(
    ("start", "goal"),
    [
        ((-1, 0), (2, 2)),
        ((0, 0), (3, 2)),
    ],
)
def test_astar_rejects_out_of_bounds_endpoint(
    start: tuple[int, int],
    goal: tuple[int, int],
) -> None:
    with pytest.raises(ValueError):
        find_shortest_path_astar(
            start,
            goal,
            width=3,
            height=3,
            blocked=set(),
        )


@pytest.mark.parametrize("blocked_position", [(0, 0), (2, 2)])
def test_astar_rejects_blocked_endpoint(
    blocked_position: tuple[int, int],
) -> None:
    with pytest.raises(ValueError):
        find_shortest_path_astar(
            (0, 0),
            (2, 2),
            width=3,
            height=3,
            blocked={blocked_position},
        )
