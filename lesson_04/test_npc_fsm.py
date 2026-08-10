import pytest

from npc_fsm import transition_state


def test_rejects_unknown_state() -> None:
    with pytest.raises(ValueError):
        transition_state("sleep", 100, False, 5.0)


@pytest.mark.parametrize("health", [-1, 101])
def test_rejects_invalid_health(health: int) -> None:
    with pytest.raises(ValueError):
        transition_state("patrol", health, False, 5.0)


def test_rejects_negative_distance() -> None:
    with pytest.raises(ValueError):
        transition_state("patrol", 100, False, -0.1)


@pytest.mark.parametrize(
    ("health", "enemy_visible", "distance"),
    [
        (0, False, 5.0),
        (100, True, 1.0),
        (100, False, 10.0),
    ],
)
def test_dead_state_is_terminal(
    health: int,
    enemy_visible: bool,
    distance: float,
) -> None:
    assert transition_state("dead", health, enemy_visible, distance) == "dead"


@pytest.mark.parametrize("current_state", ["patrol", "chase", "attack", "flee"])
def test_zero_health_transitions_to_dead(current_state: str) -> None:
    assert transition_state(current_state, 0, True, 1.0) == "dead"


@pytest.mark.parametrize("current_state", ["patrol", "chase", "attack"])
def test_low_health_interrupts_active_state(current_state: str) -> None:
    assert transition_state(current_state, 20, True, 1.0) == "flee"


def test_flee_continues_before_recovery_threshold() -> None:
    assert transition_state("flee", 49, False, 5.0) == "flee"


def test_flee_returns_to_patrol_after_recovery() -> None:
    assert transition_state("flee", 50, True, 1.0) == "patrol"


def test_patrol_continues_without_visible_enemy() -> None:
    assert transition_state("patrol", 100, False, 5.0) == "patrol"


def test_patrol_starts_chasing_visible_enemy() -> None:
    assert transition_state("patrol", 100, True, 1.0) == "chase"


def test_chase_returns_to_patrol_when_enemy_disappears() -> None:
    assert transition_state("chase", 100, False, 1.0) == "patrol"


def test_chase_attacks_when_enemy_is_close() -> None:
    assert transition_state("chase", 100, True, 2.0) == "attack"


def test_chase_continues_while_enemy_is_far() -> None:
    assert transition_state("chase", 100, True, 2.1) == "chase"


def test_attack_returns_to_patrol_when_enemy_disappears() -> None:
    assert transition_state("attack", 100, False, 1.0) == "patrol"


def test_attack_returns_to_chase_when_enemy_moves_away() -> None:
    assert transition_state("attack", 100, True, 2.1) == "chase"


def test_attack_continues_while_enemy_is_close() -> None:
    assert transition_state("attack", 100, True, 2.0) == "attack"
