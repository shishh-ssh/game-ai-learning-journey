import pytest

from npc_policy import choose_action


@pytest.mark.parametrize(
    ("distance", "health", "enemy_visible", "expected"),
    [
        (1.5, 80, True, "attack"),
        (8.0, 80, True, "chase"),
        (1.0, 20, True, "flee"),
        (1.0, 0, True, "dead"),
        (1.0, 80, False, "patrol"),
        (1.0, 20, False, "patrol"),
        (1.0, 30, True, "flee"),
    ],
)
def test_choose_action(
    distance: float,
    health: int,
    enemy_visible: bool,
    expected: str,
) -> None:
    assert choose_action(distance, health, enemy_visible) == expected


@pytest.mark.parametrize(
    ("distance", "health"),
    [
        (-0.1, 80),
        (1.0, -1),
        (1.0, 101),
    ],
)
def test_choose_action_rejects_invalid_state(distance: float, health: int) -> None:
    with pytest.raises(ValueError):
        choose_action(distance, health, True)

