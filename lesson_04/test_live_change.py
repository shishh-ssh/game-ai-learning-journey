from npc_fsm import transition_state


def test_patrol_chases_visible_enemy_at_detection_boundary() -> None:
    assert transition_state("patrol", 100, True, 6.0) == "chase"


def test_patrol_ignores_visible_enemy_beyond_detection_range() -> None:
    assert transition_state("patrol", 100, True, 6.1) == "patrol"


def test_patrol_stays_patrolling_when_enemy_is_hidden_inside_range() -> None:
    assert transition_state("patrol", 100, False, 1.0) == "patrol"
