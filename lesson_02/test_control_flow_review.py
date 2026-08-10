from control_flow_review import collect_actions_until_stop


def test_collects_all_actions_without_stop() -> None:
    assert collect_actions_until_stop(
        [["move", "attack"], ["heal"]],
    ) == ["move", "attack", "heal"]


def test_returns_empty_when_stop_is_first_action() -> None:
    assert collect_actions_until_stop(
        [["stop", "move"], ["attack"]],
    ) == []


def test_stops_in_the_middle_of_a_turn() -> None:
    assert collect_actions_until_stop(
        [["move", "stop", "attack"], ["heal"]],
    ) == ["move"]


def test_stops_after_processing_earlier_turns() -> None:
    assert collect_actions_until_stop(
        [["move", "attack"], [], ["heal", "stop", "move"]],
    ) == ["move", "attack", "heal"]


def test_returns_empty_for_no_turns() -> None:
    assert collect_actions_until_stop([]) == []
