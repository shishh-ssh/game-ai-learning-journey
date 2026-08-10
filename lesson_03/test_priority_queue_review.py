from priority_queue_review import pop_positions_by_priority


def test_pops_positions_from_lowest_priority_to_highest() -> None:
    assert pop_positions_by_priority(
        [(8, "A"), (5, "B"), (7, "C")],
    ) == ["B", "C", "A"]


def test_insertion_order_does_not_control_removal_order() -> None:
    assert pop_positions_by_priority(
        [(12, "start"), (2, "goal_side"), (6, "detour")],
    ) == ["goal_side", "detour", "start"]


def test_handles_one_entry() -> None:
    assert pop_positions_by_priority([(4, "only")]) == ["only"]


def test_returns_empty_for_no_entries() -> None:
    assert pop_positions_by_priority([]) == []


def test_does_not_modify_input_entries() -> None:
    entries = [(4, "A"), (1, "B"), (3, "C")]
    original_entries = entries.copy()

    pop_positions_by_priority(entries)

    assert entries == original_entries
