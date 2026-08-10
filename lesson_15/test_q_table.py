import torch

from q_table import update_q_value


def test_updates_current_pair_from_non_terminal_transition():
    q_table = torch.tensor(
        [
            [1.0, 2.0],
            [4.0, 3.0],
            [6.0, 8.0],
        ]
    )
    original = q_table.clone()

    new_q = update_q_value(
        q_table,
        state=0,
        action=1,
        reward=1.0,
        next_state=2,
        done=False,
        gamma=0.5,
        alpha=0.25,
    )

    assert new_q == 2.75
    assert q_table[0, 1].item() == 2.75
    original[0, 1] = 2.75
    assert torch.equal(q_table, original)


def test_terminal_transition_does_not_use_next_q_value():
    q_table = torch.tensor([[2.0, 0.0], [100.0, 200.0]])

    new_q = update_q_value(
        q_table,
        state=0,
        action=0,
        reward=10.0,
        next_state=1,
        done=True,
        gamma=0.9,
        alpha=0.5,
    )

    assert new_q == 6.0
    assert q_table[0, 0].item() == 6.0
