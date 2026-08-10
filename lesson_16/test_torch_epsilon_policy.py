import torch

from torch_epsilon_policy import select_action


def test_zero_epsilon_always_uses_greedy_action():
    q_table = torch.tensor([[2.0, 7.0, 4.0]])
    generator = torch.Generator().manual_seed(42)

    actions = [select_action(q_table, 0, 0.0, generator) for _ in range(10)]

    assert actions == [1] * 10


def test_full_exploration_returns_legal_actions():
    q_table = torch.zeros((1, 3))
    generator = torch.Generator().manual_seed(42)

    actions = [select_action(q_table, 0, 1.0, generator) for _ in range(30)]

    assert all(0 <= action < 3 for action in actions)
    assert len(set(actions)) > 1


def test_same_seed_reproduces_action_sequence():
    q_table = torch.tensor([[2.0, 7.0, 4.0]])
    first_generator = torch.Generator().manual_seed(7)
    second_generator = torch.Generator().manual_seed(7)

    first = [select_action(q_table, 0, 0.5, first_generator) for _ in range(20)]
    second = [select_action(q_table, 0, 0.5, second_generator) for _ in range(20)]

    assert first == second
