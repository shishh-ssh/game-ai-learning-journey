from baselines import AlwaysRightPolicy, RandomPolicy
from line_world_env import LineWorldEnv


def test_always_right_policy_returns_right_for_every_observation():
    policy = AlwaysRightPolicy()

    actions = [policy.select_action(observation) for observation in range(5)]

    assert actions == [LineWorldEnv.RIGHT] * 5


def test_random_policy_only_returns_legal_actions():
    policy = RandomPolicy(seed=42)

    actions = [policy.select_action(observation=0) for _ in range(100)]

    assert set(actions) == {LineWorldEnv.LEFT, LineWorldEnv.RIGHT}


def test_random_policy_reproduces_sequence_with_same_seed():
    first_policy = RandomPolicy(seed=7)
    second_policy = RandomPolicy(seed=7)

    first_actions = [first_policy.select_action(0) for _ in range(20)]
    second_actions = [second_policy.select_action(0) for _ in range(20)]

    assert first_actions == second_actions


def test_random_policy_state_advances_between_calls():
    policy = RandomPolicy(seed=11)

    actions = [policy.select_action(0) for _ in range(20)]

    assert len(set(actions)) == 2
