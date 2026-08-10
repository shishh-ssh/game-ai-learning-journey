import numpy as np

from lesson_06.epsilon_policy import choose_epsilon_greedy


def test_zero_epsilon_always_uses_best_actions() -> None:
    result = choose_epsilon_greedy(
        [[2.0, 7.0, 4.0], [9.0, 3.0, 5.0]],
        epsilon=0.0,
        rng=np.random.default_rng(11),
    )

    np.testing.assert_array_equal(result, np.array([1, 0]))


def test_one_epsilon_uses_seeded_random_actions() -> None:
    score_table = [[9.0, 1.0, 0.0], [9.0, 1.0, 0.0], [9.0, 1.0, 0.0]]
    expected_rng = np.random.default_rng(42)
    expected_rng.random(3)
    expected = expected_rng.integers(3, size=3)

    result = choose_epsilon_greedy(
        score_table,
        epsilon=1.0,
        rng=np.random.default_rng(42),
    )

    np.testing.assert_array_equal(result, expected)


def test_mixed_decisions_match_seeded_numpy_operations() -> None:
    score_table = [[4.0, 9.0, 2.0], [8.0, 3.0, 5.0], [1.0, 6.0, 7.0]]
    expected_rng = np.random.default_rng(7)
    explore_mask = expected_rng.random(3) < 0.5
    random_actions = expected_rng.integers(3, size=3)
    greedy_actions = np.array(score_table, dtype=np.float64).argmax(axis=1)
    expected = np.where(explore_mask, random_actions, greedy_actions)

    result = choose_epsilon_greedy(
        score_table,
        epsilon=0.5,
        rng=np.random.default_rng(7),
    )

    np.testing.assert_array_equal(result, expected)


def test_same_seed_reproduces_actions_and_returns_one_dimension() -> None:
    score_table = [[1.0, 5.0], [7.0, 2.0], [3.0, 4.0], [8.0, 6.0]]

    first = choose_epsilon_greedy(score_table, 0.6, np.random.default_rng(19))
    second = choose_epsilon_greedy(score_table, 0.6, np.random.default_rng(19))

    assert isinstance(first, np.ndarray)
    assert first.shape == (4,)
    assert np.issubdtype(first.dtype, np.integer)
    np.testing.assert_array_equal(first, second)


def test_input_table_is_not_modified() -> None:
    score_table = [[2.0, 8.0], [6.0, 1.0]]
    original = [row.copy() for row in score_table]

    choose_epsilon_greedy(score_table, 0.4, np.random.default_rng(3))

    assert score_table == original
