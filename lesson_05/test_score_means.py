import numpy as np

from array_basics import compute_score_means


def test_computes_action_and_state_means() -> None:
    action_means, state_means = compute_score_means(
        [
            [2.0, 4.0],
            [6.0, 8.0],
            [10.0, 12.0],
        ]
    )

    np.testing.assert_allclose(action_means, np.array([6.0, 8.0]))
    np.testing.assert_allclose(state_means, np.array([3.0, 7.0, 11.0]))


def test_uses_axes_for_a_different_non_square_shape() -> None:
    action_means, state_means = compute_score_means(
        [
            [1.0, 3.0, 5.0],
            [2.0, 4.0, 6.0],
        ]
    )

    np.testing.assert_allclose(action_means, np.array([1.5, 3.5, 5.5]))
    np.testing.assert_allclose(state_means, np.array([3.0, 4.0]))


def test_returns_float64_arrays_with_expected_shapes() -> None:
    action_means, state_means = compute_score_means(
        [
            [1, 2],
            [3, 4],
            [5, 6],
        ]
    )

    assert isinstance(action_means, np.ndarray)
    assert isinstance(state_means, np.ndarray)
    assert action_means.dtype == np.float64
    assert state_means.dtype == np.float64
    assert action_means.shape == (2,)
    assert state_means.shape == (3,)


def test_does_not_modify_input_table() -> None:
    score_table = [[1.0, 2.0], [3.0, 4.0]]

    compute_score_means(score_table)

    assert score_table == [[1.0, 2.0], [3.0, 4.0]]
