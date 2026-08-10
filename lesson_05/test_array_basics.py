import numpy as np

from array_basics import scale_rewards


def test_returns_float64_numpy_array() -> None:
    result = scale_rewards([1, 2, 3], 2)

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64


def test_scales_every_reward() -> None:
    result = scale_rewards([1.0, -2.0, 0.5], 1.5)

    np.testing.assert_allclose(result, np.array([1.5, -3.0, 0.75]))


def test_preserves_one_dimensional_shape() -> None:
    result = scale_rewards([3.0, 4.0, 5.0, 6.0], 0.5)

    assert result.shape == (4,)
    np.testing.assert_allclose(result, np.array([1.5, 2.0, 2.5, 3.0]))


def test_empty_rewards_return_empty_float64_array() -> None:
    result = scale_rewards([], 3.0)

    assert result.shape == (0,)
    assert result.dtype == np.float64


def test_does_not_modify_input_list() -> None:
    rewards = [1.0, 2.0, 3.0]

    scale_rewards(rewards, 10.0)

    assert rewards == [1.0, 2.0, 3.0]
