import pytest

from lesson_09.linear_training import train_linear_once


def test_single_sample_matches_manual_gradient() -> None:
    loss, weight, bias = train_linear_once([1.0], [2.0], 0.1)

    assert loss == pytest.approx(4.0)
    assert weight == pytest.approx(0.4)
    assert bias == pytest.approx(0.4)


def test_input_value_scales_weight_gradient() -> None:
    loss, weight, bias = train_linear_once([2.0], [4.0], 0.1)

    assert loss == pytest.approx(16.0)
    assert weight == pytest.approx(1.6)
    assert bias == pytest.approx(0.8)


def test_batch_uses_mean_squared_error() -> None:
    loss, weight, bias = train_linear_once([1.0, 2.0], [2.0, 4.0], 0.1)

    assert loss == pytest.approx(10.0)
    assert weight == pytest.approx(1.0)
    assert bias == pytest.approx(0.6)


def test_zero_targets_leave_zero_parameters() -> None:
    loss, weight, bias = train_linear_once([1.0, 3.0], [0.0, 0.0], 0.2)

    assert loss == pytest.approx(0.0)
    assert weight == pytest.approx(0.0)
    assert bias == pytest.approx(0.0)


def test_returns_plain_floats_and_does_not_modify_inputs() -> None:
    inputs = [1.0, 2.0]
    targets = [2.0, 4.0]
    original_inputs = inputs.copy()
    original_targets = targets.copy()

    result = train_linear_once(inputs, targets, 0.1)

    assert all(type(item) is float for item in result)
    assert inputs == original_inputs
    assert targets == original_targets
