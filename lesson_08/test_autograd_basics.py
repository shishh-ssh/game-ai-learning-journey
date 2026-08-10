import pytest

from lesson_08.autograd_basics import square_gradient_step


def test_positive_value_moves_toward_zero() -> None:
    loss, gradient, updated = square_gradient_step(4.0, 0.25)

    assert loss == pytest.approx(16.0)
    assert gradient == pytest.approx(8.0)
    assert updated == pytest.approx(2.0)


def test_negative_value_moves_toward_zero() -> None:
    loss, gradient, updated = square_gradient_step(-3.0, 0.1)

    assert loss == pytest.approx(9.0)
    assert gradient == pytest.approx(-6.0)
    assert updated == pytest.approx(-2.4)


def test_zero_has_zero_loss_and_gradient() -> None:
    loss, gradient, updated = square_gradient_step(0.0, 0.3)

    assert loss == pytest.approx(0.0)
    assert gradient == pytest.approx(0.0)
    assert updated == pytest.approx(0.0)


def test_zero_learning_rate_does_not_change_value() -> None:
    loss, gradient, updated = square_gradient_step(2.5, 0.0)

    assert loss == pytest.approx(6.25)
    assert gradient == pytest.approx(5.0)
    assert updated == pytest.approx(2.5)


def test_returns_plain_python_floats() -> None:
    result = square_gradient_step(1.0, 0.1)

    assert len(result) == 3
    assert all(type(item) is float for item in result)
