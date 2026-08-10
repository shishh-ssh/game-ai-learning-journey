import pytest

from lesson_08.autograd_basics import train_square_parameter


def test_records_loss_before_each_update() -> None:
    losses, final_value = train_square_parameter(4.0, 0.25, 2)

    assert losses == pytest.approx([16.0, 4.0])
    assert final_value == pytest.approx(1.0)


def test_multiple_steps_use_updated_parameter() -> None:
    losses, final_value = train_square_parameter(3.0, 0.1, 3)

    assert losses == pytest.approx([9.0, 5.76, 3.6864])
    assert final_value == pytest.approx(1.536)


def test_negative_value_moves_toward_zero() -> None:
    losses, final_value = train_square_parameter(-2.0, 0.1, 2)

    assert losses == pytest.approx([4.0, 2.56])
    assert final_value == pytest.approx(-1.28)


def test_zero_steps_returns_initial_value_and_empty_history() -> None:
    losses, final_value = train_square_parameter(5.0, 0.2, 0)

    assert losses == []
    assert final_value == pytest.approx(5.0)


def test_returns_plain_python_values() -> None:
    losses, final_value = train_square_parameter(1.0, 0.1, 2)

    assert all(type(loss) is float for loss in losses)
    assert type(final_value) is float
