from multi_epoch_training import train_linear_many


def test_loss_history_and_parameters_improve():
    losses, validation_loss, weight, bias = train_linear_many(
        [1.0, 2.0, 3.0, 4.0],
        [3.0, 5.0, 7.0, 9.0],
        [5.0, 6.0],
        [11.0, 13.0],
        learning_rate=0.01,
        epochs=100,
    )

    assert len(losses) == 100
    assert losses[-1] < losses[0]
    assert validation_loss < 0.1
    assert abs(weight - 2.0) < 0.2
    assert abs(bias - 1.0) < 0.3
