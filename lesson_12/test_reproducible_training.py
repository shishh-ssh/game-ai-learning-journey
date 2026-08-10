from reproducible_training import train_reproducible


def test_same_seed_reproduces_training():
    arguments = (
        [1.0, 2.0, 3.0, 4.0],
        [3.0, 5.0, 7.0, 9.0],
        0.01,
        100,
    )

    first = train_reproducible(*arguments, seed=42)
    second = train_reproducible(*arguments, seed=42)

    assert first == second


def test_different_seed_changes_training_result():
    arguments = (
        [1.0, 2.0, 3.0, 4.0],
        [3.0, 5.0, 7.0, 9.0],
        0.01,
        1,
    )

    first = train_reproducible(*arguments, seed=42)
    second = train_reproducible(*arguments, seed=7)

    assert first != second
