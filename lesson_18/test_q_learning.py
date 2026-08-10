from unittest.mock import patch

import torch
import pytest

from lesson_17.line_world_env import LineWorldEnv
from q_learning import (
    calculate_metric_means,
    calculate_metric_stds,
    evaluate_q_table,
    run_multi_seed_experiment,
    train_one_episode,
    train_q_learning,
)


def test_train_one_episode_updates_visited_pair_and_reaches_goal():
    environment = LineWorldEnv(goal_state=1, max_steps=1)
    q_table = torch.tensor(
        [[0.0, 1.0], [0.0, 0.0]],
        dtype=torch.float64,
    )
    generator = torch.Generator().manual_seed(42)

    result = train_one_episode(
        environment,
        q_table,
        epsilon=0.0,
        gamma=0.9,
        alpha=0.5,
        generator=generator,
    )

    assert result == (10.0, 1, True, False)
    assert torch.equal(
        q_table,
        torch.tensor([[0.0, 5.5], [0.0, 0.0]], dtype=torch.float64),
    )


def test_truncation_stops_episode_but_keeps_bootstrap_value():
    environment = LineWorldEnv(goal_state=2, max_steps=1)
    q_table = torch.tensor(
        [[0.0, 1.0], [2.0, 5.0], [0.0, 0.0]],
        dtype=torch.float64,
    )
    generator = torch.Generator().manual_seed(42)

    result = train_one_episode(
        environment,
        q_table,
        epsilon=0.0,
        gamma=0.9,
        alpha=1.0,
        generator=generator,
    )

    assert result == (-1.0, 1, False, True)
    assert q_table[0, LineWorldEnv.RIGHT].item() == 3.5


def test_train_q_learning_records_each_episode():
    environment = LineWorldEnv(goal_state=1, max_steps=3)

    q_table, return_history, steps_history = train_q_learning(
        environment,
        episodes=3,
        epsilon=0.0,
        gamma=0.9,
        alpha=1.0,
        seed=42,
    )

    assert return_history == [9.0, 10.0, 10.0]
    assert steps_history == [2, 1, 1]
    assert q_table[0, LineWorldEnv.RIGHT].item() == 10.0


def test_epsilon_decays_between_episodes_not_between_steps():
    environment = LineWorldEnv(goal_state=2, max_steps=3)
    observed_epsilons = []

    def select_right(q_table, state, epsilon, generator):
        observed_epsilons.append(epsilon)
        return LineWorldEnv.RIGHT

    with patch("q_learning.select_action", side_effect=select_right):
        train_q_learning(
            environment,
            episodes=3,
            epsilon=1.0,
            gamma=0.9,
            alpha=0.5,
            seed=42,
            epsilon_decay=0.5,
            epsilon_min=0.25,
        )

    assert observed_epsilons == [1.0, 1.0, 0.5, 0.5, 0.25, 0.25]


def test_evaluate_q_table_is_greedy_and_does_not_modify_q_table():
    environment = LineWorldEnv(goal_state=2, max_steps=3)
    q_table = torch.tensor(
        [[0.0, 3.0], [0.0, 5.0], [0.0, 0.0]],
        dtype=torch.float64,
    )
    original = q_table.clone()

    metrics = evaluate_q_table(environment, q_table, episodes=2)

    assert metrics == {
        "success_rate": 1.0,
        "average_return": 9.0,
        "average_steps": 2.0,
        "average_success_steps": 2.0,
    }
    assert torch.equal(q_table, original)


def test_multi_seed_experiment_returns_one_result_per_seed():
    results = run_multi_seed_experiment([1, 7, 42])

    assert [result["seed"] for result in results] == [1, 7, 42]
    assert all(result["success_rate"] == 1.0 for result in results)
    assert all(result["average_return"] == 7.0 for result in results)
    assert all(result["average_steps"] == 4.0 for result in results)


def test_metric_summary_uses_mean_and_sample_standard_deviation():
    results = [
        {"success_rate": 1.0, "average_return": 7.0, "average_steps": 4.0},
        {"success_rate": 1.0, "average_return": 5.0, "average_steps": 6.0},
        {"success_rate": 0.0, "average_return": 0.0, "average_steps": 8.0},
    ]

    means = calculate_metric_means(results)
    stds = calculate_metric_stds(results)

    assert means == pytest.approx(
        {
            "success_rate": 2.0 / 3.0,
            "average_return": 4.0,
            "average_steps": 6.0,
        }
    )
    assert stds == pytest.approx(
        {
            "success_rate": 0.5773502691896258,
            "average_return": 3.605551275463989,
            "average_steps": 2.0,
        }
    )
