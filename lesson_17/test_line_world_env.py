import pytest
from gymnasium.utils.env_checker import check_env

from line_world_env import LineWorldEnv


def test_environment_passes_gymnasium_checker():
    check_env(LineWorldEnv())


def test_reset_returns_start_observation_and_info():
    environment = LineWorldEnv(goal_state=4)

    observation, info = environment.reset(seed=42)

    assert observation == 0
    assert info == {"distance_to_goal": 4}
    assert environment.observation_space.contains(observation)


def test_moves_right_and_terminates_at_goal():
    environment = LineWorldEnv(goal_state=2)
    environment.reset()

    first = environment.step(LineWorldEnv.RIGHT)
    second = environment.step(LineWorldEnv.RIGHT)

    assert first == (1, -1.0, False, False, {"distance_to_goal": 1})
    assert second == (2, 10.0, True, False, {"distance_to_goal": 0})


def test_left_boundary_keeps_agent_at_zero():
    environment = LineWorldEnv()
    environment.reset()

    result = environment.step(LineWorldEnv.LEFT)

    assert result == (0, -1.0, False, False, {"distance_to_goal": 4})


def test_invalid_action_raises_value_error():
    environment = LineWorldEnv()
    environment.reset()

    with pytest.raises(ValueError, match="action"):
        environment.step(2)


def test_time_limit_returns_truncated_without_termination():
    environment = LineWorldEnv(goal_state=4, max_steps=1)
    environment.reset()

    observation, reward, terminated, truncated, info = environment.step(
        LineWorldEnv.LEFT
    )

    assert observation == 0
    assert reward == -1.0
    assert terminated is False
    assert truncated is True
    assert info["distance_to_goal"] == 4


def test_reaching_goal_at_time_limit_is_terminated_not_truncated():
    environment = LineWorldEnv(goal_state=1, max_steps=1)
    environment.reset()

    _, _, terminated, truncated, _ = environment.step(LineWorldEnv.RIGHT)

    assert terminated is True
    assert truncated is False
