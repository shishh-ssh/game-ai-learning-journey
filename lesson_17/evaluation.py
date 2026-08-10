"""评估 LineWorld 基线策略的多局表现。"""

from baselines import AlwaysRightPolicy, RandomPolicy
from episode import run_episode
from line_world_env import LineWorldEnv


def evaluate_policy(
    environment: LineWorldEnv,
    policy: AlwaysRightPolicy | RandomPolicy,
    episodes: int,
) -> dict[str, float]:
    """运行多局并返回成功率、平均回报和平均步数。"""
    if episodes < 1:
        raise ValueError("episodes必须大于等于1")
    success_count = 0
    total_return = 0.0
    total_steps = 0
    for _ in range(episodes):
        episode_return, steps, terminated, truncated = run_episode(environment=environment, policy=policy)
        total_return += episode_return
        total_steps += steps
        if terminated:
            success_count += 1
    return {
        "success_rate": success_count/episodes,
        "average_return": total_return/episodes,
        "average_steps": total_steps/episodes,
    }
