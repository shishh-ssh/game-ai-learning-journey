"""运行 LineWorld 单个 episode 的交互循环。"""

from baselines import AlwaysRightPolicy, RandomPolicy
from line_world_env import LineWorldEnv


def run_episode(
    environment: LineWorldEnv,
    policy: AlwaysRightPolicy | RandomPolicy,
) -> tuple[float, int, bool, bool]:
    """运行一个 episode，返回累计回报、步数和结束原因。"""
    observation, _ = environment.reset()
    episode_return = 0.0
    steps = 0
    terminated = False
    truncated = False
    done = False
    while not done:
        action = policy.select_action(observation)
        observation, reward, terminated, truncated, _ = environment.step(action)
        episode_return += reward
        steps += 1
        done = terminated or truncated
    
    return episode_return, steps, terminated, truncated
