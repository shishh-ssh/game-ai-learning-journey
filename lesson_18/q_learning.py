"""在 LineWorld 中运行 Q-learning 训练。"""

import torch

from lesson_15.q_table import update_q_value
from lesson_16.torch_epsilon_policy import select_action
from lesson_17.line_world_env import LineWorldEnv


def create_q_table(environment: LineWorldEnv) -> torch.Tensor:
    """根据环境的状态数和动作数创建零初始化 Q 表。"""
    return torch.zeros(
        (
            environment.observation_space.n,
            environment.action_space.n,
        ),
        dtype=torch.float64,
    )


def train_one_episode(
    environment: LineWorldEnv,
    q_table: torch.Tensor,
    epsilon: float,
    gamma: float,
    alpha: float,
    generator: torch.Generator,
) -> tuple[float, int, bool, bool]:
    """训练一个 episode，原地更新 Q 表并返回本局结果。"""
    state, _ = environment.reset()
    episode_return = 0.0
    steps = 0
    terminated = False
    truncated = False
    episode_done = False
    while not episode_done:
        action = select_action(q_table, state, epsilon, generator)
        next_state, reward, terminated, truncated, _ = environment.step(action)
        target_done = terminated
        update_q_value(
            q_table,
            state,
            action,
            reward,
            next_state,
            target_done,
            gamma,
            alpha,
        )
        episode_return += reward
        steps += 1
        state = next_state
        episode_done = terminated or truncated
    return episode_return, steps, terminated, truncated


def train_q_learning(
    environment: LineWorldEnv,
    episodes: int,
    epsilon: float,
    gamma: float,
    alpha: float,
    seed: int,
    epsilon_min: float=0.0,
    epsilon_decay: float=1.0,
) -> tuple[torch.Tensor, list[float], list[int]]:
    """训练多个 episode，返回 Q 表、回报历史和步数历史。"""
    if episodes < 1:
        raise ValueError("episodes必须大于等于1")
    generator = torch.Generator().manual_seed(seed)
    return_history = []
    steps_history = []
    q_table = create_q_table(environment)
    for _ in range(episodes):
        episode_return = 0.0
        steps = 0
        terminated = False
        truncated = False
        episode_done = False
        state, _ = environment.reset()
        while not episode_done:
            action = select_action(q_table=q_table, state=state, epsilon=epsilon, generator=generator)
            next_state, reward, terminated, truncated, _= environment.step(action)
            update_q_value(
                q_table=q_table,
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=terminated,
                gamma=gamma,
                alpha=alpha,
                
            )
            episode_return += reward
            steps += 1
            state = next_state
            episode_done = truncated or terminated
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        return_history.append(episode_return)
        steps_history.append(steps)
    return q_table, return_history, steps_history


def evaluate_q_table(
    environment: LineWorldEnv,
    q_table: torch.Tensor,
    episodes: int,
) -> dict[str, float]:
    """使用 greedy 策略独立评估训练后的 Q 表。"""
    if episodes < 1:
        raise ValueError("episodes必须大于等于1")
    success_count = 0
    total_return = 0.0
    total_steps = 0
    success_step = 0
    for _ in range(episodes):
        state, _ = environment.reset()
        episode_return = 0.0
        steps = 0
        terminated = False
        truncated = False
        episode_done = False
        while not episode_done:
            action = torch.argmax(q_table[state]).item()
            next_state, reward, terminated, truncated, _ = environment.step(action)
            episode_return += reward
            steps += 1
            state = next_state
            episode_done = terminated or truncated
        if  terminated:
            success_count += 1
            success_step += steps
        total_return += episode_return
        total_steps += steps
    if success_count == 0:
        average_success_step = 0.0
    else:
        average_success_step = success_step/success_count
    return {
        "success_rate": success_count/episodes,
        "average_return": total_return/episodes,
        "average_steps": total_steps/episodes,
        "average_success_steps": average_success_step,
    }


def run_multi_seed_experiment(
        seeds: list[int],
) -> list[dict[str, float | int]]:
    """使用不同的随机种子运行多次实验并返回每次实验的评估指标。"""
    results = []

    for seed in seeds:
        training_environment = LineWorldEnv(
            goal_state=4,
            max_steps=10,
        )

        q_table, return_history, steps_history = train_q_learning(
            environment=training_environment,
            episodes=500,
            epsilon=1.0,
            gamma=0.9,
            alpha=0.5,
            seed=seed,
            epsilon_decay=0.98,
            epsilon_min=0.05,
        )
        evaluation_environment = LineWorldEnv(
            goal_state=4,
            max_steps=10,
        )

        metrics = evaluate_q_table(
            environment=evaluation_environment,
            q_table=q_table,
            episodes=100,
        )

        metrics["seed"] = seed
        results.append(metrics)
    return results       


def calculate_metric_means(
    results: list[dict[str, float | int]],
) -> dict[str, float]:
    if results == []:
        raise ValueError("results不能为空")
    
    metric_names = {
        "success_rate": 0.0,
        "average_return": 0.0,
        "average_steps": 0.0,
        }
    for result in results:
        for metric in metric_names:
            metric_names[metric] += result[metric]
    for metric in metric_names:
        metric_names[metric] /= len(results)
    return metric_names

def calculate_metric_stds(
    results: list[dict[str, float | int]],
) -> dict[str, float]:
    if len(results) < 2 :
        raise ValueError("results至少有2个")
    metric_means = calculate_metric_means(results)
    squared_difference_sums = {
            "success_rate": 0.0,
            "average_return": 0.0,
            "average_steps": 0.0,
            }
    
    for result in results:
        for metric in metric_means:
            squared_difference_sums[metric] += (result[metric] - metric_means[metric]) ** 2
    for metric in squared_difference_sums:
        squared_difference_sums[metric] /= (len(results) - 1)
        squared_difference_sums[metric] = squared_difference_sums[metric] ** 0.5
    return squared_difference_sums
    
