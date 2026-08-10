# 第 17 课：环境建模与 Game AI Benchmark v0.1

本课使用当前主流的 Gymnasium 自定义环境接口，不再继续使用旧版 Gym 的单个 `done` 返回值。

查询日期：2026-08-06

- Gymnasium 版本：`1.3.0`
- 官方自定义环境指南：https://gymnasium.farama.org/introduction/create_custom_env/
- 官方 Env API：https://gymnasium.farama.org/api/env/

## 学习目标

完成一个继承 `gymnasium.Env` 的一维 LineWorld 环境，并理解环境与智能体之间的职责边界。本课不以“接口能运行”为终点，而是第一个项目里程碑：环境、基线、指标、测试和失败案例必须组成完整评测闭环。

```text
状态：0 - 1 - 2 - 3 - 4
起点：0
终点：4
动作0：left
动作1：right
普通移动奖励：-1
到达终点奖励：10
```

## 当前主流接口

环境必须定义状态空间和动作空间：

```python
self.observation_space = gym.spaces.Discrete(5)
self.action_space = gym.spaces.Discrete(2)
```

`reset()` 开始一个新 episode：

```python
def reset(self, *, seed=None, options=None):
    super().reset(seed=seed)
    self.state = 0
    return self.state, {}
```

返回：

```text
(observation, info)
```

`step()` 执行一个动作：

```text
(observation, reward, terminated, truncated, info)
```

- `terminated`：到达任务本身定义的终止条件，例如到达目标或角色死亡；
- `truncated`：因任务外限制提前截断，例如超过最大步数；
- `info`：不属于观测但有助于调试、统计或评测的额外信息。

训练循环通常使用：

```python
done = terminated or truncated
```

但计算 Q-learning 或 DQN 的未来价值目标时，必须区分两种结束原因。本课先完成环境接口；终止与截断对 TD target 的影响将在训练循环课中单独推导。

## 为什么不使用旧接口

旧版代码常见：

```python
next_state, reward, done, info = env.step(action)
```

单个 `done` 无法区分“任务真正结束”和“达到时间限制”。Gymnasium 将其拆成 `terminated` 与 `truncated`，可以避免错误清除本应保留的未来价值。

## 实现与验收顺序

1. 继承 `gymnasium.Env`；
2. 定义 `observation_space` 与 `action_space`；
3. 在 `reset()` 第一行调用 `super().reset(seed=seed)`；
4. 实现边界移动、奖励和 `terminated`；
5. 先令 `truncated=False`；
6. 使用 `gymnasium.utils.env_checker.check_env()` 检查接口；
7. 加入最大步数限制并测试 `truncated`；
8. 实现随机策略和“始终向右”规则策略；
9. 分别运行至少 3 个随机种子，统计成功率、平均回报和平均步数；
10. 记录至少一个失败案例，并完成一次规则现场修改。

实现文件为 `line_world_env.py`。测试同时覆盖 Gymnasium 官方环境检查器、重置、移动、边界、终止奖励和非法动作。

## 本课必须回答的问题

- observation 是否包含做出最优决策所需的信息，是否发生信息泄露？
- 边界动作是无效、原地停留还是受惩罚，为什么？
- 到达目标是 `terminated`，超过步数是 `truncated`，二者如何影响训练循环？
- 为什么 episode 结束条件通常是 `terminated or truncated`，但 TD target 的 bootstrap mask 不能机械地使用同一个值？
- 随机基线、规则基线和学习策略分别验证什么？

完整的第 17 课以后路线见 [第 17 章及以后：项目驱动加速课程](../CURRICULUM_17_PLUS.md)。

## 当前编码任务：基线策略

只修改 `baselines.py` 中两个 `select_action` 方法的函数体：

- `AlwaysRightPolicy` 对任何观察都返回 `LineWorldEnv.RIGHT`；
- `RandomPolicy` 使用构造函数已经创建的独立 `random_generator`；
- 随机策略只能返回 `LineWorldEnv.LEFT` 或 `LineWorldEnv.RIGHT`；
- 不得在 `select_action` 内重新设置种子；
- 不修改类名、方法签名、已有导入或测试。

定向验证：

```powershell
D:\anaconda\envs\rl\python.exe -m pytest lesson_17/test_baselines.py -q -p no:cacheprovider
```

完成后还需要进行代码解释和一次现场修改，才能通过基线策略验收。

## 当前编码任务：单个 episode 交互

在 `episode.py` 中实现 `run_episode`，返回：

```text
(episode_return, steps, terminated, truncated)
```

本任务只负责运行一局，不进行多局统计，不更新策略，也不实现 Q-learning。

## 当前编码任务：多局基线评测

在 `evaluation.py` 中实现 `evaluate_policy`，重复调用 `run_episode`，返回：

```python
{
    "success_rate": ...,
    "average_return": ...,
    "average_steps": ...,
}
```

本任务只做基础汇总，不在代码中实现绘图、标准差或统计检验。
