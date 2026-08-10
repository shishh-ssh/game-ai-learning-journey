# 第 6 课：NumPy 随机采样与 epsilon-greedy 决策

本课把第 5 课的 `argmax` 确定性策略扩展为强化学习常用的探索/利用决策。以概率 `epsilon` 随机探索，以概率 `1 - epsilon` 选择最高分动作；`epsilon=0` 完全利用，`epsilon=1` 完全探索。

本课只使用 NumPy，不引入 PyTorch、Q-learning、神经网络或训练循环。

## 学习目标

- 理解随机数生成器、随机种子和可复现性。
- 能解释探索与利用的区别。
- 能预测 `epsilon=0`、`epsilon=1` 及固定种子下的行为。
- 能区分动作索引与动作分数，并保持一维返回形状。

## 计划接口

```python
def choose_epsilon_greedy(
    score_table: list[list[float]],
    epsilon: float,
    rng: np.random.Generator,
) -> np.ndarray:
```

输入为非空、等长二维分数表；返回每个状态一个动作索引的一维 `np.ndarray`。必须使用传入的 `rng`，不能依赖全局随机状态；本课暂不增加非法输入校验。

## 当前任务

只修改 `lesson_06/epsilon_policy.py` 中 `choose_epsilon_greedy` 的函数体：

1. 把完整分数表转换为 `np.float64` 二维数组。
2. 使用 `rng.random(状态数) < epsilon` 生成探索掩码。
3. 使用 `rng.integers(动作数, size=状态数)` 生成随机动作。
4. 使用 `argmax(axis=1)` 生成利用动作。
5. 使用 `np.where` 按掩码合并并返回一维动作索引数组。

随机调用的顺序是接口的一部分：先一次生成全部 `roll`，再一次生成全部随机动作。不要逐行交错调用随机数，不使用全局 `np.random`，不修改输入、函数签名、导入或测试。

定向验证：

```powershell
python -m pytest lesson_06/test_epsilon_policy.py -q -p no:cacheprovider
```

当前骨架预期为 `5 failed`。正确实现后应为 `5 passed`，随后再运行全仓回归。

## 结课验收

- 定向测试：`5/5` 通过。
- 全仓回归：`153/153` 通过。
- 代码解释：通过；能够说明探索掩码、状态数与动作数、随机调用顺序及 `np.where` 的逐状态选择。
- 现场修改：通过；将 `epsilon_mask` 重命名为语义更准确的 `explore_mask`，行为保持不变。
- 当前状态：本课已掌握；下一课进入 PyTorch 张量基础。
