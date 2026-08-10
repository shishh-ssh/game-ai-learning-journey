# 第 7 课：PyTorch 张量基础

本课从 NumPy 过渡到 PyTorch，只学习 CPU 张量的创建、`dtype`、`shape` 和 batch 维度。暂不进入 GPU、自动求导、神经网络、Q-learning 或训练循环。

## 当前任务：状态列表转 batch 张量

修改 `lesson_07/tensor_basics.py` 中 `state_to_tensor` 的函数体：

```python
def state_to_tensor(state: list[float]) -> torch.Tensor:
```

要求：

1. 将输入 `state` 转换为 `torch.float32` 张量。
2. 返回形状必须是 `torch.Size([1, len(state)])`。
3. 不修改输入列表。
4. 不使用 NumPy。
5. 不修改导入、函数签名或测试。

可以使用：

```python
torch.tensor(..., dtype=torch.float32)
reshape(1, -1)
```

验证命令：

```powershell
conda run -n rl python -m pytest lesson_07 -q -p no:cacheprovider
```

当前骨架预期为 `5 failed`。正确实现后应为 `5 passed`。

## 第二个任务：按状态选择最高 Q 值动作

只修改 `select_greedy_actions` 的函数体：

```python
def select_greedy_actions(q_values: list[list[float]]) -> torch.Tensor:
```

要求：

1. 将完整二维列表转换为 `torch.float32` 张量。
2. 使用 `argmax(dim=1)` 为每个状态选择一个动作索引。
3. 返回一维 `torch.int64` 张量，长度等于状态数。
4. 不修改输入，不使用 NumPy，不修改已有函数或测试。

定向验证：

```powershell
D:\anaconda\envs\rl\python.exe -m pytest lesson_07/test_greedy_actions.py -q -p no:cacheprovider
```

当前新练习预期为 `5 failed`；正确实现后第 7 课应为 `10 passed`。

## 结课验收

- 定向测试：`10/10` 通过。
- 全仓回归：`163/163` 通过。
- 代码解释：通过；能够区分状态数、动作数、batch 维度、Q 值和动作索引。
- 现场修改：通过；将 `argmax` 结果保存为语义明确的 `best_actions`。
- 当前状态：本课已掌握；下一课学习张量运算与自动求导，暂不进入 Q-learning。
