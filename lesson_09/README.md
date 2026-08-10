# 第 9 课：线性模型、损失函数与优化器

本课把上一课的手工参数更新扩展为 PyTorch 标准训练步骤：`torch.nn.Linear` 管理权重和偏置，`torch.nn.MSELoss` 衡量预测误差，`torch.optim.SGD` 清零并更新全部模型参数。

## 当前任务：执行一次线性模型训练

只修改 `lesson_09/linear_training.py` 中 `train_linear_once` 的函数体：

```python
def train_linear_once(
    inputs: list[float],
    targets: list[float],
    learning_rate: float,
) -> tuple[float, float, float]:
```

输入保证非空且长度相同。每个列表元素代表一个单特征样本或对应目标。

要求：

1. 创建 `torch.nn.Linear(1, 1)`。
2. 在 `torch.no_grad()` 中使用 `copy_()` 将 `weight` 和 `bias` 都设为 `0.0`。
3. 将 `inputs` 和 `targets` 分别转换为 `torch.float32` 张量，并 reshape 为 `(-1, 1)`。
4. 创建 `torch.nn.MSELoss()`。
5. 创建 `torch.optim.SGD(layer.parameters(), lr=learning_rate)`。
6. 严格执行：`zero_grad -> 前向预测 -> 计算损失 -> backward -> step`。
7. 在 `step()` 前保存更新前的 `loss.item()`。
8. 返回三个 Python `float`：`(更新前损失, 更新后的 weight, 更新后的 bias)`。
9. 不修改输入，不修改导入、签名或测试。

参数都是单元素张量，可以使用：

```python
layer.weight.item()
layer.bias.item()
```

验证命令：

```powershell
D:\anaconda\envs\rl\python.exe -m pytest lesson_09 -q -p no:cacheprovider
```

当前骨架预期为 `5 failed`，正确实现后应为 `5 passed`。

## 结课验收

- 定向测试：`5/5` 通过。
- 全仓回归：`178/178` 通过。
- 代码解释：通过；已纠正样本数与特征数、参数初始化与梯度清零、`backward()` 与 `step()` 的职责混淆。
- 独立实操：完成从零参数开始的一次线性模型训练，正确返回更新前损失和更新后参数。
- 现场修改：通过；修正参数读取时机，使损失在更新前保存、权重和偏置在更新后读取。
- 当前状态：本课已掌握；下一课建立多轮线性回归训练并观察损失变化，暂不进入 Q-learning。
