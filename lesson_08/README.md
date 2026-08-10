# 第 8 课：PyTorch 自动求导与单步更新

本课学习 `requires_grad`、`backward()`、`.grad`、梯度累加、`torch.no_grad()` 和一次最小参数更新。暂不引入优化器、神经网络或 Q-learning。

## 当前任务：对平方损失执行一次梯度下降

修改 `lesson_08/autograd_basics.py` 中 `square_gradient_step` 的函数体：

```python
def square_gradient_step(
    value: float,
    learning_rate: float,
) -> tuple[float, float, float]:
```

要求：

1. 创建值为 `value`、类型为 `torch.float32` 且 `requires_grad=True` 的标量张量 `x`。
2. 计算 `loss = x ** 2`。
3. 调用 `loss.backward()`。
4. 在 `torch.no_grad()` 中执行 `x -= learning_rate * x.grad`。
5. 按 `(更新前的损失, 本次梯度, 更新后的 x)` 顺序返回三个 Python `float`。
6. 在更新参数前保存损失和梯度；不要在函数中执行第二次反向传播。

张量标量可以使用 `.item()` 转成 Python `float`。

验证命令：

```powershell
D:\anaconda\envs\rl\python.exe -m pytest lesson_08 -q -p no:cacheprovider
```

当前骨架预期为 `5 failed`，正确实现后应为 `5 passed`。

## 第二个任务：多轮梯度下降

只修改 `train_square_parameter` 的函数体：

```python
def train_square_parameter(
    value: float,
    learning_rate: float,
    steps: int,
) -> tuple[list[float], float]:
```

要求：

1. 在循环外创建一个 `torch.float32`、`requires_grad=True` 的标量张量 `x`。
2. 创建空的 Python 列表 `losses`。
3. 重复 `steps` 次：计算 `loss=x**2`，将更新前的 `loss.item()` 追加到列表，调用 `backward()`，在 `torch.no_grad()` 中更新 `x`，最后用 `x.grad.zero_()` 清零。
4. 返回 `(losses, x.item())`。
5. 每轮顺序必须是前向、记录损失、反向、更新、清零。
6. 不修改已有函数、导入、签名或测试，不使用优化器。

定向验证：

```powershell
D:\anaconda\envs\rl\python.exe -m pytest lesson_08/test_training_loop.py -q -p no:cacheprovider
```

当前新练习预期为 `5 failed`；正确实现后第 8 课应为 `10 passed`。

## 结课验收

- 定向测试：`10/10` 通过。
- 全仓回归：`173/173` 通过。
- 代码解释：通过；能够说明前向计算、`backward()` 的副作用、`.item()`、`torch.no_grad()`、梯度更新与清零顺序。
- 独立实操：完成单步梯度下降和复用同一参数的多轮训练循环，正确处理 `steps=0`。
- 现场修改：通过；修正返回类型、保存时机、循环边界、损失历史和零轮变量生命周期。
- 当前状态：本课已掌握；下一课学习 `torch.nn` 模型、损失函数与优化器，暂不进入 Q-learning。
