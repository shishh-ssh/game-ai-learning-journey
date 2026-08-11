# 第 19 课：两层 MLP 分类器（Day 1）

本课先把单层线性回归扩展为一个真正的神经网络模块。今天只学习模型结构和张量形状，不训练数据、不接游戏环境。

## 目标

实现：

```text
[batch, input_dim]
    -> Linear(input_dim, hidden_dim)
    -> ReLU
    -> Linear(hidden_dim, num_classes)
    -> [batch, num_classes] logits
```

`MLPClassifier` 必须继承 `torch.nn.Module`，并在 `__init__` 中调用 `super().__init__()`，这样两层 `Linear` 的参数才会被注册到 `model.parameters()` 中。

## 今日任务

只修改 `mlp_classifier.py` 中标记为 TODO 的部分：

1. 创建隐藏层 `Linear(input_dim, hidden_dim)`；
2. 创建 `ReLU`；
3. 创建输出层 `Linear(hidden_dim, num_classes)`；
4. 在 `forward` 中按“隐藏层 -> 激活 -> 输出层”的顺序计算；
5. 直接返回输出层结果，不要在模型内部调用 softmax。

例如 `input_dim=2、hidden_dim=8、num_classes=2` 时：

```text
输入 [4, 2] -> 隐藏层 [4, 8] -> 输出 [4, 2]
```

最后一个维度是每个样本对应各类别的 logits，尚不是概率。

## 验证

在 Conda `rl` 环境中运行：

```powershell
conda activate rl
python -m pytest lesson_19/test_mlp_classifier.py -q -p no:cacheprovider
```

骨架预期先失败；完成今日任务后应为 `5 passed`。不要修改测试。

## 今日口述

完成测试后，用自己的话回答：为什么 `nn.Module` 需要 `super().__init__()`？输入 batch 形状为 `[B, 2]` 时，两层 Linear 后的形状分别是什么？

## Day 2：构造最小 XOR 数据

XOR 有两个输入特征。两个输入不相同时标签为 `1`，相同时标签为 `0`：

```text
[0, 0] -> 0
[0, 1] -> 1
[1, 0] -> 1
[1, 1] -> 0
```

只修改 `nonlinear_data.py` 中的 `make_xor_data`：

1. `inputs` 使用 `torch.float32`，形状为 `[4, 2]`；
2. `labels` 使用 `torch.int64`，形状为 `[4]`；
3. 样本和标签严格按照上面的顺序；
4. 返回 `(inputs, labels)`，不要把标签 reshape 成 `[4, 1]`。

验证命令：

```powershell
& 'D:\anaconda\envs\rl\python.exe' -m pytest lesson_19/test_nonlinear_data.py -q -p no:cacheprovider
```

骨架预期先失败；实现完成后应为 `4 passed`。

## Day 3：logits 与 CrossEntropyLoss

当前模型对每个样本输出两个 logits，形状为 `[batch, 2]`。标签是一维类别索引，形状为 `[batch]`。

只修改 `classification_loss.py` 中的 `forward_and_loss`：

1. 调用 `model(inputs)` 得到 logits；
2. 创建 `torch.nn.CrossEntropyLoss()`；
3. 直接把 logits 和 labels 交给损失函数；
4. 返回 `(logits, loss)`；
5. 不要手动调用 softmax，不要在这个函数中执行 `backward()` 或更新参数。

验证命令：

```powershell
& 'D:\anaconda\envs\rl\python.exe' -m pytest lesson_19/test_classification_loss.py -q -p no:cacheprovider
```

骨架预期先失败；实现完成后应为 `4 passed`。

## Day 4：完成一个训练 step

只修改 `training_step.py` 中的 `train_one_step`，按以下顺序执行：

```text
zero_grad
-> forward
-> CrossEntropyLoss
-> backward
-> step
```

要求：

1. 调用 `model.train()`；
2. 使用传入的 optimizer 清空旧梯度；
3. 调用模型得到 logits；
4. 使用 `CrossEntropyLoss` 计算 loss；
5. 在更新前保存 `loss.item()`；
6. 执行反向传播和参数更新；
7. 返回更新前的 loss，类型为 Python `float`。

验证命令：

```powershell
& 'D:\anaconda\envs\rl\python.exe' -m pytest lesson_19/test_training_step.py -q -p no:cacheprovider
```

骨架预期先失败；实现完成后应为 `5 passed`。

## Day 5：重复训练并记录 loss

只修改 `train_loop.py` 中的 `train_for_steps`：

1. 创建空的 `losses` 列表；
2. 循环执行指定次数的 `train_one_step`；
3. 将每次返回的 loss 按顺序加入列表；
4. 返回完整的 loss 历史；
5. `steps=0` 时返回空列表。

先使用固定 seed、`hidden_dim=8`、SGD 学习率 `0.1` 和 XOR 数据训练 `100` 步。预期 loss 会从约 `0.796` 降到约 `0.412`，预测类别变为 `[0, 1, 1, 0]`。具体数值以当前环境实际运行结果为准。

验证命令：

```powershell
& 'D:\anaconda\envs\rl\python.exe' -m pytest lesson_19/test_train_loop.py -q -p no:cacheprovider
```

骨架预期先失败；实现完成后应为 `4 passed`。

## Day 6：独立评估 loss 和 accuracy

只修改 `evaluation.py` 中的 `evaluate_classifier`：

1. 调用 `model.eval()` 切换到评估模式；
2. 在 `with torch.no_grad():` 中完成前向计算；
3. 使用 `CrossEntropyLoss` 计算评估 loss；
4. 使用 `logits.argmax(dim=1)` 得到预测类别；
5. 计算预测正确的比例作为 accuracy；
6. 返回 `{"loss": loss_float, "accuracy": accuracy_float}`；
7. 不调用 `backward()`、`zero_grad()` 或 `optimizer.step()`。

验证命令：

```powershell
& 'D:\anaconda\envs\rl\python.exe' -m pytest lesson_19/test_evaluation.py -q -p no:cacheprovider
```

骨架预期先失败；实现完成后应为 `4 passed`。

## Day 7：保存和加载 MLP checkpoint

这一部分迁移第 11 课已经学过的 `state_dict`，不引入新的保存框架。

只修改 `checkpoint.py`：

1. `save_mlp_model` 使用 `torch.save(model.state_dict(), path)`；
2. `load_mlp_model` 按传入维度新建 `MLPClassifier`；
3. 使用 `torch.load(path, weights_only=True)` 读取 state dict；
4. 调用 `model.load_state_dict(...)` 恢复参数；
5. 将加载后的模型切换为评估模式并返回；
6. 不保存整个 Python 模型对象。

验证命令：

```powershell
& 'D:\anaconda\envs\rl\python.exe' -m pytest lesson_19/test_checkpoint.py -q -p no:cacheprovider
```

骨架预期先失败；实现完成后应为 `3 passed`。
