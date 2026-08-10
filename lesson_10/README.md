# 第 10 课：多轮线性回归训练

本课将第 9 课的单轮训练扩展为多轮训练，并保存每轮损失，观察模型是否逐步接近数据关系。

当前只修改 `multi_epoch_training.py` 中 `train_linear_many` 的函数体。

要求：创建 `Linear(1, 1)`，将参数初始化为 `0.0`，把训练和验证数据转换为 `float32` 的 `(-1, 1)` 张量，严格执行 `zero_grad -> forward -> loss -> backward -> 保存 loss -> step`。训练结束后使用 `eval()` 与 `no_grad()` 计算验证损失，最后返回训练损失历史、验证损失、最终权重和最终偏置。

验证命令：

```powershell
D:\anaconda\envs\rl\python.exe -m pytest lesson_10 -q -p no:cacheprovider
```
