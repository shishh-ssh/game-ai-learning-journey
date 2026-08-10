# 第 12 课：可复现训练

本课使用 `torch.manual_seed(seed)` 控制 PyTorch 模型初始化，并保持数据、模型结构、学习率、epoch 和操作顺序一致，验证相同配置可以复现相同训练结果。

只修改 `reproducible_training.py` 中 `train_reproducible` 的函数体。

验证命令：

```powershell
D:\anaconda\envs\rl\python.exe -m pytest lesson_12 -q -p no:cacheprovider
```
