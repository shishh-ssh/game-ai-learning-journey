# 第 11 课：保存与加载模型参数

本课使用 `state_dict()`、`torch.save()`、`torch.load()` 和 `load_state_dict()` 保存并恢复线性层参数，再使用相同输入验证加载前后的预测行为。加载时必须提供与参数匹配的输入和输出特征数量。

只修改 `model_persistence.py` 中两个函数的函数体。

验证命令：

```powershell
D:\anaconda\envs\rl\python.exe -m pytest lesson_11 -q -p no:cacheprovider
```
