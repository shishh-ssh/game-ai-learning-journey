# 第 16 课：Torch epsilon-greedy 动作选择

本课根据 Torch Q 表选择动作：以 `epsilon` 概率随机探索，否则选择当前状态下 Q 值最大的动作。随机操作通过独立的 `torch.Generator` 实现可复现。

只修改 `torch_epsilon_policy.py` 中 `select_action` 的函数体。

验证命令：

```powershell
D:\anaconda\envs\rl\python.exe -m pytest lesson_16 -q -p no:cacheprovider
```
