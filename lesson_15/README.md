# 第 15 课：Torch Q 表单步更新

本课使用 PyTorch 二维张量保存 Q 表，并根据一次 `(state, action, reward, next_state, done)` 交互更新当前状态—动作对应的 Q 值。

只修改 `q_table.py` 中 `update_q_value` 的函数体。函数原地更新 Q 表，并返回更新后的 Python 浮点数。

验证命令：

```powershell
D:\anaconda\envs\rl\python.exe -m pytest lesson_15 -q -p no:cacheprovider
```
