# 第 13 课：实验记录

本课使用 JSON 保存和读取实验配置与结果，使训练结果可以追踪和复核。

实验记录包含 `seed`、`learning_rate`、`epochs`、`final_loss`、`weight` 和 `bias`。只修改 `experiment_record.py` 中两个函数的函数体。

验证命令：

```powershell
D:\anaconda\envs\rl\python.exe -m pytest lesson_13 -q -p no:cacheprovider
```
