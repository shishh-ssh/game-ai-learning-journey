# 第 14 课：完整可复现训练实验

本课整合固定种子、多轮训练、独立验证、模型参数保存和 JSON 实验记录，形成第一个完整实验模块。

`run_training_experiment` 接收训练/验证数据、超参数和两个输出路径，返回与 JSON 文件相同的实验记录。

验证命令：

```powershell
D:\anaconda\envs\rl\python.exe -m pytest lesson_14 -q -p no:cacheprovider
```
