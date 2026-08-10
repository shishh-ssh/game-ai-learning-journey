# 第 18 课：完整 Q-learning 训练闭环

本课把已经完成的三个模块连接起来：

- 第 15 课：`update_q_value` 单步 Q-learning 更新；
- 第 16 课：`select_action` epsilon-greedy 动作选择；
- 第 17 课：Gymnasium 环境与 episode 交互。

## 学习目标

本课不重新推导已经学过的单步更新，而是建立正确的数据流：

```text
state
  ↓
select_action(q_table, state, epsilon, generator)
  ↓
environment.step(action)
  ↓
next_state, reward, terminated, truncated
  ↓
update_q_value(state, action, reward, next_state, ...)
  ↓
state = next_state
```

## 课程顺序

1. 实现单个训练 episode；
2. 验证旧状态没有被 `next_state` 提前覆盖；
3. 区分 episode 停止条件与 TD target 终止条件；
4. 扩展为多个训练 episode；
5. 记录每局累计回报和步数；
6. 使用 `epsilon=0` 独立评估训练后的策略；
7. 与 Random 和 AlwaysRight 基线比较；
8. 最后再加入测试和学习曲线。

## 已完成初版：单个训练 episode

`q_learning.py` 中的 `train_one_episode` 已有初版实现。当前仍需通过定向测试和代码讲解确认旧状态、结束条件及累计回报处理正确。

函数应原地更新传入的 Q 表，并返回：

```text
(episode_return, steps, terminated, truncated)
```

## 已完成初版：多个训练 episode

`train_q_learning` 已能返回：

```text
(q_table, return_history, steps_history)
```

当前版本使用固定 epsilon，并有意独立重写单局训练循环，用于验证能否把已经理解的数据流迁移到多局训练。验收通过后，再从项目维护角度比较“保留独立实现”和“调用 `train_one_episode`”的取舍。epsilon 衰减将在训练闭环验证完成后单独加入。

## 已完成初版：独立评估

实现 `evaluate_q_table`：

- 使用训练完成的 Q 表；
- 每一步选择当前状态下 Q 值最大的动作；
- 不进行随机探索；
- 不调用 `update_q_value`；
- 不修改传入的 Q 表；
- 返回成功率、平均回报和平均步数。

## 当前任务：训练闭环算法验收

1. 审查独立编写的多局循环，确认它与单局训练的数据流一致；
2. 审查 greedy 评估，确认评估期间没有探索和 Q 值更新；
3. 由教师提供定向测试，验证单局训练只更新实际访问的状态—动作格子；
4. 由教师提供定向测试，验证 truncation 会停止 episode，但不会被当成 TD target 的真正终止；
5. 由教师提供定向测试，验证多局历史、评估只读和成功/截断统计；
6. 学习者当前负责理解测试目标与结果，不要求独立编写 pytest；
7. 通过算法验收后，再比较 Random、AlwaysRight 与 Q-learning。

当前第 18 课已掌握：定向测试 `7/7`、全仓回归 `208/208` 通过，覆盖终止更新、truncation bootstrap、多局历史、按 episode 衰减 epsilon、只读 greedy 评估、多训练种子实验以及均值/样本标准差。学习者不要求独立编写 pytest，也不手工重复实现实验表排版、曲线绘制和结果文件生成；这些产物后续由 AI 自动生成，学习者负责审核实验设置、统计口径和结论。

## 本课重点错误

- 在 Q 更新前用 `next_state` 覆盖旧 `state`；
- 调用 `select_action` 时遗漏 `epsilon` 或 `generator`；
- 更新 `Q(next_state, action)` 而不是 `Q(state, action)`；
- 把最后一步即时奖励当成整个 episode 回报；
- 把 `terminated` 和 `truncated` 无条件合并后传给 TD target；
- 训练评估时仍然保留随机探索。
