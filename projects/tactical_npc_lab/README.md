# AI Native NPC Agent Lab

一个面向 AI Native 游戏场景的持久化 NPC Agent 与评测平台。项目围绕角色扮演、任务规划、工具调用、环境交互、长期记忆、自主决策和多智能体展开，同时保留可控执行层与 Unity 场景。

## 最终场景

NPC 能够在跨会话剧情中保持 Persona，记住玩家行为、角色关系、任务状态和关键事件；根据当前世界状态进行 Planning，通过 Function Calling 调用移动、对话、调查、交付和战斗等工具；在执行失败后更新计划或触发 Reflection。系统使用固定任务集评测任务完成率、角色一致性、叙事合理性、工具调用效果、状态一致性、安全性和延迟。

## 系统边界

```text
玩家/剧情输入 -> Persona + State Tracking
                       |
短期上下文 + 长期记忆检索 -> Planning / ReAct
                                  |
                    结构化 Tool Call + 安全校验
                                  |
                     游戏世界 / Unity 环境
                                  |
事件日志 -> Memory 写入 -> Reflection -> Agent Eval
```

核心模块按真实功能逐步创建：

- `world`：角色、关系、剧情、任务和确定性游戏规则；
- `task_agent`：ReAct、Planning、Tool Use、Reflection 和恢复；
- `memory`：短期上下文、长期记忆、向量检索和写入策略；
- `state_tracking`：任务、剧情、关系和环境状态的一致性维护；
- `evaluation`：固定任务集、自动评测、失败分类和人工复核；
- `alignment`：小模型 SFT/DPO/GRPO 或 PPO 复现实验；
- `recording`：事件日志、轨迹与回放；
- `adapters`：Unity、模型服务和外部工具适配。

## 第一条垂直链路

首个可运行场景包含：

- 一名具有固定 Persona 的 NPC；
- 三段连续任务和跨任务状态；
- 玩家、NPC、任务物品和两个地点；
- `move`、`talk`、`inspect`、`give_item` 等结构化工具；
- 短期记忆、长期事件记忆和关系状态；
- 成功、工具调用失败、角色偏离和状态矛盾等评测标签。

第一版先使用 Python 世界模拟器和确定性 mock planner 贯通 State Tracking、Tool Use、Memory 与 Eval，再接入小型 LLM。Unity 同期建立最小交互场景，但主要作为环境适配与演示层，不取代 Agent 核心研发。

## 学习者与 AI 的分工

学习者负责理解和实现：Transformer 核心、结构化工具调用、ReAct/Planning、记忆检索与写入、Agent Eval、SFT/DPO 基本训练流程，以及 PPO/GRPO 的关键目标函数。BFS/A*、Q-learning 和 DQN 作为已有算法基础与低层技能保留，不再占据项目主线。

AI 可以负责：配置样板、批量测试、图表、结果文件、报告排版和重复性适配代码。学习者必须审核测试含义、统计口径和实验结论。

## 当前里程碑

`P1-M1`：建立确定性的游戏世界、任务状态、Persona、结构化工具接口和固定评测样例。先证明同一条任务轨迹能够被执行、记录、重放和判定，再接入 LLM。
