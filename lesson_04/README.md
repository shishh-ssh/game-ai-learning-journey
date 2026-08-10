# 第 4 课：NPC 有限状态机

本课学习 FSM（Finite State Machine，有限状态机），使用明确状态和转换规则管理 NPC 的持续行为模式。

## 为什么需要 FSM

A* 负责回答“怎样到达目标”，但不负责决定 NPC 当前应该巡逻、追击、攻击还是逃跑。第 0 课的条件函数也没有保存上一帧状态。FSM 将“当前模式”作为输入，使相同观测在不同状态下可以产生不同转换。

## 核心关系

```text
下一状态 = 转换规则（当前状态，当前观测）
```

本课使用五个状态：

- `patrol`：巡逻；
- `chase`：追击；
- `attack`：攻击；
- `flee`：逃跑；
- `dead`：死亡。

## 转换规则

规则按下列优先级判断：

1. `dead` 是终止状态，普通观测不能使其离开；
2. 任意存活状态在 `health == 0` 时进入 `dead`；
3. `patrol`、`chase`、`attack` 在 `health < 30` 时进入 `flee`；
4. `flee` 在 `health < 50` 时保持，在 `health >= 50` 时回到 `patrol`；
5. 其余情况使用当前状态专属规则。

| 当前状态 | 条件 | 下一状态 |
| --- | --- | --- |
| `patrol` | 看见敌人 | `chase` |
| `patrol` | 看不见敌人 | `patrol` |
| `chase` | 看不见敌人 | `patrol` |
| `chase` | 敌人可见且距离 `<= 2` | `attack` |
| `chase` | 敌人可见且距离 `> 2` | `chase` |
| `attack` | 看不见敌人 | `patrol` |
| `attack` | 敌人可见且距离 `> 2` | `chase` |
| `attack` | 敌人可见且距离 `<= 2` | `attack` |

## 学习顺序

1. 理解状态、事件、转换和跨帧状态保存；
2. 实现死亡终止状态；
3. 实现低生命值打断与逃跑恢复；
4. 实现三个普通状态的专属转换；
5. 运行完整测试并进行口试和现场修改。

单帧转换曾按规则优先级分步实现，第一步只处理死亡规则并保留占位异常。

验证死亡规则：

```powershell
python -m pytest lesson_04/test_npc_fsm.py -k dead -q -p no:cacheprovider
```

正确时预期为 `7 passed, 17 deselected`。

单帧转换验收：

```powershell
python -m pytest lesson_04 -q -p no:cacheprovider
```

单帧转换完成后的结果为 `24 passed`。

## 参考资料

- [Game Programming Patterns: State](https://gameprogrammingpatterns.com/state.html)：通过游戏角色行为说明状态模式、状态转换和状态对象。
- [Wikipedia: Finite-state machine](https://en.wikipedia.org/wiki/Finite-state_machine)：用于查阅 FSM 的标准术语与基本定义。

状态对象、分层状态机和行为树属于后续拓展，本课暂不引入。

## 跨帧状态序列

单帧状态转换完成后，本课继续验证调用者如何保存每次 `transition_state` 的返回值，使下一帧从新状态继续。

修改文件：`lesson_04/state_sequence.py`

完整接口：

```python
def run_state_sequence(
    initial_state: str,
    observations: list[tuple[int, bool, float]],
) -> list[str]:
```

参数与返回值：

- `initial_state` 是处理第一帧之前的状态；
- `observations` 中每个元组依次表示一帧的 `health`、`enemy_visible`、`distance`；
- 返回列表首先包含 `initial_state`，随后依次包含每帧处理后的新状态；
- 因此返回列表长度必须等于 `len(observations) + 1`。

实现要求：

1. 建立独立的当前状态变量和状态历史列表；
2. 按顺序处理每个观测元组；
3. 每帧只调用一次 `transition_state`；
4. 将返回值保存为下一帧使用的当前状态；
5. 将每帧的新状态追加到历史中，最后返回完整历史。

实现时保持函数签名和导入不变，由 `run_state_sequence` 的函数体负责跨帧状态保存。

定向验证：

```powershell
python -m pytest lesson_04/test_state_sequence.py -q -p no:cacheprovider
```

该部分完成后的结果为 `4 passed`。随后运行第 4 课回归测试：

```powershell
python -m pytest lesson_04 -q -p no:cacheprovider
```

加入跨帧测试后的结果为 `28 passed`。

## 结课现场修改：巡逻侦测范围

正式口试通过后，完成了一个不改变函数接口的需求：巡逻状态的侦测范围为 `6` 个距离单位。

修改范围限定为 `lesson_04/npc_fsm.py` 中 `current_state == "patrol"` 的状态专属分支，其他状态规则、输入校验、函数签名和测试文件保持不变。

新规则：

- 当前状态为 `patrol`、敌人可见且 `distance <= 6` 时，下一状态为 `chase`；
- 当前状态为 `patrol`、敌人不可见时，下一状态为 `patrol`；
- 当前状态为 `patrol`、敌人可见但 `distance > 6` 时，下一状态也为 `patrol`。

定向验证：

```powershell
python -m pytest lesson_04/test_live_change.py -q -p no:cacheprovider
```

现场修改结果为 `3 passed`。第 4 课最终为 `31 passed`，全仓为 `123 passed`。口头验收能够解释 `6.0` 和 `6.1` 的边界差异、侦测距离与攻击距离的职责，以及完整回归测试对旧规则的保护作用。

## 结课结论

第 4 课的单帧转换、跨帧状态保存、正式口试和独立现场修改均已通过。用户能够实现并解释状态优先级、终止状态、滞回、一帧一次转换和调用者保存状态的机制。
