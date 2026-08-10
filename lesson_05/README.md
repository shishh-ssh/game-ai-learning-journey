# 第 5 课：NumPy 数组基础

本课开始阶段 2 的数值计算基础。第一小节只学习一维数组、`shape`、`dtype` 和向量化缩放，暂不引入广播、矩阵乘法、PyTorch 或训练循环。

## 为什么不用普通列表直接训练

Python 列表乘以整数会重复序列，而 NumPy 数组乘以标量会逐元素计算：

```python
import numpy as np

python_values = [1, 2, 3]
array_values = np.array([1, 2, 3])

print(python_values * 2)  # [1, 2, 3, 1, 2, 3]
print(array_values * 2)   # [2 4 6]
```

NumPy 数组的几个最小概念：

- `shape` 描述每个轴的长度，三个元素的一维数组形状为 `(3,)`；
- `dtype` 描述统一的元素类型，例如 `float64`；
- 向量化表示直接对整个数组运算，不在 Python 中手写逐元素循环。

## 当前任务：缩放奖励

修改文件：`lesson_05/array_basics.py`

完整接口：

```python
def scale_rewards(
    rewards: list[float],
    factor: float,
) -> np.ndarray:
```

参数与返回值：

- `rewards` 是一局游戏中依次记录的奖励，可以包含整数或小数；
- `factor` 是应用到每个奖励上的缩放因子；
- 返回一个独立的一维 NumPy 数组。

全部规则：

1. 将 `rewards` 转换为元素类型为 `np.float64` 的 NumPy 数组；
2. 使用数组与标量的逐元素乘法完成缩放；
3. 返回数组的形状必须为 `(len(rewards),)`；
4. 空列表应返回形状为 `(0,)`、元素类型为 `np.float64` 的空数组；
5. 不修改输入列表；
6. 不使用显式 `for`、`while` 或列表推导式，让 NumPy 完成向量化运算。

只允许修改 `scale_rewards` 的函数体。保持导入、函数签名和测试文件不变。

定向验证：

```powershell
python -m pytest lesson_05/test_array_basics.py -q -p no:cacheprovider
```

正确实现后预期为 `5 passed`，全仓预期为 `128 passed`。

## 二维数组与整行选择

二维数组的 `shape` 是二元组。对于形状为 `(2, 3)` 的数组：

- `shape[0] == 2`，表示有 2 行；
- `shape[1] == 3`，表示每行有 3 列；
- `array[1]` 选出第 2 整行，结果是一维数组，形状为 `(3,)`。

当前只练习按状态索引选出一整行动作分数。题目保证 `score_table` 是非空、每行长度相同的二维列表，并且 `state_index` 是有效的非负行索引；本小节不增加输入校验。

修改文件：`lesson_05/array_basics.py`

完整接口：

```python
def select_state_scores(
    score_table: list[list[float]],
    state_index: int,
) -> np.ndarray:
```

全部规则：

1. 先将完整的 `score_table` 转换为元素类型为 `np.float64` 的二维 NumPy 数组；
2. 再从这个二维 NumPy 数组中使用 `state_index` 选择对应的一整行；
3. 返回值必须是一维 `np.ndarray`，形状为 `(列数,)`；
4. 不修改输入的嵌套列表；
5. 不使用显式循环、列表推导式、切片或矩阵乘法。

只允许修改 `select_state_scores` 的函数体。保持 `scale_rewards`、导入、函数签名和测试不变。

实现顺序也是验收内容：不能先从 Python 列表 `score_table` 中选行再转换，必须先建立完整二维数组，再对数组使用整行索引。

定向验证：

```powershell
python -m pytest lesson_05/test_state_scores.py -q -p no:cacheprovider
```

正确实现后预期为 `4 passed`；第 5 课应为 `9 passed`，全仓应为 `132 passed`。

## 按轴计算平均值

对于“行表示状态、列表示动作”的二维分数表：

- `array.mean(axis=0)` 消掉行维度，返回每个动作跨所有状态的平均分，结果长度等于列数；
- `array.mean(axis=1)` 消掉列维度，返回每个状态内部所有动作的平均分，结果长度等于行数。

本小节使用一个函数同时计算两个方向，避免把轴概念拆成重复的小练习。题目保证输入是非空、每行长度相同的二维列表。

修改文件：`lesson_05/array_basics.py`

完整接口：

```python
def compute_score_means(
    score_table: list[list[float]],
) -> tuple[np.ndarray, np.ndarray]:
```

返回顺序必须为：

```text
(每个动作的平均分, 每个状态的平均分)
```

全部规则：

1. 将完整 `score_table` 转换为 `np.float64` 二维数组；
2. 使用 `axis=0` 计算每个动作的平均分；
3. 使用 `axis=1` 计算每个状态的平均分；
4. 按规定顺序返回两个一维数组；
5. 不修改输入，不使用显式循环或列表推导式。

只允许修改 `compute_score_means` 的函数体。保持已有函数、导入、签名和测试不变。

定向验证：

```powershell
python -m pytest lesson_05/test_score_means.py -q -p no:cacheprovider
```

正确实现后预期为 `4 passed`；第 5 课应为 `13 passed`，全仓应为 `136 passed`。

## 每个状态的最佳动作

`argmax` 返回最大值所在的索引，`max` 返回最大值本身。对于行表示状态、列表示动作的二维表，两者都使用 `axis=1` 在每个状态内部比较动作：

```python
score_table.argmax(axis=1)  # 每个状态的最佳动作索引
score_table.max(axis=1)     # 每个状态的最高动作分数
```

当前任务同时返回这两个结果，强化“动作索引”和“动作分数”的区别。题目保证输入是非空、每行长度相同的二维列表；本小节不处理并列最大值策略。

修改文件：`lesson_05/array_basics.py`

完整接口：

```python
def select_best_actions(
    score_table: list[list[float]],
) -> tuple[np.ndarray, np.ndarray]:
```

返回顺序必须为：

```text
(每个状态的最佳动作索引, 每个状态的最高动作分数)
```

全部规则：

1. 将完整 `score_table` 转换为 `np.float64` 二维数组；
2. 使用 `argmax(axis=1)` 计算每个状态的最佳动作索引；
3. 使用 `max(axis=1)` 计算每个状态的最高动作分数；
4. 两个结果都应是一维数组，长度等于状态数；
5. 不修改输入，不使用显式循环或列表推导式。

只允许修改 `select_best_actions` 的函数体。保持已有函数、导入、签名和测试不变。

定向验证：

```powershell
python -m pytest lesson_05/test_best_actions.py -q -p no:cacheprovider
```

正确实现后预期为 `4 passed`；第 5 课应为 `17 passed`，全仓应为 `140 passed`。

## 基础广播：每个动作的统一修正值

数组加法是逐元素运算。NumPy 从最右侧比较形状；对应维度相等或其中一个维度为 `1` 时可以广播：

```text
分数表       (状态数, 动作数)
动作修正值            (动作数,)
```

长度为“动作数”的一维数组会与二维表的列对齐，并逻辑上应用到每个状态。这里不是矩阵乘法，也不需要手动复制修正值。

题目保证 `score_table` 是非空等长二维列表，`action_bonus` 长度等于动作列数。本小节不增加形状校验。

修改文件：`lesson_05/array_basics.py`

完整接口：

```python
def apply_action_bonus(
    score_table: list[list[float]],
    action_bonus: list[float],
) -> np.ndarray:
```

全部规则：

1. 将完整 `score_table` 和 `action_bonus` 分别转换为 `np.float64` 数组；
2. 直接将二维分数数组与一维修正数组相加，使用 NumPy 广播；
3. 返回形状与原分数表相同的独立二维数组；
4. 不修改输入；
5. 不使用循环、列表推导式、`tile`、`repeat`、手动复制或矩阵乘法。

只允许修改 `apply_action_bonus` 的函数体。保持已有函数、导入、签名和测试不变。

定向验证：

```powershell
python -m pytest lesson_05/test_broadcasting.py -q -p no:cacheprovider
```

正确实现后预期为 `4 passed`；第 5 课应为 `21 passed`，全仓应为 `144 passed`。

## 综合任务：调整分数并生成策略

动作修正向量形状为 `(动作数,)`，会按列应用。状态修正向量原始形状为 `(状态数,)`，需要先改成 `(状态数, 1)` 列向量，才能按行应用到所有动作：

```text
score_table          (状态数, 动作数)
action_bonus                  (动作数,)
state_bonus_column   (状态数,      1)
```

当前任务同时应用两个方向的广播，再使用 `argmax(axis=1)` 为每个状态选择最佳动作。

题目保证三个输入非空，分数表每行等长，动作修正长度等于列数，状态修正长度等于行数；本任务不增加输入校验。

修改文件：`lesson_05/array_basics.py`

完整接口：

```python
def build_adjusted_policy(
    score_table: list[list[float]],
    action_bonus: list[float],
    state_bonus: list[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
```

返回顺序必须为：

```text
(应用全部修正后的二维分数表, 每个状态的最佳动作索引, 每个状态的最高分)
```

全部规则：

1. 将三个完整输入分别转换为 `np.float64` 数组；
2. 将状态修正数组改为 `(状态数, 1)`，可以使用 `reshape(len(state_bonus), 1)`；
3. 通过广播同时应用动作修正与状态修正；
4. 在调整后的分数表上使用 `argmax(axis=1)`，得到每个状态的最佳动作索引；
5. 在同一个调整后分数表上使用 `max(axis=1)`，得到每个状态的最高分；
6. 按规定顺序返回二维分数表、一维动作索引数组和一维最高分数组；
7. `best_actions.shape == (状态数,)`，`best_scores.shape == (状态数,)`，且 `best_scores.dtype == np.float64`；
8. 不修改输入，不使用循环、列表推导式、`tile` 或 `repeat`。

只允许修改 `build_adjusted_policy` 的函数体。保持已有函数、导入、签名和测试不变。

定向验证：

```powershell
python -m pytest lesson_05/test_adjusted_policy.py -q -p no:cacheprovider
```

正确实现后预期为 `4 passed`；第 5 课应为 `25 passed`，全仓应为 `148 passed`。

## 结课验收

- 分项测试：`25/25` 通过；全仓回归：`148/148` 通过；
- 正式口试：通过；能够解释数组类型、形状、轴、广播和索引与数值的区别；
- 独立现场修改：通过；`build_adjusted_policy` 现在同时返回调整后的分数表、最佳动作索引和每个状态的最高分；
- 当前状态：本课已掌握，下一课尚未开始。
