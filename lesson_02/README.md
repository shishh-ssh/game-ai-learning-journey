# 第 2 课：GridWorld 与 BFS

## 学习目标

本课在第 1 课坐标与状态转换的基础上，学习集合、相邻状态、队列、访问记录和广度优先搜索，并在带障碍物的网格中寻找最短路径。

本课只使用 Python 标准库，不引入类、NumPy、Gym、A* 或强化学习框架。

## 网格约定

- `(0, 0)` 位于左上角；
- `x` 向右增加，`y` 向下增加；
- `blocked` 是障碍物坐标的集合；
- 本课假定 `blocked` 中保存的是合法地图坐标，不额外校验障碍物本身；
- 相邻格子的固定检查顺序为：上、下、左、右。

固定顺序很重要：存在多条等长最短路径时，它让测试结果保持可复现。

## 相邻状态接口

```python
def get_neighbors(
    position: tuple[int, int],
    width: int,
    height: int,
    blocked: set[tuple[int, int]],
) -> list[tuple[int, int]]:
```

规则：

1. `width <= 0` 或 `height <= 0`，抛出 `ValueError`；
2. `position` 越界或位于障碍物中，抛出 `ValueError`；
3. 返回所有未越界且不在 `blocked` 中的相邻格子；
4. 返回顺序固定为上、下、左、右；
5. 没有可行邻居时返回空列表。

## 最短路径接口

```python
def find_shortest_path(
    start: tuple[int, int],
    goal: tuple[int, int],
    width: int,
    height: int,
    blocked: set[tuple[int, int]],
) -> list[tuple[int, int]]:
```

规则：

1. 地图尺寸非法，抛出 `ValueError`；
2. 起点或终点越界、位于障碍物中，抛出 `ValueError`；
3. 返回的路径同时包含起点和终点；
4. `start == goal` 时返回 `[start]`；
5. 存在路径时返回按固定邻居顺序找到的最短路径；
6. 不存在路径时返回空列表 `[]`。

## 学习顺序

1. 集合与障碍物成员判断；
2. 计算候选相邻坐标；
3. 过滤越界位置和障碍物；
4. 理解 FIFO 队列与 BFS；
5. 使用 `visited` 防止重复搜索；
6. 使用 `parents` 重建最短路径；
7. 完成口试和现场需求修改。

不要提前实现尚未讲解的步骤。每一步通过测试和代码审查后再继续。

