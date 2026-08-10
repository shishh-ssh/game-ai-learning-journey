# 第 3 课：A* 搜索

## 本课要解决的问题

BFS 不知道终点位于哪个方向。它会按照离起点的步数一层一层向四周扩展，即使终点明显位于右侧，也会同时检查大量向左、向上和向下的位置。

A* 的目标是在仍然寻找最短路径的同时，利用“终点大概在哪个方向”的信息减少无关搜索。

本课不会直接背诵公式或照抄完整代码。先理解 BFS 的不足，再逐步建立 A*。

## 分级学习顺序

### 第 1 级：观察 BFS 的搜索范围

在一张无障碍小地图上，比较终点方向与 BFS 的逐层扩展。此时不引入 `g(n)`、`h(n)`、`f(n)` 或优先队列。

### 第 2 级：估计至少还要走几步

只允许上、下、左、右移动时，先实际数出横向和纵向各需要几步，再给出“曼哈顿距离”这个名称。明确它只依据移动规则和两个坐标，不预测具体路线。

### 第 3 级：理解启发式 `h(n)`

本课选择曼哈顿距离作为 `h(n)`。依次观察：无障碍时估计可能正好等于真实距离；有障碍时估计可能偏小；终点不可达时仍然能算出估计值。完成这些观察后，才概括 `h(n)` 是“预计剩余代价”。

### 第 4 级：理解实际代价 `g(n)`

沿一条已经走过的路径逐步累计代价。`g(n)` 来自当前已知路径，而不是只根据当前位置坐标推算。完整 A* 中若发现更短路线，还要更新该位置的 `g(n)`。

### 第 5 级：理解为什么使用 `g(n) + h(n)`

比较三种思路：BFS 只按已经走过的层数扩展；只看 `h(n)` 容易被障碍误导；A* 同时考虑已经付出的代价和终点方向。教师先演示完整评分表，再由学习者补充一行。

### 第 6 级：单独学习优先队列

脱离 A*，先用小型练习掌握 `heapq.heappush`、`heapq.heappop` 和“最小优先级先取出”。确认理解后才把它接回搜索算法。

当前优先队列练习接口：

```python
def pop_positions_by_priority(
    entries: list[tuple[int, str]],
) -> list[str]:
```

`entries` 中每个元素都是 `(priority, position_name)`。当前练习保证优先级互不相同，因此暂时不处理相同优先级的规则。

实现规则：

1. 新建空列表作为最小堆，不修改传入的 `entries`；
2. 使用 `for` 和 `heapq.heappush` 把所有条目加入最小堆；
3. 使用 `while` 和 `heapq.heappop` 逐个取出条目；
4. 只把位置名称加入结果列表；
5. 返回按照优先级从小到大排列的位置名称；
6. 输入为空时返回空列表；
7. 不允许使用 `sorted()` 或 `list.sort()` 代替最小堆。

当前允许修改的唯一区域是 `priority_queue_review.py` 中 `pop_positions_by_priority` 的函数体。必须保持导入语句、函数签名、文档字符串和测试文件不变。

验证命令：

```powershell
python -m pytest lesson_03/test_priority_queue_review.py -q -p no:cacheprovider
```

实现正确时预期为 `5 passed`。

### 第 7 级：从 BFS 逐步改成 A*

继续复用邻居生成、边界、障碍物、`parents` 和路径回溯。分步骤加入 `g_score`、候选代价、较优路径更新、优先级和最小堆，不一次加入所有结构。

当前先实现其中一个独立步骤：

```python
def update_neighbor_if_better(
    current: tuple[int, int],
    neighbor: tuple[int, int],
    goal: tuple[int, int],
    g_score: dict[tuple[int, int], int],
    parents: dict[tuple[int, int], tuple[int, int] | None],
) -> tuple[int, tuple[int, int]] | None:
```

规则：

1. 本练习每次移动代价固定为 `1`；
2. 使用 `g_score[current] + 1` 得到到达 `neighbor` 的新候选代价；
3. `neighbor` 从未记录，或者新候选代价更低时，同时更新 `g_score[neighbor]` 与 `parents[neighbor]`；
4. 更新后使用曼哈顿距离计算 `f`，返回 `(f, neighbor)`；
5. 已知路线代价小于或等于新候选代价时，不修改两个字典并返回 `None`；
6. 当前假定 `current` 一定存在于 `g_score`，暂时不增加输入异常处理；
7. 不修改 `manhattan_distance` 和测试文件。

当前允许修改的唯一区域是 `astar.py` 中 `update_neighbor_if_better` 的函数体。

验证命令：

```powershell
python -m pytest lesson_03/test_astar.py -q -p no:cacheprovider
```

实现正确时预期为 `10 passed`。

候选代价更新通过后，完整搜索接口为：

```python
def find_shortest_path_astar(
    start: tuple[int, int],
    goal: tuple[int, int],
    width: int,
    height: int,
    blocked: set[tuple[int, int]],
) -> list[tuple[int, int]]:
```

行为契约与第 2 课 BFS 保持一致：

1. 地图尺寸非法，抛出 `ValueError`；
2. 起点或终点越界、位于障碍物中，抛出 `ValueError`；
3. `start == goal` 时返回 `[start]`；
4. 存在路径时返回同时包含起点和终点的最短路径；
5. 存在多条等长路径时不要求与 BFS 返回完全相同的坐标序列；
6. 不可达时返回 `[]`；
7. 必须复用 `get_neighbors`、`manhattan_distance` 和 `update_neighbor_if_better`。

完整函数分四段实现：输入校验与初始化；最小堆取出与目标判断；相邻位置更新；父位置回溯。每段单独检查，不一次完成。

最终验证命令：

```powershell
python -m pytest lesson_03/test_astar.py -q -p no:cacheprovider
```

完整实现正确时预期为 `21 passed`。

### 第 8 级：比较与验收

在同一地图比较 BFS 与 A*：是否得到同样长度的最短路径、分别处理多少位置、A* 为什么通常处理更少。最后只进行一次综合口试和一次独立实操。

### 拓展内容

可采纳性、一致性、对角移动、加权地形和节点重新打开均放在完整 A* 通过之后，不参与当前验收。

## 完成状态

本课已经完成曼哈顿距离、最小优先队列、候选代价更新、完整 A*、正式口试和独立路径回溯实操。A* 与 BFS 在现有无权网格测试中返回相同长度的最短路径。

结课验证：

```powershell
python -m pytest lesson_03 -q -p no:cacheprovider
```

预期结果为 `28 passed`。

## 参考资料

- [Red Blob Games: Introduction to the A* Algorithm](https://www.redblobgames.com/pathfinding/a-star/introduction.html)：当前主要资料，按 BFS、移动代价、启发式、A* 的顺序建立可视化直觉。
- [Berkeley CS188: Informed Search](https://inst.eecs.berkeley.edu/~cs188/textbook/search/informed.html)：完成直觉阶段后，用于理解启发函数、贪心搜索、A* 和最优性。
- [Python 官方文档：heapq](https://docs.python.org/3/library/heapq.html)：只在第 6 级优先队列阶段使用。
- [Red Blob Games: Implementation of A*](https://www.redblobgames.com/pathfinding/a-star/implementation.html)：只在开始完整实现时使用。
