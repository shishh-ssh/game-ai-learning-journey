"""第 3 课：使用 A* 在网格中搜索路径。"""

import heapq

from lesson_02.grid_search import get_neighbors


def manhattan_distance(
    position: tuple[int, int],
    goal: tuple[int, int],
) -> int:
    """返回两个二维位置之间的曼哈顿距离。"""
    x, y = position
    goal_x, goal_y = goal
    distance = abs(x - goal_x) + abs(goal_y - y)
    return distance


def update_neighbor_if_better(
    current: tuple[int, int],
    neighbor: tuple[int, int],
    goal: tuple[int, int],
    g_score: dict[tuple[int, int], int],
    parents: dict[tuple[int, int], tuple[int, int] | None],
) -> tuple[int, tuple[int, int]] | None:
    """发现更低代价时更新记录，并返回新的优先队列条目。"""
    tentative_g = g_score[current] + 1
    if neighbor not in g_score or tentative_g < g_score[neighbor]:
        g_score[neighbor] = tentative_g
        parents[neighbor] = current
        h = manhattan_distance(neighbor, goal)
        f = tentative_g + h
        return (f, neighbor)
    return None


def reconstruct_path(
    goal: tuple[int, int],
    parents: dict[tuple[int, int], tuple[int, int] | None],
) -> list[tuple[int, int]]:
    """从终点沿父节点关系回溯，并返回从起点到终点的路径。"""
    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = parents[current]
    path.reverse()
    return path


def find_shortest_path_astar(
    start: tuple[int, int],
    goal: tuple[int, int],
    width: int,
    height: int,
    blocked: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    """使用 A* 返回包含起点和终点的最短路径。"""
    if width <= 0 or height <= 0:
        raise ValueError("网格宽度和高度必须为正整数。")
    x, y = start
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError("起点必须在网格范围内。")
    g_x, g_y = goal
    if not (0 <= g_x < width and 0 <= g_y < height):
        raise ValueError("终点必须在网格范围内。")
    if start in blocked:
        raise ValueError("起点不能被阻挡。")
    if goal in blocked:
        raise ValueError("终点不能被阻挡。")
    if start == goal:
        return [start]
    start_priority = manhattan_distance(start, goal)
    frontier = []
    heapq.heappush(frontier, (start_priority, start))
    g_score = {start: 0}
    parents = {start: None}
    while frontier:
        _, current = heapq.heappop(frontier)
        if current == goal:
            return reconstruct_path(goal, parents)
        for neighbor in get_neighbors(current, width, height, blocked):
            new_entry = update_neighbor_if_better(
                current, neighbor, goal, g_score, parents
            )
            if new_entry is not None:
                heapq.heappush(frontier, new_entry)

    return []  # 如果没有路径，则返回空列表
