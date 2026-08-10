"""第 2 课：在带障碍物的网格中搜索最短路径。"""

from collections import deque


def get_neighbors(
    position: tuple[int, int],
    width: int,
    height: int,
    blocked: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    """按上、下、左、右顺序返回可行的相邻位置。"""

    if width <= 0 or height <= 0:
        raise ValueError("网格的宽度和高度必须大于零。")

    x, y = position
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError("位置出错，不在网格范围内。")
    if position in blocked:
        raise ValueError("位置被阻挡。")

    candidate_positions = [
        (x, y - 1),
        (x, y + 1),
        (x - 1, y),
        (x + 1, y),
    ]

    # 只保留未越界且未被阻挡的候选位置。
    valid_positions = []
    for candidate_position in candidate_positions:
        candidate_x, candidate_y = candidate_position
        if (
            0 <= candidate_x < width
            and 0 <= candidate_y < height
            and candidate_position not in blocked
        ):
            valid_positions.append(candidate_position)

    return valid_positions


def find_shortest_path(
    start: tuple[int, int],
    goal: tuple[int, int],
    width: int,
    height: int,
    blocked: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    """使用 BFS 返回包含起点和终点的最短路径。"""

    if width <= 0 or height <= 0:
        raise ValueError("地图大小错误。")

    start_x, start_y = start
    goal_x, goal_y = goal
    if not (0 <= start_x < width and 0 <= start_y < height):
        raise ValueError("起点位置错误。")
    if not (0 <= goal_x < width and 0 <= goal_y < height):
        raise ValueError("终点位置错误。")
    if start in blocked:
        raise ValueError("起点被阻挡。")
    if goal in blocked:
        raise ValueError("终点被阻挡。")
    if start == goal:
        return [start]

    frontier = deque([start])
    visited = {start}
    parents = {start: None}

    while frontier:
        current = frontier.popleft()
        if current == goal:
            # 目标出队时，parents 已记录一条最短路径。
            path = []
            path_position = current
            while path_position is not None:
                path.append(path_position)
                path_position = parents[path_position]
            path.reverse()
            return path
        for neighbor in get_neighbors(current, width, height, blocked):
            if neighbor not in visited:
                visited.add(neighbor)
                parents[neighbor] = current
                frontier.append(neighbor)

    return []
