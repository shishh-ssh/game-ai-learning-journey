"""第 3 课补充练习：最小堆优先队列。"""

import heapq


def pop_positions_by_priority(
    entries: list[tuple[int, str]],
) -> list[str]:
    """按照优先级从小到大取出位置名称。"""
    frontier = []
    for entry in entries:
        heapq.heappush(frontier, entry)
    result = []
    while frontier:
        priority, position_name = heapq.heappop(frontier)
        result.append(position_name)
    return result
