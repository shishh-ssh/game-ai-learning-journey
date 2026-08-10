"""第 2 课补充练习：嵌套循环与提前返回。"""
from collections import deque


def collect_actions_until_stop(turns: list[list[str]]) -> list[str]:
    """按回合收集动作，遇到 stop 时立即返回已经收集的动作。"""
    turns = deque(turns)
    actions = []
    while turns:
        other = turns.popleft()
        for action in other:
            if action == "stop":
                return actions
            actions.append(action)
    return actions
