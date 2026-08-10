"""第 1 课：用纯函数实现文字角色移动。"""

COMMAND_DELTAS: dict[str, tuple[int, int]] = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
    "stay": (0, 0),
}


def _validate_world_state(
    position: tuple[int, int],
    width: int,
    height: int,
) -> None:
    """校验地图尺寸和角色位置。"""

    if width <= 0 or height <= 0:
        raise ValueError("宽度和高度必须为正数")

    x, y = position
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError("角色位置必须在地图范围内")


def move_character(
    position: tuple[int, int],
    command: str,
    width: int,
    height: int,
) -> tuple[int, int]:
    """执行一个动作并返回角色的新位置。"""
    _validate_world_state(position, width, height)
    if command not in COMMAND_DELTAS:
        raise ValueError(f"未知的动作: {command}")
    delta_x, delta_y = COMMAND_DELTAS[command]
    next_x = position[0] + delta_x
    next_y = position[1] + delta_y
    if not (0 <= next_x < width and 0 <= next_y < height):
        return position
    return (next_x, next_y)


def execute_commands(
    start_position: tuple[int, int],
    commands: list[str],
    width: int,
    height: int,
) -> list[tuple[int, int]]:
    """依次执行动作并返回包含起点的完整轨迹。"""
    _validate_world_state(start_position, width, height)
    current_position = start_position
    trajectory = [current_position]
    for command in commands:
        current_position = move_character(current_position, command, width, height)
        trajectory.append(current_position)
    return trajectory
