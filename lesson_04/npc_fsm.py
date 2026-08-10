"""第 4 课：使用有限状态机管理 NPC 行为模式。"""

VALID_STATES = {"patrol", "chase", "attack", "flee", "dead"}


def transition_state(
    current_state: str,
    health: int,
    enemy_visible: bool,
    distance: float,
) -> str:
    """根据当前状态和观测条件返回下一帧状态。"""
    if current_state not in VALID_STATES:
        raise ValueError("当前状态不属于 FSM 的合法状态。")
    if not 0 <= health <= 100:
        raise ValueError("生命值必须位于 0 到 100 之间。")
    if distance < 0:
        raise ValueError("敌人距离不能为负数。")
    if current_state == "dead":
        return "dead"
    if health == 0:
        return "dead"
    if health < 30:
        return "flee"
    if current_state == "flee":
        if health >= 50:
            return "patrol"
        return "flee"
    if current_state == "patrol":
        if enemy_visible and distance <= 6:
            return "chase"
        return "patrol"
    if current_state == "chase":
        if not enemy_visible:
            return "patrol"
        if distance <= 2:
            return "attack"
        return "chase"
    if current_state == "attack":
        if not enemy_visible:
            return "patrol"
        if distance > 2:
            return "chase"
        return "attack"
    raise NotImplementedError("请分步骤实现 FSM 状态转换。")
