"""第一个编码练习：确定性的 NPC 决策策略。"""


def choose_action(
    distance: float,
    health: int,
    enemy_visible: bool,
) -> str:
    """根据课程规则返回 NPC 行为。"""

    if distance < 0:
        raise ValueError("距离不能为负数")
    if health < 0 or health > 100:
        raise ValueError("健康值必须在0到100之间")
    if health == 0:
        return "dead"
    if not enemy_visible:
        return "patrol"
    if health <= 30:
        return "flee"
    if distance <= 2:
        return "attack"
    return "chase"
