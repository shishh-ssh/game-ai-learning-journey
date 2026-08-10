"""第 4 课：连续处理多帧观测并保存 NPC 状态。"""

from npc_fsm import transition_state


def run_state_sequence(
    initial_state: str,
    observations: list[tuple[int, bool, float]],
) -> list[str]:
    """返回包含初始状态和每帧转换结果的完整状态历史。"""
    history_states = [initial_state]
    current_state = initial_state
    for health, enemy_visible, distance in observations:
        return_state = transition_state(current_state, health, enemy_visible, distance)
        history_states.append(return_state)
        current_state = return_state
    return history_states
