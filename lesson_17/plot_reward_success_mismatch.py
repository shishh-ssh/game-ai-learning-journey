"""生成第17课用的奖励与成功率不一致教学图。"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


def main() -> None:
    episodes = np.arange(0, 301, 10)
    mean_reward = 15 + 70 / (1.0 + np.exp(-(episodes - 120) / 45))
    success_rate = 28 + 3.5 / (1.0 + np.exp(-(episodes - 125) / 40))

    figure, axis = plt.subplots(figsize=(8, 4.5), dpi=150)
    axis.plot(
        episodes,
        mean_reward,
        color="#b42318",
        linewidth=2.5,
        label="平均回报（归一化）",
    )
    axis.plot(
        episodes,
        success_rate,
        color="#1769aa",
        linewidth=2.5,
        label="成功率（%）",
    )
    axis.annotate(
        "回报上升，但成功率几乎不变",
        xy=(225, success_rate[22]),
        xytext=(110, 45),
        arrowprops={"arrowstyle": "->", "color": "#555555"},
        color="#333333",
    )
    axis.set_title("指标不一致：可能存在奖励投机或奖励设计偏差")
    axis.set_xlabel("训练回合数")
    axis.set_ylabel("百分比 / 归一化数值")
    axis.set_xlim(0, 300)
    axis.set_ylim(0, 100)
    axis.grid(alpha=0.25)
    axis.legend(loc="center right")
    figure.tight_layout()

    output_path = Path(__file__).with_name("reward_success_mismatch.png")
    figure.savefig(output_path)
    print(output_path)


if __name__ == "__main__":
    main()
