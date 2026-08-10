"""生成第17课用的训练后期崩溃曲线教学图。"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


def main() -> None:
    episodes = np.arange(0, 301, 10)
    learned = -8.5 + 17.0 / (1.0 + np.exp(-(episodes - 100) / 28))
    collapse = np.maximum(episodes - 215, 0) * 0.15
    mean_return = learned - collapse + 0.18 * np.sin(episodes / 14)

    figure, axis = plt.subplots(figsize=(8, 4.5), dpi=150)
    axis.plot(
        episodes,
        mean_return,
        color="#b42318",
        linewidth=2.5,
        label="平均回报（教学模拟）",
    )
    axis.axvline(215, color="#555555", linestyle="--", linewidth=1.2)
    axis.annotate(
        "已学会的策略在后期快速退化",
        xy=(260, mean_return[26]),
        xytext=(125, -4.5),
        arrowprops={"arrowstyle": "->", "color": "#555555"},
        color="#333333",
    )
    axis.set_title("训练后期崩溃：回报先收敛后快速下降")
    axis.set_xlabel("训练回合数")
    axis.set_ylabel("平均回报")
    axis.set_xlim(0, 300)
    axis.set_ylim(-10, 11)
    axis.grid(alpha=0.25)
    axis.legend(loc="upper left")
    figure.tight_layout()

    output_path = Path(__file__).with_name("training_collapse.png")
    figure.savefig(output_path)
    print(output_path)


if __name__ == "__main__":
    main()
