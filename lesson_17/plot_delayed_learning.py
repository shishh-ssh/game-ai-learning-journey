"""生成第17课用的延迟学习曲线教学图。"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


def main() -> None:
    episodes = np.arange(0, 301, 10)
    mean_return = -8.6 + 17.5 / (1.0 + np.exp(-(episodes - 205) / 13))
    mean_return += 0.12 * np.sin(episodes / 15)

    figure, axis = plt.subplots(figsize=(8, 4.5), dpi=150)
    axis.plot(
        episodes,
        mean_return,
        color="#147d64",
        linewidth=2.5,
        label="平均回报（教学模拟）",
    )
    axis.axvline(185, color="#555555", linestyle="--", linewidth=1.2)
    axis.annotate(
        "首次获得关键成功经验后快速改善",
        xy=(215, mean_return[21]),
        xytext=(88, 1.5),
        arrowprops={"arrowstyle": "->", "color": "#555555"},
        color="#333333",
    )
    axis.set_title("延迟学习：长期低回报后突然改善")
    axis.set_xlabel("训练回合数")
    axis.set_ylabel("平均回报")
    axis.set_xlim(0, 300)
    axis.set_ylim(-10, 11)
    axis.grid(alpha=0.25)
    axis.legend(loc="upper left")
    figure.tight_layout()

    output_path = Path(__file__).with_name("delayed_learning.png")
    figure.savefig(output_path)
    print(output_path)


if __name__ == "__main__":
    main()
