"""生成第17课用的过早平台曲线教学图。"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


def main() -> None:
    episodes = np.arange(0, 301, 10)
    mean_return = -8.5 + 8.0 / (1.0 + np.exp(-(episodes - 65) / 18))
    mean_return += 0.12 * np.sin(episodes / 17)

    figure, axis = plt.subplots(figsize=(8, 4.5), dpi=150)
    axis.plot(
        episodes,
        mean_return,
        color="#2563a6",
        linewidth=2.5,
        label="平均回报（教学模拟）",
    )
    axis.axhline(8.0, color="#2f855a", linestyle="--", linewidth=1.4,
                 label="可达到的较优水平（参考）")
    axis.annotate(
        "较早停止改善，但仍远低于参考水平",
        xy=(185, mean_return[18]),
        xytext=(105, 3.0),
        arrowprops={"arrowstyle": "->", "color": "#555555"},
        color="#333333",
    )
    axis.set_title("过早平台：曲线稳定不等于策略足够好")
    axis.set_xlabel("训练回合数")
    axis.set_ylabel("平均回报")
    axis.set_xlim(0, 300)
    axis.set_ylim(-10, 10)
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right")
    figure.tight_layout()

    output_path = Path(__file__).with_name("premature_plateau.png")
    figure.savefig(output_path)
    print(output_path)


if __name__ == "__main__":
    main()
