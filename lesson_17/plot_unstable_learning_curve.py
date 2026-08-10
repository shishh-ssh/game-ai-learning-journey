"""生成第17课用的训练震荡曲线教学图。"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


def main() -> None:
    episodes = np.arange(0, 301, 5)
    trend = -7.5 + 12.5 / (1.0 + np.exp(-(episodes - 120) / 38))
    oscillation = 2.1 * np.sin(episodes / 9) + 1.25 * np.sin(episodes / 3.8)
    mean_return = trend + oscillation
    rolling_std = 1.35 + 0.45 * np.sin(episodes / 18) ** 2

    figure, axis = plt.subplots(figsize=(8, 4.5), dpi=150)
    axis.plot(
        episodes,
        mean_return,
        color="#7a3db8",
        linewidth=2.2,
        label="平均回报（教学模拟）",
    )
    axis.fill_between(
        episodes,
        mean_return - rolling_std,
        mean_return + rolling_std,
        color="#7a3db8",
        alpha=0.15,
        label="波动范围（示意）",
    )
    axis.annotate(
        "回报提高后仍反复大幅下降",
        xy=(235, mean_return[47]),
        xytext=(130, -4.8),
        arrowprops={"arrowstyle": "->", "color": "#555555"},
        color="#333333",
    )
    axis.set_title("训练不稳定：平均回报持续剧烈震荡")
    axis.set_xlabel("训练回合数")
    axis.set_ylabel("平均回报")
    axis.set_xlim(0, 300)
    axis.set_ylim(-12, 10)
    axis.grid(alpha=0.25)
    axis.legend(loc="upper left")
    figure.tight_layout()

    output_path = Path(__file__).with_name("unstable_learning_curve.png")
    figure.savefig(output_path)
    print(output_path)


if __name__ == "__main__":
    main()
