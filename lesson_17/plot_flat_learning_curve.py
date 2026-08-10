"""生成第17课用的未学习成功曲线教学图。"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


def main() -> None:
    episodes = np.arange(0, 301, 10)
    # 教学用模拟数据：回报长期低位波动，不代表真实训练结果。
    mean_return = -8.4 + 0.35 * np.sin(episodes / 13)
    mean_return += 0.18 * np.cos(episodes / 7)
    rolling_std = 0.9 + 0.15 * np.sin(episodes / 25) ** 2

    figure, axis = plt.subplots(figsize=(8, 4.5), dpi=150)
    axis.plot(
        episodes,
        mean_return,
        color="#b42318",
        linewidth=2.5,
        label="平均回报（教学模拟）",
    )
    axis.fill_between(
        episodes,
        mean_return - rolling_std,
        mean_return + rolling_std,
        color="#b42318",
        alpha=0.16,
        label="波动范围（示意）",
    )
    axis.axhline(-8.4, color="#555555", linestyle="--", linewidth=1.2)
    axis.annotate(
        "长期停留在低回报区域",
        xy=(180, mean_return[18]),
        xytext=(120, -5.7),
        arrowprops={"arrowstyle": "->", "color": "#555555"},
        color="#333333",
    )
    axis.set_title("未学习成功：平均回报长期低位波动")
    axis.set_xlabel("训练回合数")
    axis.set_ylabel("平均回报")
    axis.set_xlim(0, 300)
    axis.set_ylim(-11, 1)
    axis.grid(alpha=0.25)
    axis.legend(loc="upper right")
    figure.tight_layout()

    output_path = Path(__file__).with_name("flat_learning_curve.png")
    figure.savefig(output_path)
    print(output_path)


if __name__ == "__main__":
    main()
