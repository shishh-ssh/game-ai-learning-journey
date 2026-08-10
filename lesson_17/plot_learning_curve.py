"""生成第17课用的健康学习曲线教学图。"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


def main() -> None:
    episodes = np.arange(0, 301, 10)
    # 教学用模拟数据：先学习，后进入平台期，不代表真实训练结果。
    mean_return = -9.5 + 19.0 / (1.0 + np.exp(-(episodes - 125) / 32))
    mean_return += 0.12 * np.sin(episodes / 16)
    rolling_std = 0.35 + 0.45 * np.exp(-episodes / 90)

    figure, axis = plt.subplots(figsize=(8, 4.5), dpi=150)
    axis.plot(
        episodes,
        mean_return,
        color="#1769aa",
        linewidth=2.5,
        label="平均回报（教学模拟）",
    )
    axis.fill_between(
        episodes,
        mean_return - rolling_std,
        mean_return + rolling_std,
        color="#1769aa",
        alpha=0.16,
        label="波动范围（示意）",
    )
    axis.axvline(220, color="#d97706", linestyle="--", linewidth=1.5)
    axis.annotate(
        "平台期\n继续训练收益变小",
        xy=(220, mean_return[22]),
        xytext=(235, 6.4),
        arrowprops={"arrowstyle": "->", "color": "#d97706"},
        color="#9a5200",
    )
    axis.set_title("健康学习曲线：上升后趋于稳定")
    axis.set_xlabel("训练回合数")
    axis.set_ylabel("平均回报")
    axis.set_xlim(0, 300)
    axis.set_ylim(-11, 11)
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right")
    figure.tight_layout()

    output_path = Path(__file__).with_name("healthy_learning_curve.png")
    figure.savefig(output_path)
    print(output_path)


if __name__ == "__main__":
    main()
