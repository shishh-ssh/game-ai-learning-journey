"""生成第17课用的环境难度泛化曲线教学图。"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


def main() -> None:
    difficulty = np.arange(1, 11)
    success_rate = np.array([96, 95, 94, 92, 89, 80, 66, 48, 31, 19])

    figure, axis = plt.subplots(figsize=(8, 4.5), dpi=150)
    axis.plot(
        difficulty,
        success_rate,
        marker="o",
        color="#8b3a3a",
        linewidth=2.5,
        label="固定策略的评估成功率",
    )
    axis.axvline(5.5, color="#555555", linestyle="--", linewidth=1.2)
    axis.annotate(
        "超出训练分布后快速下降",
        xy=(8, success_rate[7]),
        xytext=(5.8, 55),
        arrowprops={"arrowstyle": "->", "color": "#555555"},
        color="#333333",
    )
    axis.set_title("泛化下降：环境难度增加后的成功率变化")
    axis.set_xlabel("环境难度（障碍/敌人/任务长度的综合等级）")
    axis.set_ylabel("成功率（%）")
    axis.set_xticks(difficulty)
    axis.set_ylim(0, 100)
    axis.grid(alpha=0.25)
    axis.legend(loc="lower left")
    figure.tight_layout()

    output_path = Path(__file__).with_name("generalization_drop.png")
    figure.savefig(output_path)
    print(output_path)


if __name__ == "__main__":
    main()
