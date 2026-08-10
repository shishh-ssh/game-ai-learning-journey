"""生成第17课用的训练与评估差距教学图。"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


def main() -> None:
    episodes = np.arange(0, 301, 10)
    train_return = -8.5 + 18.0 / (1.0 + np.exp(-(episodes - 115) / 34))
    evaluation_return = -8.4 + 11.5 / (1.0 + np.exp(-(episodes - 105) / 31))
    evaluation_return -= np.maximum(episodes - 185, 0) * 0.022

    figure, axis = plt.subplots(figsize=(8, 4.5), dpi=150)
    axis.plot(
        episodes,
        train_return,
        color="#1769aa",
        linewidth=2.5,
        label="训练平均回报",
    )
    axis.plot(
        episodes,
        evaluation_return,
        color="#d97706",
        linewidth=2.5,
        label="评估平均回报",
    )
    axis.axvline(185, color="#666666", linestyle="--", linewidth=1.2)
    axis.annotate(
        "训练继续改善，评估开始下降",
        xy=(230, evaluation_return[23]),
        xytext=(128, -5.4),
        arrowprops={"arrowstyle": "->", "color": "#555555"},
        color="#333333",
    )
    axis.set_title("训练与评估曲线分离：可能出现过拟合")
    axis.set_xlabel("训练回合数")
    axis.set_ylabel("平均回报")
    axis.set_xlim(0, 300)
    axis.set_ylim(-10, 11)
    axis.grid(alpha=0.25)
    axis.legend(loc="upper left")
    figure.tight_layout()

    output_path = Path(__file__).with_name("train_eval_gap.png")
    figure.savefig(output_path)
    print(output_path)


if __name__ == "__main__":
    main()
