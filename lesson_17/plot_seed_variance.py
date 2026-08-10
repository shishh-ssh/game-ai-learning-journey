"""生成第17课用的多随机种子差异教学图。"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


def main() -> None:
    episodes = np.arange(0, 301, 10)
    seed_1 = -8.5 + 18.0 / (1.0 + np.exp(-(episodes - 90) / 28))
    seed_2 = -8.5 + 15.0 / (1.0 + np.exp(-(episodes - 155) / 38))
    seed_3 = -8.5 + 5.5 / (1.0 + np.exp(-(episodes - 175) / 42))
    seed_1 += 0.22 * np.sin(episodes / 18)
    seed_2 += 0.38 * np.sin(episodes / 15)
    seed_3 += 0.30 * np.sin(episodes / 13)

    returns = np.vstack([seed_1, seed_2, seed_3])
    mean_return = returns.mean(axis=0)

    figure, axis = plt.subplots(figsize=(8, 4.5), dpi=150)
    colors = ["#1769aa", "#d97706", "#8b5cf6"]
    for index, values in enumerate(returns, start=1):
        axis.plot(
            episodes,
            values,
            color=colors[index - 1],
            linewidth=1.8,
            alpha=0.82,
            label=f"seed {index}",
        )
    axis.plot(
        episodes,
        mean_return,
        color="#222222",
        linewidth=2.8,
        linestyle="--",
        label="三个种子的平均值",
    )
    axis.annotate(
        "相同配置，不同种子得到不同结局",
        xy=(255, seed_3[25]),
        xytext=(118, -4.8),
        arrowprops={"arrowstyle": "->", "color": "#555555"},
        color="#333333",
    )
    axis.set_title("随机种子敏感：单次运行不能代表算法表现")
    axis.set_xlabel("训练回合数")
    axis.set_ylabel("平均回报")
    axis.set_xlim(0, 300)
    axis.set_ylim(-10, 11)
    axis.grid(alpha=0.25)
    axis.legend(loc="upper left")
    figure.tight_layout()

    output_path = Path(__file__).with_name("seed_variance.png")
    figure.savefig(output_path)
    print(output_path)


if __name__ == "__main__":
    main()
