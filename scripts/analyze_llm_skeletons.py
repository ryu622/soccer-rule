"""フェーズ3(LLM骨格 v1/v2)とフェーズ2ベースラインを並べて比較する可視化。

生成物:
    documents/figures/llm_vs_baseline_f1.png

uv run python scripts/analyze_llm_skeletons.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Hiragino Sans"
import numpy as np
import pandas as pd

FIG_DIR = "documents/figures"


def main() -> None:
    baseline = pd.read_csv("documents/results_summary.csv", index_col=0)
    v1 = pd.read_csv("documents/llm_skeleton_results_v1.csv")
    v2 = pd.read_csv("documents/llm_skeleton_results.csv")

    v1_summary = v1.groupby("skeleton")["f1"].agg(["mean", "std"])
    v2_summary = v2.groupby("skeleton")["f1"].agg(["mean", "std"])

    labels = []
    means = []
    stds = []
    colors = []

    for model in ["decision_tree(depth=3)", "logistic_regression", "xgboost"]:
        labels.append(model.replace("(depth=3)", "\n(depth=3)").replace("_", "\n"))
        means.append(baseline.loc[model, "f1_mean"])
        stds.append(baseline.loc[model, "f1_std"])
        colors.append("tab:gray")

    for skeleton, row in v1_summary.iterrows():
        labels.append(f"LLM v1\n{skeleton.replace('skeleton_', '')}\n(AND鎖)")
        means.append(row["mean"])
        stds.append(row["std"])
        colors.append("tab:red")

    for skeleton, row in v2_summary.iterrows():
        labels.append(f"LLM v2\n{skeleton.replace('skeleton_', '')}\n(多数決)")
        means.append(row["mean"])
        stds.append(row["std"])
        colors.append("tab:purple")

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=4, color=colors)
    for xi, m, s in zip(x, means, stds):
        ax.text(xi, m + s + 0.015, f"{m:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("F1 score (positive class = 仕掛けた)")
    ax.set_ylim(0, 0.85)
    ax.set_title("決定木・強ベースライン vs LLM骨格(v1: AND鎖 / v2: 多数決方式)")
    ax.axhline(baseline.loc["decision_tree(depth=3)", "f1_mean"], color="gray", linestyle="--", linewidth=1, alpha=0.6)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/llm_vs_baseline_f1.png", dpi=140)
    plt.close(fig)
    print("saved figure")


if __name__ == "__main__":
    main()
