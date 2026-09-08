"""フェーズ2ベースライン結果の可視化・分析用スクリプト。

生成物:
    documents/figures/f1_comparison.png       モデル別F1比較(誤差棒つき)
    documents/figures/f1_per_match.png        試合ごとのF1(leave-one-match-out各fold)
    documents/figures/pr_curve.png            集約out-of-fold予測によるPrecision-Recall曲線
    documents/figures/confusion_matrices.png  閾値0.5での混同行列(3モデル)
    documents/figures/decision_tree.png       決定木(depth=3, 全データ学習)の可視化
    documents/figures/feature_importance.png  決定木 vs XGBoostの特徴量重要度比較
    documents/results_summary.csv             モデル別集計結果(表としてmdに埋め込み用)

uv run python scripts/analyze_baselines.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Hiragino Sans"
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.tree import DecisionTreeClassifier, plot_tree

from train_baselines import (
    FEATURE_COLUMNS,
    load_dataset,
    leave_one_match_out_eval,
    leave_one_match_out_oof,
)

FIG_DIR = "documents/figures"
MODEL_ORDER = ["decision_tree(depth=3)", "logistic_regression", "xgboost"]
MODEL_COLORS = {"decision_tree(depth=3)": "tab:orange", "logistic_regression": "tab:blue", "xgboost": "tab:green"}
MODEL_LABELS = {"decision_tree(depth=3)": "Decision Tree\n(depth=3)", "logistic_regression": "Logistic\nRegression", "xgboost": "XGBoost"}


def plot_f1_comparison(results: pd.DataFrame) -> None:
    summary = results.groupby("model")[["f1", "precision", "recall"]].agg(["mean", "std"]).loc[MODEL_ORDER]

    fig, ax = plt.subplots(figsize=(7, 4.8))
    x = np.arange(len(MODEL_ORDER))
    means = summary[("f1", "mean")].values
    stds = summary[("f1", "std")].values
    colors = [MODEL_COLORS[m] for m in MODEL_ORDER]
    ax.bar(x, means, yerr=stds, capsize=6, color=colors)
    for xi, m, s in zip(x, means, stds):
        ax.text(xi, m + s + 0.02, f"{m:.3f}", ha="center", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER])
    ax.set_ylabel("F1 score (positive class = 仕掛けた)")
    ax.set_ylim(0, 0.85)
    ax.set_title("Leave-one-match-out CV: F1 comparison\n(mean ± std over 7 folds)")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/f1_comparison.png", dpi=140)
    plt.close(fig)


def plot_f1_per_match(results: pd.DataFrame) -> None:
    pivot = results.pivot(index="held_out_match", columns="model", values="f1")[MODEL_ORDER]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(pivot.index))
    width = 0.25
    for i, model in enumerate(MODEL_ORDER):
        ax.bar(x + (i - 1) * width, pivot[model].values, width=width, label=MODEL_LABELS[model].replace("\n", " "), color=MODEL_COLORS[model])
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=0)
    ax.set_ylabel("F1 score")
    ax.set_xlabel("held-out match (leave-one-match-out)")
    ax.set_title("F1 per held-out match")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/f1_per_match.png", dpi=140)
    plt.close(fig)


def plot_pr_curve(oof: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for model in MODEL_ORDER:
        sub = oof[oof.model == model]
        precision, recall, _ = precision_recall_curve(sub.y_true, sub.y_proba)
        ax.plot(recall, precision, label=MODEL_LABELS[model].replace("\n", " "), color=MODEL_COLORS[model])
    base_rate = oof[oof.model == MODEL_ORDER[0]].y_true.mean()
    ax.axhline(base_rate, color="gray", linestyle="--", linewidth=1, label=f"chance (base rate={base_rate:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve (pooled out-of-fold predictions)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/pr_curve.png", dpi=140)
    plt.close(fig)


def plot_confusion_matrices(oof: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, model in zip(axes, MODEL_ORDER):
        sub = oof[oof.model == model]
        cm = confusion_matrix(sub.y_true, sub.y_pred, labels=[0, 1])
        im = ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black", fontsize=12)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["pred: 待った", "pred: 仕掛けた"])
        ax.set_yticklabels(["true: 待った", "true: 仕掛けた"])
        ax.set_title(MODEL_LABELS[model].replace("\n", " "))
    fig.suptitle("Confusion matrix (threshold=0.5, pooled out-of-fold predictions)")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/confusion_matrices.png", dpi=140)
    plt.close(fig)


def plot_decision_tree(df: pd.DataFrame) -> DecisionTreeClassifier:
    X = df[FEATURE_COLUMNS]
    y = df["label"].values
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)
    tree = DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=0)
    tree.fit(X_imputed, y)

    fig, ax = plt.subplots(figsize=(16, 8))
    plot_tree(
        tree,
        feature_names=FEATURE_COLUMNS,
        class_names=["待った", "仕掛けた"],
        filled=True,
        rounded=True,
        fontsize=9,
        ax=ax,
    )
    ax.set_title("Decision tree (max_depth=3, trained on full data, class_weight=balanced)")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/decision_tree.png", dpi=140)
    plt.close(fig)
    return tree


def plot_feature_importance(df: pd.DataFrame, tree: DecisionTreeClassifier) -> None:
    from xgboost import XGBClassifier

    X = df[FEATURE_COLUMNS]
    y = df["label"].values
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)
    xgb = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, scale_pos_weight=pos_weight, eval_metric="logloss", random_state=0)
    xgb.fit(X_imputed, y)

    tree_imp = pd.Series(tree.feature_importances_, index=FEATURE_COLUMNS)
    xgb_imp = pd.Series(xgb.feature_importances_, index=FEATURE_COLUMNS)

    order = xgb_imp.sort_values(ascending=True).index
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    axes[0].barh(order, tree_imp[order], color=MODEL_COLORS["decision_tree(depth=3)"])
    axes[0].set_title("Decision Tree feature importance")
    axes[1].barh(order, xgb_imp[order], color=MODEL_COLORS["xgboost"])
    axes[1].set_title("XGBoost feature importance")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/feature_importance.png", dpi=140)
    plt.close(fig)


def main() -> None:
    df = load_dataset()
    results = leave_one_match_out_eval(df)
    oof = leave_one_match_out_oof(df)

    summary = results.groupby("model")[["f1", "precision", "recall"]].agg(["mean", "std"]).loc[MODEL_ORDER]
    summary.columns = ["_".join(c) for c in summary.columns]
    summary.to_csv("documents/results_summary.csv")
    print(summary)

    plot_f1_comparison(results)
    plot_f1_per_match(results)
    plot_pr_curve(oof)
    plot_confusion_matrices(oof)
    tree = plot_decision_tree(df)
    plot_feature_importance(df, tree)

    print("figures saved to", FIG_DIR)


if __name__ == "__main__":
    main()
