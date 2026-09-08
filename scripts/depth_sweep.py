"""決定木のmax_depthを振って、7-fold leave-one-match-out F1がXGBoostにどこまで近づくか確認する。

ユーザーの仮説検証用:
    depth=5〜8あたりでXGBoost(F1=0.634)にほぼ追いつく(0.60台後半に達する)なら、
    「深さの制約さえなければ軸並行分割でも十分」=タスク選択側の問題という説を補強する。
    差が大きく残るなら、複合条件(LLM骨格・進化計算)に意味がある余地が残っていることになる。

phase2のtrain_baselines.pyと同じデータ・特徴量・分割・class_weight="balanced"を使う。
XGBoostの数値もfin参照用に同じ設定で再計算する(phase2の0.634と一致するはず)。

uv run python scripts/depth_sweep.py
"""

from __future__ import annotations

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from train_baselines import FEATURE_COLUMNS, load_dataset

DEPTHS = [3, 5, 8, None]


def make_tree(max_depth: int | None) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("clf", DecisionTreeClassifier(max_depth=max_depth, class_weight="balanced", random_state=0)),
        ]
    )


def make_xgb(pos_weight: float) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            (
                "clf",
                XGBClassifier(
                    n_estimators=200, max_depth=4, learning_rate=0.05,
                    scale_pos_weight=pos_weight, eval_metric="logloss", random_state=0,
                ),
            ),
        ]
    )


def leave_one_match_out_eval(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURE_COLUMNS]
    y = df["label"].values
    matches = df["match_id"].unique()

    results = []
    for match_id in matches:
        train_mask = df["match_id"] != match_id
        test_mask = ~train_mask
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        n_pos, n_neg = (y_train == 1).sum(), (y_train == 0).sum()
        pos_weight = n_neg / max(n_pos, 1)

        models = {f"decision_tree(depth={d})": make_tree(d) for d in DEPTHS}
        models["xgboost"] = make_xgb(pos_weight)

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            n_leaves = None
            if "decision_tree" in name:
                n_leaves = model.named_steps["clf"].get_n_leaves()
                actual_depth = model.named_steps["clf"].get_depth()
            results.append(
                {
                    "held_out_match": match_id,
                    "model": name,
                    "f1": f1_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred, zero_division=0),
                    "recall": recall_score(y_test, y_pred, zero_division=0),
                    "n_leaves": n_leaves,
                    "actual_depth": actual_depth if "decision_tree" in name else None,
                }
            )
    return pd.DataFrame(results)


def main() -> None:
    df = load_dataset()
    results = leave_one_match_out_eval(df)

    order = [f"decision_tree(depth={d})" for d in DEPTHS] + ["xgboost"]
    summary = results.groupby("model")[["f1", "precision", "recall"]].agg(["mean", "std"]).loc[order]
    print(summary)
    print()

    leaf_info = results[results.n_leaves.notna()].groupby("model")[["n_leaves", "actual_depth"]].mean()
    print("decision tree complexity (mean over folds):")
    print(leaf_info.loc[[f"decision_tree(depth={d})" for d in DEPTHS]])
    print()

    print("per-fold F1:")
    print(results.pivot(index="held_out_match", columns="model", values="f1")[order].round(3))

    summary.columns = ["_".join(c) for c in summary.columns]
    summary.to_csv("documents/depth_sweep_results.csv")
    results.to_csv("documents/depth_sweep_per_fold.csv", index=False)
    print()
    print("saved documents/depth_sweep_results.csv, documents/depth_sweep_per_fold.csv")


if __name__ == "__main__":
    main()
