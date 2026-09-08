"""1v1デュエル(仕掛けるvs待つ)タスクの決定木・強ベースラインを評価する。

research_plan.md 3.2節「評価」・3.3節「データ分割」に対応。
- 決定木(max_depth=3、解釈可能性重視のベースライン)
- ロジスティック回帰(線形の強ベースライン)
- XGBoost(非線形の強ベースライン)
を試合単位(leave-one-match-out)の交差検証でF1スコア比較する。

sync_error_suspect=True の行(distance_m>10mの正例、同期誤差ノイズの疑い)は
デフォルトで学習・評価から除外する(EXCLUDE_SYNC_SUSPECT参照)。

uv run python scripts/train_baselines.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text
from xgboost import XGBClassifier

DATA_PATH = "data/tackling_features.csv"
EXCLUDE_SYNC_SUSPECT = True

FEATURE_COLUMNS = [
    "distance_m",
    "approach_angle_deg",
    "carrier_speed_mps",
    "opponent_speed_mps",
    "closing_speed_mps",
    "second_nearest_dist_m",
    "n_supporting_teammates",
    "dist_to_sideline_m",
    "dist_to_goal_line_m",
]


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    if EXCLUDE_SYNC_SUSPECT:
        n_before = len(df)
        df = df[~df["sync_error_suspect"]].reset_index(drop=True)
        print(f"excluded {n_before - len(df)} sync_error_suspect rows -> {len(df)} rows remain")
    return df


def make_models(pos_weight: float) -> dict:
    return {
        "decision_tree(depth=3)": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("clf", DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=0)),
            ]
        ),
        "logistic_regression": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(class_weight="balanced", max_iter=1000)),
            ]
        ),
        "xgboost": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "clf",
                    XGBClassifier(
                        n_estimators=200,
                        max_depth=4,
                        learning_rate=0.05,
                        scale_pos_weight=pos_weight,
                        eval_metric="logloss",
                        random_state=0,
                    ),
                ),
            ]
        ),
    }


def leave_one_match_out_oof(df: pd.DataFrame) -> pd.DataFrame:
    """モデルごとに全foldのout-of-fold予測(確率つき)を1つのDataFrameに集約する。

    PR曲線・混同行列など、閾値やfold単位の集計を後段で自由に行うための生の予測結果。
    """
    X = df[FEATURE_COLUMNS]
    y = df["label"].values
    matches = df["match_id"].unique()

    rows = []
    for match_id in matches:
        train_mask = df["match_id"] != match_id
        test_mask = ~train_mask

        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        n_pos, n_neg = (y_train == 1).sum(), (y_train == 0).sum()
        pos_weight = n_neg / max(n_pos, 1)

        for name, model in make_models(pos_weight).items():
            model.fit(X_train, y_train)
            proba = model.predict_proba(X_test)[:, 1]
            pred = model.predict(X_test)
            for yt, yp, pp in zip(y_test, pred, proba):
                rows.append({"held_out_match": match_id, "model": name, "y_true": yt, "y_pred": yp, "y_proba": pp})
    return pd.DataFrame(rows)


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

        for name, model in make_models(pos_weight).items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            results.append(
                {
                    "held_out_match": match_id,
                    "model": name,
                    "n_test": len(y_test),
                    "f1": f1_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred, zero_division=0),
                    "recall": recall_score(y_test, y_pred, zero_division=0),
                }
            )
    return pd.DataFrame(results)


def print_example_tree(df: pd.DataFrame) -> None:
    X = df[FEATURE_COLUMNS]
    y = df["label"].values
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)
    tree = DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=0)
    tree.fit(X_imputed, y)
    print()
    print("=== decision tree trained on full data (for inspection only) ===")
    print(export_text(tree, feature_names=FEATURE_COLUMNS))


def main() -> None:
    df = load_dataset()
    print(f"positives={int((df.label==1).sum())} negatives={int((df.label==0).sum())}")
    print()

    results = leave_one_match_out_eval(df)

    summary = results.groupby("model")[["f1", "precision", "recall"]].agg(["mean", "std"])
    print("=== leave-one-match-out cross-validation (7 folds) ===")
    print(summary)
    print()
    print("=== per-fold F1 ===")
    print(results.pivot(index="held_out_match", columns="model", values="f1").round(3))

    print_example_tree(df)


if __name__ == "__main__":
    main()
