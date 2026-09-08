"""拡張特徴量セット(build_features_v2.py、元9特徴量+トレンド/加速度/方向転換率/
3番目の相手/密集度/直前イベント履歴の7特徴量=計16特徴量)で、決定木のmax_depthを振って
XGBoostとの差がどう動くかを見る診断。

レイヤー2/3切り分け実験(phase3b_evolution_results.md 3.4.3/3.4.4節の議論を参照):
    特徴量を増やしても決定木-XGBoostの差が縮まらない/変わらない
        -> レイヤー2寄り(1v1というタスク自体が本質的にシンプル)
    特徴量を増やしたらXGBoostの優位が広がる(決定木が追いつけなくなる)
        -> レイヤー3寄り(元の特徴量セットが薄かった)

depth_sweep.pyと同じ手法(class_weight="balanced"、7-fold leave-one-match-out、
sync_error_suspect除外)を、特徴量セットとデータだけ差し替えて使う。

uv run python scripts/depth_sweep_v2.py
"""

from __future__ import annotations

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

DATA_PATH = "data/tackling_features_v2.csv"
EXCLUDE_SYNC_SUSPECT = True
DEPTHS = [3, 5, 8, None]

FEATURE_COLUMNS_ORIGINAL = [
    "distance_m", "approach_angle_deg", "carrier_speed_mps", "opponent_speed_mps",
    "closing_speed_mps", "second_nearest_dist_m", "n_supporting_teammates",
    "dist_to_sideline_m", "dist_to_goal_line_m",
]
FEATURE_COLUMNS_NEW = [
    "distance_trend_1s_mps", "closing_accel_mps2", "opponent_heading_change_deg",
    "third_nearest_dist_m", "n_opponents_nearby_10m", "seconds_since_prev_event", "prev_event_was_pass",
]
FEATURE_COLUMNS = FEATURE_COLUMNS_ORIGINAL + FEATURE_COLUMNS_NEW


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    if EXCLUDE_SYNC_SUSPECT:
        n_before = len(df)
        df = df[~df["sync_error_suspect"]].reset_index(drop=True)
        print(f"excluded {n_before - len(df)} sync_error_suspect rows -> {len(df)} rows remain")
    return df


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


def leave_one_match_out_eval(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    X = df[feature_columns]
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
            results.append(
                {
                    "held_out_match": match_id,
                    "model": name,
                    "f1": f1_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred, zero_division=0),
                    "recall": recall_score(y_test, y_pred, zero_division=0),
                }
            )
    return pd.DataFrame(results)


def main() -> None:
    df = load_dataset()

    order = [f"decision_tree(depth={d})" for d in DEPTHS] + ["xgboost"]

    print(f"=== 元の9特徴量のみ(参照用、data/tackling_features_v2.csvの同一行で再計算) ===")
    results_orig = leave_one_match_out_eval(df, FEATURE_COLUMNS_ORIGINAL)
    summary_orig = results_orig.groupby("model")[["f1", "precision", "recall"]].agg(["mean", "std"]).loc[order]
    print(summary_orig)
    print()

    print(f"=== 拡張16特徴量(元9 + トレンド/加速度/方向転換/3番目相手/密集度/直前イベント) ===")
    results_new = leave_one_match_out_eval(df, FEATURE_COLUMNS)
    summary_new = results_new.groupby("model")[["f1", "precision", "recall"]].agg(["mean", "std"]).loc[order]
    print(summary_new)
    print()

    gap_orig = summary_orig.loc["xgboost", ("f1", "mean")] - summary_orig.loc["decision_tree(depth=3)", ("f1", "mean")]
    gap_new = summary_new.loc["xgboost", ("f1", "mean")] - summary_new.loc["decision_tree(depth=3)", ("f1", "mean")]
    peak_orig = summary_orig.loc[[f"decision_tree(depth={d})" for d in DEPTHS], ("f1", "mean")].max()
    peak_new = summary_new.loc[[f"decision_tree(depth={d})" for d in DEPTHS], ("f1", "mean")].max()
    print(f"決定木(depth=3)-XGBoost のF1差: 元9特徴量={gap_orig:.3f} -> 拡張16特徴量={gap_new:.3f}")
    print(f"決定木のピークF1(depth振り): 元9特徴量={peak_orig:.3f} -> 拡張16特徴量={peak_new:.3f}")
    print(f"XGBoostのF1: 元9特徴量={summary_orig.loc['xgboost', ('f1','mean')]:.3f} -> 拡張16特徴量={summary_new.loc['xgboost', ('f1','mean')]:.3f}")

    summary_orig.columns = ["_".join(c) for c in summary_orig.columns]
    summary_new.columns = ["_".join(c) for c in summary_new.columns]
    summary_orig.to_csv("documents/depth_sweep_v2_original_features.csv")
    summary_new.to_csv("documents/depth_sweep_v2_expanded_features.csv")
    print("\nsaved documents/depth_sweep_v2_original_features.csv, documents/depth_sweep_v2_expanded_features.csv")


if __name__ == "__main__":
    main()
