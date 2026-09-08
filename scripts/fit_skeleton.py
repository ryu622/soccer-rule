"""LLM骨格(predict_take_on)のパラメータをデータに最適化し、
leave-one-match-out CVでF1を評価する。

research_plan.md 3.2節「データ最適化の役割」・3.3節「LLMコード生成」の
try-exceptラッパーに対応。

各fold(held-out match)ごとに:
    1. 学習用6試合でランダムサーチ+座標降下法によりPARAM_SPECSの範囲内でF1を最大化するパラメータを探す
    2. held-out試合でそのパラメータのF1を評価する
を llm_skeletons/skeleton_seed*.py それぞれについて行う。

uv run python scripts/fit_skeleton.py
"""

from __future__ import annotations

import glob
import random

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from train_baselines import FEATURE_COLUMNS, load_dataset

SKELETON_GLOB = "llm_skeletons/skeleton_seed*.py"
N_RANDOM_SEARCH = 400
N_REFINE_ROUNDS = 5
RNG_SEED = 0

SAFE_BUILTINS = {
    "abs": abs,
    "min": min,
    "max": max,
    "bool": bool,
    "int": int,
    "float": float,
    "len": len,
}


def load_skeleton(path: str) -> tuple[dict, callable]:
    with open(path) as f:
        code = f.read()
    namespace: dict = {"__builtins__": SAFE_BUILTINS}
    exec(code, namespace)
    if "PARAM_SPECS" not in namespace or "predict_take_on" not in namespace:
        raise ValueError(f"{path}: PARAM_SPECS または predict_take_on が定義されていません")
    return namespace["PARAM_SPECS"], namespace["predict_take_on"]


def to_records(df: pd.DataFrame) -> list[dict]:
    return df[FEATURE_COLUMNS].to_dict("records")


def safe_predict(fn, row: dict, params: dict, error_counter: list[int] | None = None) -> int:
    try:
        result = fn(row, params)
        return 1 if result else 0
    except Exception:
        if error_counter is not None:
            error_counter[0] += 1
        return 0  # フェイルセーフ: 例外時は「仕掛けない」として扱う


def evaluate_params(fn, rows: list[dict], y, params: dict) -> float:
    preds = [safe_predict(fn, row, params) for row in rows]
    return f1_score(y, preds, zero_division=0)


def optimize_params(fn, rows: list[dict], y, param_specs: dict, rng: random.Random) -> tuple[dict, float]:
    names = list(param_specs.keys())
    best_params = {name: param_specs[name][2] for name in names}
    best_f1 = evaluate_params(fn, rows, y, best_params)

    for _ in range(N_RANDOM_SEARCH):
        candidate = {name: rng.uniform(param_specs[name][0], param_specs[name][1]) for name in names}
        f1 = evaluate_params(fn, rows, y, candidate)
        if f1 > best_f1:
            best_f1, best_params = f1, candidate

    for _ in range(N_REFINE_ROUNDS):
        improved = False
        for name in names:
            low, high, _ = param_specs[name]
            span = (high - low) * 0.1
            for delta in (-span, span):
                candidate = dict(best_params)
                candidate[name] = min(high, max(low, candidate[name] + delta))
                f1 = evaluate_params(fn, rows, y, candidate)
                if f1 > best_f1:
                    best_f1, best_params = f1, candidate
                    improved = True
        if not improved:
            break

    return best_params, best_f1


def leave_one_match_out_for_skeleton(df: pd.DataFrame, param_specs: dict, fn, skeleton_name: str) -> list[dict]:
    matches = df["match_id"].unique()
    results = []
    for match_id in matches:
        train_df = df[df.match_id != match_id]
        test_df = df[df.match_id == match_id]

        train_rows = to_records(train_df)
        y_train = train_df["label"].values
        test_rows = to_records(test_df)
        y_test = test_df["label"].values

        rng = random.Random(RNG_SEED)
        best_params, train_f1 = optimize_params(fn, train_rows, y_train, param_specs, rng)

        error_counter = [0]
        test_preds = [safe_predict(fn, row, best_params, error_counter) for row in test_rows]

        results.append(
            {
                "skeleton": skeleton_name,
                "held_out_match": match_id,
                "train_f1": train_f1,
                "f1": f1_score(y_test, test_preds, zero_division=0),
                "precision": precision_score(y_test, test_preds, zero_division=0),
                "recall": recall_score(y_test, test_preds, zero_division=0),
                "n_errors": error_counter[0],
                "best_params": best_params,
            }
        )
    return results


def main() -> None:
    df = load_dataset()
    all_results = []
    for path in sorted(glob.glob(SKELETON_GLOB)):
        skeleton_name = path.split("/")[-1].replace(".py", "")
        print(f"=== {skeleton_name} ===")
        param_specs, fn = load_skeleton(path)
        results = leave_one_match_out_for_skeleton(df, param_specs, fn, skeleton_name)
        for r in results:
            print(f"  held_out={r['held_out_match']} f1={r['f1']:.3f} (train_f1={r['train_f1']:.3f}, errors={r['n_errors']})")
        all_results.extend(results)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv("documents/llm_skeleton_results.csv", index=False)

    print()
    summary = results_df.groupby("skeleton")[["f1", "precision", "recall"]].agg(["mean", "std"])
    print(summary)


if __name__ == "__main__":
    main()
