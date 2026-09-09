"""既存の3つのコード資産を、新しい閾値フィッティング(分位点グリッド+座標降下法の全探索)で
再フィッティングし、held-out F1と端張り付きの度合いがどう変わるかを確認する。
追加のLLM呼び出しは不要。

対象:
    A. 16条件ルール(evolve_skeleton_v2の2回目の実行、11特徴量、held-out F1=0.370で
       distance_m単独と99.6%一致するまで退化していたことが判明したもの)
    B. ind90(evolve_skeletonのv1、9特徴量、held-out F1=0.491)
    C. v1最良個体(seed0、AND連鎖、9特徴量、7-fold平均F1=0.467)

いずれも同じデータ分割(evolve_set 5試合で再フィット→held_out_test 2試合で評価)で比較する。

uv run python scripts/compare_refit_methods.py
"""

from __future__ import annotations

import random

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from evolve_skeleton_v2 import (
    DATA_PATH,
    EVOLVE_SET_MATCHES,
    HELD_OUT_TEST_MATCHES,
    RNG_SEED_PARAM_FIT,
    build_candidate_grid,
    infer_param_feature_mapping,
    optimize_params,
    to_records,
)
from fit_skeleton import SAFE_BUILTINS, safe_predict

TARGETS = {
    "A_16cond_v2": "evolution_runs_v2/20260908_230936/best_individual.py",
    "B_ind90_v1": "evolution_runs/20260908_205621/best_individual.py",
    "C_v1seed0_AND": "llm_skeletons/v1_and_chain/skeleton_seed0.py",
}


def strip_comments(code: str) -> str:
    return "\n".join(line for line in code.split("\n") if not line.startswith("#"))


def refit_and_evaluate(name: str, path: str, refit_df: pd.DataFrame, held_out_df: pd.DataFrame) -> dict:
    code = strip_comments(open(path).read())
    namespace: dict = {"__builtins__": SAFE_BUILTINS}
    exec(code, namespace)
    fn = namespace["predict_take_on"]
    param_specs = namespace["PARAM_SPECS"]

    param_feature_map = infer_param_feature_mapping(code)
    unmapped = [p for p in param_specs if p not in param_feature_map]

    refit_rows = to_records(refit_df)
    y_refit = list(refit_df["label"].values)
    candidates = build_candidate_grid(param_specs, param_feature_map, refit_rows)

    rng = random.Random(RNG_SEED_PARAM_FIT)
    refit_params, refit_f1 = optimize_params(fn, refit_rows, y_refit, param_specs, rng, candidates)

    # 端張り付き監査: フィット値が候補グリッドの最小/最大(=実データの0/100パーセンタイル)と一致するか
    pinned = []
    for p, val in refit_params.items():
        c = candidates[p]
        if abs(val - min(c)) < 1e-9 or abs(val - max(c)) < 1e-9:
            pinned.append(p)

    all_preds, all_y = [], []
    for match_id in HELD_OUT_TEST_MATCHES:
        match_df = held_out_df[held_out_df.match_id == match_id]
        rows = to_records(match_df)
        y = list(match_df["label"].values)
        preds = [safe_predict(fn, row, refit_params) for row in rows]
        all_preds.extend(preds)
        all_y.extend(y)

    return {
        "name": name,
        "n_params": len(param_specs),
        "n_pinned": len(pinned),
        "pinned_params": pinned,
        "unmapped_params": unmapped,
        "refit_f1_on_evolve_set": refit_f1,
        "held_out_f1": f1_score(all_y, all_preds, zero_division=0),
        "held_out_precision": precision_score(all_y, all_preds, zero_division=0),
        "held_out_recall": recall_score(all_y, all_preds, zero_division=0),
        "refit_params": refit_params,
    }


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    df = df[~df["sync_error_suspect"]].reset_index(drop=True)
    refit_df = df[df.match_id.isin(EVOLVE_SET_MATCHES)]
    held_out_df = df[df.match_id.isin(HELD_OUT_TEST_MATCHES)]

    print(f"{'name':<16} {'n_params':>8} {'n_pinned':>8} {'refit_F1':>9} {'held-out F1':>12} {'precision':>10} {'recall':>8}")
    results = []
    for name, path in TARGETS.items():
        r = refit_and_evaluate(name, path, refit_df, held_out_df)
        results.append(r)
        print(f"{r['name']:<16} {r['n_params']:>8} {r['n_pinned']:>8} {r['refit_f1_on_evolve_set']:>9.3f} {r['held_out_f1']:>12.3f} {r['held_out_precision']:>10.3f} {r['held_out_recall']:>8.3f}")
        if r["pinned_params"]:
            print(f"    pinned: {r['pinned_params']}")
        if r["unmapped_params"]:
            print(f"    unmapped(フォールバック使用): {r['unmapped_params']}")

    print()
    print("参考値: 決定木(depth=3, 7-fold平均) F1 = 0.554")
    print("参考値: 旧フィッティング(連続空間ランダムサーチ)でのA(16条件v2) held-out F1 = 0.370")
    print("参考値: 旧フィッティングでのB(ind90) held-out F1 = 0.491")


if __name__ == "__main__":
    main()
