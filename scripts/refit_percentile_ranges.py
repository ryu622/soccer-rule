"""ind90の骨格(6パターンのコード)はそのまま、PARAM_SPECSの探索範囲だけを
evolve_train(4試合)のみのデータ分位点(P5〜P95)から再計算し、fit_skeleton.optimize_paramsを
回し直す。追加のLLM呼び出しは無し。

documents/phase3b_evolution_results.md 3.4.3節「端張り付き監査」で、7パラメータ中2個
(space_after_beat_distance_m, sideline_danger_distance_m)が探索範囲の端に張り付き実質的に
装飾化していたことが判明した。LLMが手探りで決めた探索範囲([0.5,5.0]等)がそもそも
「効きやすい値」を含んでいなかった可能性があるため、実データの分布から範囲を再定義して
同じ骨格構造でどこまで性能が伸びるかを確認する。

リーク回避: 分位点はevolve_train(4試合)のみから計算する。evolve_fitness/held_out_testの
データを混ぜて範囲を決めると「テストデータの分布を覗いてから探索範囲を決めた」形のリークになる
(このプロジェクトで一貫して避けてきたリークの一種)。

uv run python scripts/refit_percentile_ranges.py
"""

from __future__ import annotations

import glob
import json
import random

from sklearn.metrics import f1_score, precision_score, recall_score

from evolve_skeleton import EVOLVE_FITNESS_MATCH, EVOLVE_TRAIN_MATCHES, HELD_OUT_TEST_MATCHES
from fit_skeleton import SAFE_BUILTINS, optimize_params, safe_predict, to_records
from train_baselines import load_dataset

RNG_SEED_PARAM_FIT = 0
PERCENTILE_LOW = 0.05
PERCENTILE_HIGH = 0.95

# ind90のPARAM_SPECSキー -> 対応する生特徴量(predict_take_on内の比較対象)
FEATURE_MAP = {
    "tight_space_distance_m": "distance_m",
    "head_on_approach_angle_deg": "approach_angle_deg",
    "high_closing_speed_mps": "closing_speed_mps",
    "space_after_beat_distance_m": "second_nearest_dist_m",
    "support_min_count": "n_supporting_teammates",
    "sideline_danger_distance_m": "dist_to_sideline_m",
    "attacking_third_distance_m": "dist_to_goal_line_m",
}


def latest_run_dir() -> str:
    dirs = sorted(glob.glob("evolution_runs/*"))
    return dirs[-1]


def load_ind90_code(run_dir: str) -> str:
    with open(f"{run_dir}/population.jsonl") as f:
        for line in f:
            row = json.loads(line)
            if row["id"] == "ind90":
                return row["code"]
    raise ValueError("ind90 not found")


def build_percentile_param_specs(evolve_train_df) -> dict:
    param_specs = {}
    for param, feature in FEATURE_MAP.items():
        low = evolve_train_df[feature].quantile(PERCENTILE_LOW)
        high = evolve_train_df[feature].quantile(PERCENTILE_HIGH)
        initial = evolve_train_df[feature].quantile(0.5)
        param_specs[param] = (float(low), float(high), float(initial))
    return param_specs


def main() -> None:
    run_dir = latest_run_dir()
    code = load_ind90_code(run_dir)

    namespace: dict = {"__builtins__": SAFE_BUILTINS}
    exec(code, namespace)
    fn = namespace["predict_take_on"]
    original_param_specs = namespace["PARAM_SPECS"]

    df = load_dataset()
    evolve_train_df = df[df.match_id.isin(EVOLVE_TRAIN_MATCHES)]
    evolve_fitness_df = df[df.match_id == EVOLVE_FITNESS_MATCH]
    held_out_df = df[df.match_id.isin(HELD_OUT_TEST_MATCHES)]
    refit_df = df[df.match_id.isin(EVOLVE_TRAIN_MATCHES + [EVOLVE_FITNESS_MATCH])]

    new_param_specs = build_percentile_param_specs(evolve_train_df)

    print("=== 探索範囲の比較 (元のLLM指定 vs evolve_train分位点P5-P95) ===")
    for name in original_param_specs:
        old = original_param_specs[name]
        new = new_param_specs[name]
        print(f"{name:<28} old=({old[0]:.2f}, {old[1]:.2f}) -> new=({new[0]:.2f}, {new[1]:.2f})")
    print()

    # --- Step 1: evolve_train でフィット -> evolve_fitness で「一次シグナル」を確認 ---
    rng = random.Random(RNG_SEED_PARAM_FIT)
    train_rows = to_records(evolve_train_df)
    y_train = evolve_train_df["label"].values
    fitted_params, train_f1 = optimize_params(fn, train_rows, y_train, new_param_specs, rng)

    fitness_rows = to_records(evolve_fitness_df)
    y_fitness = evolve_fitness_df["label"].values
    fitness_preds = [safe_predict(fn, row, fitted_params) for row in fitness_rows]
    fitness_f1 = f1_score(y_fitness, fitness_preds, zero_division=0)
    fitness_precision = precision_score(y_fitness, fitness_preds, zero_division=0)
    fitness_recall = recall_score(y_fitness, fitness_preds, zero_division=0)

    print("=== Step1: evolve_trainでフィット -> evolve_fitness(1試合)で評価(一次シグナル) ===")
    print(f"fitted_params: {fitted_params}")
    print(f"train_f1={train_f1:.3f}  fitness_f1={fitness_f1:.3f} precision={fitness_precision:.3f} recall={fitness_recall:.3f}")
    print(f"(参考: 元の探索範囲でのind90 fitness_f1=0.598)")
    print()

    # --- Step 2: evolve_train+evolve_fitness(5試合)で再フィット -> held_out_test(2試合)で最終評価 ---
    rng2 = random.Random(RNG_SEED_PARAM_FIT)
    refit_rows = to_records(refit_df)
    y_refit = refit_df["label"].values
    refit_params, refit_f1 = optimize_params(fn, refit_rows, y_refit, new_param_specs, rng2)

    all_preds, all_y = [], []
    per_match = {}
    for match_id in HELD_OUT_TEST_MATCHES:
        match_df = held_out_df[held_out_df.match_id == match_id]
        rows = to_records(match_df)
        y = match_df["label"].values
        preds = [safe_predict(fn, row, refit_params) for row in rows]
        all_preds.extend(preds)
        all_y.extend(y)
        per_match[match_id] = {
            "f1": f1_score(y, preds, zero_division=0),
            "precision": precision_score(y, preds, zero_division=0),
            "recall": recall_score(y, preds, zero_division=0),
        }

    pooled_f1 = f1_score(all_y, all_preds, zero_division=0)
    pooled_precision = precision_score(all_y, all_preds, zero_division=0)
    pooled_recall = recall_score(all_y, all_preds, zero_division=0)

    print("=== Step2: evolve_train+evolve_fitness(5試合)で再フィット -> held_out_test(2試合)で最終評価 ===")
    print(f"refit_params: {refit_params}")
    print(f"refit_f1_on_5matches={refit_f1:.3f}")
    print(f"held-out pooled: f1={pooled_f1:.3f} precision={pooled_precision:.3f} recall={pooled_recall:.3f}")
    for match_id, m in per_match.items():
        print(f"  {match_id}: f1={m['f1']:.3f} precision={m['precision']:.3f} recall={m['recall']:.3f}")
    print()
    print("=== 参考値 ===")
    print("元のPARAM_SPECS(LLM指定範囲)でのind90 held-out F1 = 0.491")
    print("決定木(depth=3, 7-fold平均) F1 = 0.554")
    print("ロジスティック回帰(7-fold平均) F1 = 0.543")

    result = {
        "new_param_specs": new_param_specs,
        "step1_evolve_train_fit": {"fitted_params": fitted_params, "train_f1": train_f1, "fitness_f1": fitness_f1, "precision": fitness_precision, "recall": fitness_recall},
        "step2_refit_held_out": {"refit_params": refit_params, "refit_f1_on_5matches": refit_f1, "pooled": {"f1": pooled_f1, "precision": pooled_precision, "recall": pooled_recall}, "per_match": per_match},
    }
    with open(f"{run_dir}/percentile_range_refit.json", "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nsaved to {run_dir}/percentile_range_refit.json")


if __name__ == "__main__":
    main()
