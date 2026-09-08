"""進化計算(evolve_skeleton.py)の結果を可視化し、held-out試合で最終評価する。

uv run python scripts/analyze_evolution.py [--run-id RUN_ID]
run-id省略時は evolution_runs/ 配下の最新ディレクトリを使う。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Hiragino Sans"
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from evolve_skeleton import HELD_OUT_TEST_MATCHES, RNG_SEED_PARAM_FIT
from fit_skeleton import SAFE_BUILTINS, optimize_params, safe_predict, to_records
from train_baselines import load_dataset

FIG_DIR = "documents/figures"


def latest_run_dir() -> str:
    dirs = sorted(glob.glob("evolution_runs/*"))
    if not dirs:
        raise FileNotFoundError("evolution_runs/ にディレクトリが見つかりません")
    return dirs[-1]


def load_population(run_dir: str) -> pd.DataFrame:
    rows = []
    with open(f"{run_dir}/population.jsonl") as f:
        for line in f:
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def plot_progress(df: pd.DataFrame, run_config: dict, out_path: str) -> None:
    progress = df.groupby(["generation", "current_island"])["fitness_f1"].max().reset_index()

    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = {"A": "tab:blue", "B": "tab:orange", "C": "tab:green"}
    for island, sub in progress.groupby("current_island"):
        sub = sub.sort_values("generation")
        ax.plot(sub["generation"], sub["fitness_f1"], marker="o", label=f"island {island}", color=colors.get(island))

    migration_interval = run_config.get("migration_interval", 5)
    max_gen = int(progress["generation"].max())
    for gen in range(migration_interval, max_gen + 1, migration_interval):
        ax.axvline(gen, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    ax.set_xlabel("generation")
    ax.set_ylabel("fitness F1 (evolve_fitness match, best in island)")
    ax.set_title("進化計算の世代推移(破線=移住が発生した世代)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def evaluate_on_held_out(code: str, param_specs: dict) -> dict:
    df = load_dataset()
    run_config_matches = json.load(open(f"{latest_run_dir()}/run_config.json"))
    evolve_train_matches = run_config_matches["evolve_train_matches"]
    evolve_fitness_match = run_config_matches["evolve_fitness_match"]
    refit_matches = evolve_train_matches + [evolve_fitness_match]

    refit_df = df[df.match_id.isin(refit_matches)]
    held_out_df = df[df.match_id.isin(HELD_OUT_TEST_MATCHES)]

    namespace: dict = {"__builtins__": SAFE_BUILTINS}
    exec(code, namespace)
    fn = namespace["predict_take_on"]

    rng = random.Random(RNG_SEED_PARAM_FIT)
    refit_rows = to_records(refit_df)
    y_refit = refit_df["label"].values
    refit_params, refit_f1 = optimize_params(fn, refit_rows, y_refit, param_specs, rng)

    results = {"refit_params": refit_params, "refit_f1_on_evolve_set": refit_f1, "per_match": {}}

    all_preds, all_y = [], []
    for match_id in HELD_OUT_TEST_MATCHES:
        match_df = held_out_df[held_out_df.match_id == match_id]
        rows = to_records(match_df)
        y = match_df["label"].values
        preds = [safe_predict(fn, row, refit_params) for row in rows]
        all_preds.extend(preds)
        all_y.extend(y)
        results["per_match"][match_id] = {
            "f1": f1_score(y, preds, zero_division=0),
            "precision": precision_score(y, preds, zero_division=0),
            "recall": recall_score(y, preds, zero_division=0),
        }

    results["pooled"] = {
        "f1": f1_score(all_y, all_preds, zero_division=0),
        "precision": precision_score(all_y, all_preds, zero_division=0),
        "recall": recall_score(all_y, all_preds, zero_division=0),
    }
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    run_dir = f"evolution_runs/{args.run_id}" if args.run_id else latest_run_dir()
    print(f"analyzing {run_dir}")

    run_config = json.load(open(f"{run_dir}/run_config.json"))
    df = load_population(run_dir)

    os.makedirs(FIG_DIR, exist_ok=True)
    plot_progress(df, run_config, f"{FIG_DIR}/evolution_progress.png")

    best_row = df.loc[df["fitness_f1"].idxmax()]
    print(f"best individual: {best_row['id']} island(origin)={best_row['origin_island']} method={best_row['method']} fitness_f1={best_row['fitness_f1']:.3f}")

    held_out_results = evaluate_on_held_out(best_row["code"], best_row["param_specs"])
    print("held-out evaluation:")
    print(json.dumps(held_out_results, ensure_ascii=False, indent=2))

    with open(f"{run_dir}/held_out_evaluation.json", "w") as f:
        json.dump(held_out_results, f, ensure_ascii=False, indent=2)

    lineage = df[df["origin_island"] == best_row["origin_island"]]
    print(f"\nislands present at generation 0: {sorted(df[df.generation==0]['origin_island'].unique())}")
    print(f"total individuals evaluated: {df['id'].nunique()}")
    print(f"errors encountered: {df[df['error'].notna()]['id'].nunique()} individuals")


if __name__ == "__main__":
    main()
