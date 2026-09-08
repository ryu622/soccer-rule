"""Island型進化計算によるTAKE_ON判定ルール骨格の探索。

research_plan.md フェーズ3の拡張。generate_skeleton.py/fit_skeleton.py で試した
単発生成(v1: AND連鎖 F1=0.467, v2: 均等多数決 F1=0.329)がいずれも決定木(0.554)・
XGBoost(0.634)を超えられなかったことを受け、LLMをmutation/crossoverオペレータとする
island型GAで複数世代にわたって骨格を改良する。

データ分割(単一train/evolve/test分割、7試合を固定シードでランダム割り当て):
    held_out_test: 進化ループに一切使わず、最終報告専用(analyze_evolution.pyでのみ使用)
    evolve_fitness: 個体の適応度(F1)を計算する検証用の1試合
    evolve_train: 個体ごとのパラメータ最適化(fit_skeleton.optimize_paramsと同じ手法)に使う4試合

個体表現は v1/v2 と同一(PARAM_SPECS辞書 + predict_take_on(f, p) 関数のコード文字列)。

uv run python scripts/evolve_skeleton.py [--islands N] [--island-size N] [--generations N]
    [--migration-interval N] [--patience N]
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime

from anthropic import Anthropic
from dotenv import load_dotenv
from sklearn.metrics import f1_score, precision_score, recall_score

from fit_skeleton import SAFE_BUILTINS, optimize_params, safe_predict, to_records
from generate_skeleton import FEATURE_DESCRIPTIONS, MODEL, TASK_DESCRIPTION, extract_code
from train_baselines import load_dataset

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = f"evolution_runs/{RUN_ID}"

# 固定シード(random.Random(42))によるランダム割り当て。再現手順は documents/phase3b_evolution_results.md 参照。
HELD_OUT_TEST_MATCHES = ["J03WMX", "J03WOH"]
EVOLVE_FITNESS_MATCH = "J03WPY"
EVOLVE_TRAIN_MATCHES = ["J03WN1", "J03WOY", "J03WQQ", "J03WR9"]

RNG_SEED_PARAM_FIT = 0  # fit_skeleton.py と同じ固定シード(パラメータ最適化用)
EVO_RNG_SEED = 42  # 選択・交叉相手選びなど進化操作全体で使う乱数

_id_counter = itertools.count(1)

# v1と同じ「診断情報ゼロ」のニュートラルな出力フォーマット指定(v2で追加された
# PRIOR_FINDINGS/多数決推奨は含めない。island A/B/Cはこの骨組みを共有し、タスク説明部分のみ変える)。
BASE_OUTPUT_FORMAT = """\
出力は必ず以下の2つを含むPythonコードブロック1個のみとしてください。説明文やコメント以外の
自然言語は一切書かないでください。

1. `PARAM_SPECS`という名前の辞書。キーはパラメータ名(str)、値は `(low, high, initial)` の
   3要素タプル(float)。パラメータは3〜7個程度にしてください。パラメータ名は戦術的な意味が
   わかる名前にしてください(例: "tight_space_distance_m"、"support_min_count" など)。
   数値の初期値・範囲は、上記の特徴量の説明にある単位・常識的なサッカーのスケール感を踏まえて
   設定してください(例: 距離ならおおよそ0〜10m程度、角度ならおおよそ0〜180度、人数なら0〜5程度)。

2. `predict_take_on(f: dict, p: dict) -> int` という名前の関数。
   - `f` は上記9つの特徴量を含む辞書(キー名は上記の通り)。
   - `p` は `PARAM_SPECS` のキーと同じキーを持つ辞書で、最適化後の実際のパラメータ値が入ります。
   - 戻り値は 1(仕掛ける)または 0(仕掛けない)。
   - 関数の内部では、まず戦術概念に対応する名前つきの中間変数(bool)をいくつか定義し
     (例: `is_tight_space = f["distance_m"] < p["tight_space_distance_m"]`)、
     それらを組み合わせて最終判定を行う、という書き方にしてください。
   - `p`にないパラメータ名を使わないこと。`f`にないキーを参照しないこと。
   - `import`や外部ライブラリは使わず、標準のPython演算子・比較のみで書いてください。

出力例のフォーマット(内容はダミーです):

```python
PARAM_SPECS = {
    "example_distance_m": (1.0, 8.0, 3.0),
}

def predict_take_on(f: dict, p: dict) -> int:
    is_example = f["distance_m"] < p["example_distance_m"]
    return 1 if is_example else 0
```
"""

ISLAND_TASK_B = (
    TASK_DESCRIPTION
    + """
仕掛ける理由は状況によって複数あり得ます。異なる状況に対応する複数の判定パターンを
自由に考え、いずれか1つでも満たせば仕掛けると判定する形で書いてください。
"""
)

ISLAND_TASK_C = (
    TASK_DESCRIPTION
    + """
まず大まかな状況を1つの基準で分け、その中でさらに詳しく判断する、という段階を踏んだ
考え方をしてみてください。
"""
)

ISLAND_PROMPTS = {
    "A": f"{FEATURE_DESCRIPTIONS}\n{TASK_DESCRIPTION}\n{BASE_OUTPUT_FORMAT}",
    "B": f"{FEATURE_DESCRIPTIONS}\n{ISLAND_TASK_B}\n{BASE_OUTPUT_FORMAT}",
    "C": f"{FEATURE_DESCRIPTIONS}\n{ISLAND_TASK_C}\n{BASE_OUTPUT_FORMAT}",
}

MUTATION_TEMPLATE = """\
以下はTAKE_ON判定ルールのコードです。検証用の1試合での評価スコアは
F1={f1:.3f}, Precision={precision:.3f}, Recall={recall:.3f} でした。

```python
{code}
```

このコードを改善した新しいバージョンを書いてください。

{feature_descriptions}

{output_format}
"""

CROSSOVER_TEMPLATE = """\
以下は2つの独立に開発されたTAKE_ON判定ルールです。

個体A(F1={f1_a:.3f}, Precision={precision_a:.3f}, Recall={recall_a:.3f}):
```python
{code_a}
```

個体B(F1={f1_b:.3f}, Precision={precision_b:.3f}, Recall={recall_b:.3f}):
```python
{code_b}
```

両方の良い点を組み合わせた新しいバージョンを書いてください。

{feature_descriptions}

{output_format}
"""


@dataclass
class Individual:
    id: str
    generation: int
    island: str
    parent_ids: list[str]
    method: str  # init_A / init_B / init_C / mutation / crossover
    code: str
    param_specs: dict | None
    fitted_params: dict | None
    train_f1: float
    fitness_f1: float
    precision: float
    recall: float
    error: str | None = None


def next_id() -> str:
    return f"ind{next(_id_counter)}"


def call_llm(client: Anthropic, prompt: str) -> str:
    response = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        thinking={"type": "disabled"},
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(block.text for block in response.content if block.type == "text")
    return extract_code(text)


def evaluate_individual(
    code: str,
    generation: int,
    island: str,
    parent_ids: list[str],
    method: str,
    evolve_train_rows: list[dict],
    y_train,
    evolve_fitness_rows: list[dict],
    y_fitness,
) -> Individual:
    ind_id = next_id()
    namespace: dict = {"__builtins__": SAFE_BUILTINS}
    try:
        exec(code, namespace)
        if "PARAM_SPECS" not in namespace or "predict_take_on" not in namespace:
            raise ValueError("PARAM_SPECS または predict_take_on が定義されていません")
        param_specs = namespace["PARAM_SPECS"]
        fn = namespace["predict_take_on"]
    except Exception as e:
        return Individual(
            id=ind_id, generation=generation, island=island, parent_ids=parent_ids, method=method,
            code=code, param_specs=None, fitted_params=None,
            train_f1=0.0, fitness_f1=0.0, precision=0.0, recall=0.0, error=str(e),
        )

    try:
        rng = random.Random(RNG_SEED_PARAM_FIT)
        fitted_params, train_f1 = optimize_params(fn, evolve_train_rows, y_train, param_specs, rng)
        error_counter = [0]
        preds = [safe_predict(fn, row, fitted_params, error_counter) for row in evolve_fitness_rows]
        fitness_f1 = f1_score(y_fitness, preds, zero_division=0)
        precision = precision_score(y_fitness, preds, zero_division=0)
        recall = recall_score(y_fitness, preds, zero_division=0)
        error = f"{error_counter[0]} runtime errors during prediction" if error_counter[0] else None
    except Exception as e:
        return Individual(
            id=ind_id, generation=generation, island=island, parent_ids=parent_ids, method=method,
            code=code, param_specs=param_specs, fitted_params=None,
            train_f1=0.0, fitness_f1=0.0, precision=0.0, recall=0.0, error=str(e),
        )

    return Individual(
        id=ind_id, generation=generation, island=island, parent_ids=parent_ids, method=method,
        code=code, param_specs=param_specs, fitted_params=fitted_params,
        train_f1=train_f1, fitness_f1=fitness_f1, precision=precision, recall=recall, error=error,
    )


def tournament_select(population: list[Individual], rng: random.Random, k: int = 3, exclude: set[str] = frozenset()) -> Individual:
    candidates = [ind for ind in population if ind.id not in exclude]
    pool = rng.sample(candidates, min(k, len(candidates)))
    return max(pool, key=lambda ind: ind.fitness_f1)


def mutate(client: Anthropic, parent: Individual) -> str:
    prompt = MUTATION_TEMPLATE.format(
        f1=parent.fitness_f1, precision=parent.precision, recall=parent.recall,
        code=parent.code, feature_descriptions=FEATURE_DESCRIPTIONS, output_format=BASE_OUTPUT_FORMAT,
    )
    return call_llm(client, prompt)


def crossover(client: Anthropic, parent_a: Individual, parent_b: Individual) -> str:
    prompt = CROSSOVER_TEMPLATE.format(
        f1_a=parent_a.fitness_f1, precision_a=parent_a.precision, recall_a=parent_a.recall, code_a=parent_a.code,
        f1_b=parent_b.fitness_f1, precision_b=parent_b.precision, recall_b=parent_b.recall, code_b=parent_b.code,
        feature_descriptions=FEATURE_DESCRIPTIONS, output_format=BASE_OUTPUT_FORMAT,
    )
    return call_llm(client, prompt)


def log_generation_snapshot(log_path: str, generation: int, islands: dict[str, list[Individual]]) -> None:
    with open(log_path, "a") as f:
        for island_key, population in islands.items():
            for ind in population:
                row = asdict(ind)
                row["generation"] = generation  # スナップショット時点の世代番号
                row["origin_island"] = ind.island
                row["current_island"] = island_key
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def migrate(islands: dict[str, list[Individual]]) -> None:
    ring = ["A", "B", "C"]
    bests = {key: max(islands[key], key=lambda ind: ind.fitness_f1) for key in ring}
    for i, key in enumerate(ring):
        next_key = ring[(i + 1) % len(ring)]
        worst_idx = min(range(len(islands[next_key])), key=lambda j: islands[next_key][j].fitness_f1)
        islands[next_key][worst_idx] = bests[key]


def run_evolution(n_islands: int, island_size: int, max_generations: int, migration_interval: int, patience: int) -> None:
    load_dotenv()
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    os.makedirs(OUT_DIR, exist_ok=True)
    log_path = f"{OUT_DIR}/population.jsonl"

    df = load_dataset()
    evolve_train_df = df[df.match_id.isin(EVOLVE_TRAIN_MATCHES)]
    evolve_fitness_df = df[df.match_id == EVOLVE_FITNESS_MATCH]
    evolve_train_rows = to_records(evolve_train_df)
    y_train = evolve_train_df["label"].values
    evolve_fitness_rows = to_records(evolve_fitness_df)
    y_fitness = evolve_fitness_df["label"].values

    with open(f"{OUT_DIR}/run_config.json", "w") as f:
        json.dump(
            {
                "run_id": RUN_ID, "model": MODEL, "n_islands": n_islands, "island_size": island_size,
                "max_generations": max_generations, "migration_interval": migration_interval, "patience": patience,
                "held_out_test_matches": HELD_OUT_TEST_MATCHES, "evolve_fitness_match": EVOLVE_FITNESS_MATCH,
                "evolve_train_matches": EVOLVE_TRAIN_MATCHES, "evo_rng_seed": EVO_RNG_SEED,
                "rng_seed_param_fit": RNG_SEED_PARAM_FIT,
            },
            f, ensure_ascii=False, indent=2,
        )

    evo_rng = random.Random(EVO_RNG_SEED)
    island_keys = ["A", "B", "C"][:n_islands]

    islands: dict[str, list[Individual]] = {}
    for island_key in island_keys:
        print(f"=== initial population: island {island_key} ===")
        population = []
        for _ in range(island_size):
            code = call_llm(client, ISLAND_PROMPTS[island_key])
            ind = evaluate_individual(
                code, 0, island_key, [], f"init_{island_key}",
                evolve_train_rows, y_train, evolve_fitness_rows, y_fitness,
            )
            print(f"  {ind.id} fitness_f1={ind.fitness_f1:.3f} error={ind.error}")
            population.append(ind)
        islands[island_key] = population

    log_generation_snapshot(log_path, 0, islands)

    best_ever_f1 = max(ind.fitness_f1 for pop in islands.values() for ind in pop)
    best_ever_individual = max((ind for pop in islands.values() for ind in pop), key=lambda ind: ind.fitness_f1)
    last_improved_gen = 0

    for generation in range(1, max_generations + 1):
        print(f"=== generation {generation} ===")
        new_islands: dict[str, list[Individual]] = {}
        for island_key in island_keys:
            population = islands[island_key]
            sorted_pop = sorted(population, key=lambda ind: ind.fitness_f1, reverse=True)
            elites = sorted_pop[:2]

            parent_m = tournament_select(population, evo_rng)
            code_m = mutate(client, parent_m)
            child_m = evaluate_individual(
                code_m, generation, island_key, [parent_m.id], "mutation",
                evolve_train_rows, y_train, evolve_fitness_rows, y_fitness,
            )

            parent_c1 = tournament_select(population, evo_rng)
            parent_c2 = tournament_select(population, evo_rng, exclude={parent_c1.id})
            code_c = crossover(client, parent_c1, parent_c2)
            child_c = evaluate_individual(
                code_c, generation, island_key, [parent_c1.id, parent_c2.id], "crossover",
                evolve_train_rows, y_train, evolve_fitness_rows, y_fitness,
            )

            print(f"  island {island_key}: mutation {child_m.id} f1={child_m.fitness_f1:.3f}, crossover {child_c.id} f1={child_c.fitness_f1:.3f}")
            new_islands[island_key] = elites + [child_m, child_c]

        islands = new_islands

        if generation % migration_interval == 0 and n_islands == 3:
            migrate(islands)
            print("  migration applied")

        log_generation_snapshot(log_path, generation, islands)

        gen_best = max((ind for pop in islands.values() for ind in pop), key=lambda ind: ind.fitness_f1)
        if gen_best.fitness_f1 > best_ever_f1:
            best_ever_f1 = gen_best.fitness_f1
            best_ever_individual = gen_best
            last_improved_gen = generation
            print(f"  new best: {gen_best.id} fitness_f1={best_ever_f1:.3f}")

        if generation - last_improved_gen >= patience:
            print(f"  early stop: no improvement in last {patience} generations")
            break

    with open(f"{OUT_DIR}/best_individual.py", "w") as f:
        f.write(f"# id={best_ever_individual.id} island={best_ever_individual.island} fitness_f1={best_ever_individual.fitness_f1:.3f}\n")
        f.write(f"# fitted_params={best_ever_individual.fitted_params}\n")
        f.write(best_ever_individual.code + "\n")

    print()
    print(f"best individual: {best_ever_individual.id} (island {best_ever_individual.island}) fitness_f1={best_ever_f1:.3f}")
    print(f"outputs saved to {OUT_DIR}/")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--islands", type=int, default=3)
    parser.add_argument("--island-size", type=int, default=4)
    parser.add_argument("--generations", type=int, default=15)
    parser.add_argument("--migration-interval", type=int, default=5)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()

    run_evolution(
        n_islands=args.islands, island_size=args.island_size, max_generations=args.generations,
        migration_interval=args.migration_interval, patience=args.patience,
    )


if __name__ == "__main__":
    main()
