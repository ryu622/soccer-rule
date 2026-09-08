"""Island型進化計算 v2: 拡張特徴量(11個)・適応度の複数試合平均化・複雑さペナルティ・
探索予算増加・ロギング修正を組み込んだ再実行。

documents/phase3b_evolution_results.md・documents/layer_diagnosis_expanded_features.md の
議論を受けた再設計:

1. 特徴量: 元9 + n_opponents_nearby_10m(密集度) + prev_event_was_pass(直前イベント、bool)= 11
   (XGBoostでの重要度ジャンプが特に大きかった2個のみを追加。残り5個は見送り)
2. 探索範囲: LLMが提案したPARAM_SPECSをそのまま使う(分位点較正はしない。
   documents/phase3b_evolution_results.md 3.4.4節で「較正しても改善しない」ことが判明済み)
3. 適応度: evolve側5試合(J03WPY, J03WN1, J03WOY, J03WQQ, J03WR9)の内側leave-one-out平均F1
   (単一試合への過適合を緩和。held_out_test 2試合は一切触れない)
4. 複雑さペナルティ: fitness = mean_F1 - 0.002 × (and/orで結合されたブール条件の総数、AST解析で機械的にカウント)
   選択(エリート選定・トーナメント・移住)はこの penalized fitness で行う。
   プロンプトにはペナルティの存在を一切明かさない(F1/Precision/Recallの数値のみ見せる、
   従来通り「診断コメント一切なし」の方針を維持)。
5. 探索予算: ランダムサーチ400→700反復(特徴量・パラメータ増加に伴う探索空間拡大への対応)
6. ロギング: log_generation_snapshotをmigrateより前に呼ぶ(移住で即座に上書きされる個体も
   必ず1回はログに残る、documents/phase3b_evolution_results.md 4節の既知の欠落の修正)

--- 2回目の実行(RUN_ID=20260908_223306)がv1(単一試合適応度)より悪化(held-out F1 0.372 vs 0.491)
した原因を診断した結果、island内の多様性が世代3〜4で崩壊していたことが判明した
(island size=4・エリート2体保存・トーナメント選択が4体中3体抽出という組み合わせでは、
エリートがほぼ確実に親として選ばれ続け、mutation/crossoverの入力が実質同じ個体の複製に
収束してしまう。island Bは4個体がバイトレベルで完全同一のコードに、island Cは6個体が
F1小数点以下6桁まで一致するコードに収束していた)。これを受けて以下3点を追加修正した
(呼び出し予算はほぼ変えない設計):

7. エリート保存数: 2→1(island size 4のまま)。世代0以降、各islandは実質「エリート1体+
   新規2体」の3体で回る(4体には戻さない)。保存枠を減らすことで集団の入れ替わりを速める。
8. crossoverを廃止し、「フレッシュな乱数個体」(既存個体を一切参照せず、初期生成と同じ
   island固有プロンプトからゼロ生成)に置き換えた。crossoverは既に均質化した集団内では
   同系統の個体同士を混ぜるだけで新しい遺伝子を持ち込めないため、独立な多様性源として
   fresh_randomを採用する。1世代あたりの新規個体は「mutation 1体 + fresh_random 1体」で
   従来(mutation 1体+crossover 1体)と呼び出し数は変わらない。
9. 移住間隔: 5世代→3世代。多様性崩壊が世代3〜4で起きていたため、5世代ごとでは手遅れになる。

uv run python scripts/evolve_skeleton_v2.py [--islands N] [--island-size N] [--generations N]
    [--migration-interval N] [--patience N]
"""

from __future__ import annotations

import argparse
import ast
import itertools
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime

import pandas as pd
from anthropic import Anthropic
from dotenv import load_dotenv
from sklearn.metrics import f1_score, precision_score, recall_score

from fit_skeleton import SAFE_BUILTINS, safe_predict
from generate_skeleton import FEATURE_DESCRIPTIONS as FEATURE_DESCRIPTIONS_V1
from generate_skeleton import MODEL, TASK_DESCRIPTION, extract_code

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = f"evolution_runs_v2/{RUN_ID}"

DATA_PATH = "data/tackling_features_v2.csv"
FEATURE_COLUMNS = [
    "distance_m", "approach_angle_deg", "carrier_speed_mps", "opponent_speed_mps",
    "closing_speed_mps", "second_nearest_dist_m", "n_supporting_teammates",
    "dist_to_sideline_m", "dist_to_goal_line_m",
    "n_opponents_nearby_10m", "prev_event_was_pass",
]

# 固定シード(random.Random(42))によるオリジナルの試合割り当て(evolve_skeleton.pyと同一)を継承。
HELD_OUT_TEST_MATCHES = ["J03WMX", "J03WOH"]
EVOLVE_SET_MATCHES = ["J03WN1", "J03WOY", "J03WQQ", "J03WR9", "J03WPY"]

RNG_SEED_PARAM_FIT = 0
EVO_RNG_SEED = 42
N_RANDOM_SEARCH = 700  # 400から増加(特徴量・パラメータ増加に伴う探索空間拡大への対応)
N_REFINE_ROUNDS = 5
LAMBDA_COMPLEXITY = 0.002

_id_counter = itertools.count(1)

FEATURE_DESCRIPTIONS_NEW = """\
- n_opponents_nearby_10m: ボール保持者から半径10m以内にいる相手選手の総数(密集度)。
  多いほど周囲に相手選手が密集しており、スペースが少ない状況を意味する。
- prev_event_was_pass: 直前に起きたプレー(誰のものでも)がパスだったかどうかを表す
  0または1の値(既に真偽値であり、閾値パラメータは不要)。1ならボール保持者は
  直前にパスを受けて(あるいは直前がパスの直後で)この状況に至ったことを意味する。
"""

FEATURE_DESCRIPTIONS = FEATURE_DESCRIPTIONS_V1.rstrip() + "\n" + FEATURE_DESCRIPTIONS_NEW

BASE_OUTPUT_FORMAT = """\
出力は必ず以下の2つを含むPythonコードブロック1個のみとしてください。説明文やコメント以外の
自然言語は一切書かないでください。

1. `PARAM_SPECS`という名前の辞書。キーはパラメータ名(str)、値は `(low, high, initial)` の
   3要素タプル(float)。パラメータは3〜7個程度にしてください。パラメータ名は戦術的な意味が
   わかる名前にしてください(例: "tight_space_distance_m"、"support_min_count" など)。
   数値の初期値・範囲は、上記の特徴量の説明にある単位・常識的なサッカーのスケール感を踏まえて
   設定してください(例: 距離ならおおよそ0〜10m程度、角度ならおおよそ0〜180度、人数なら0〜5程度)。
   **`prev_event_was_pass`のように既に0/1のbool値である特徴量については、閾値パラメータを
   PARAM_SPECSに追加する必要はありません。`predict_take_on`内で`f["prev_event_was_pass"]`を
   そのままbool条件として直接使ってください(例: `was_previous_pass = bool(f["prev_event_was_pass"])`)。**

2. `predict_take_on(f: dict, p: dict) -> int` という名前の関数。
   - `f` は上記11個の特徴量を含む辞書(キー名は上記の通り)。
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
    was_previous_pass = bool(f["prev_event_was_pass"])
    return 1 if (is_example and was_previous_pass) else 0
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
以下はTAKE_ON判定ルールのコードです。検証用データでの評価スコアは
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
    method: str
    code: str
    param_specs: dict | None
    fitted_params_per_fold: list | None  # 5-fold分のfitted_params(参考保存用)
    fitness_f1: float  # 5-fold平均F1(生の値、ペナルティ前)
    precision: float
    recall: float
    condition_count: int
    penalized_fitness: float  # 選択に使う値: fitness_f1 - LAMBDA_COMPLEXITY * condition_count
    error: str | None = None


def next_id() -> str:
    return f"ind{next(_id_counter)}"


def to_records(df: pd.DataFrame) -> list[dict]:
    return df[FEATURE_COLUMNS].to_dict("records")


def count_conditions(code: str) -> int:
    """and/orで結合されたブール条件の総数を機械的にカウントする(名目条件数ベースの複雑さ指標)。"""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 0
    total = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.BoolOp):
            total += len(node.values)
    return total


def call_llm(client: Anthropic, prompt: str, max_attempts: int = 4) -> str:
    last_error: Exception | None = None
    for attempt in range(max_attempts):
        try:
            response = client.messages.create(
                model=MODEL, max_tokens=2000, thinking={"type": "disabled"},
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(block.text for block in response.content if block.type == "text")
            return extract_code(text)
        except Exception as e:
            last_error = e
            wait = 5 * (attempt + 1)
            print(f"  call_llm failed (attempt {attempt + 1}/{max_attempts}): {e}. retrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError(f"call_llm failed after {max_attempts} attempts") from last_error


def optimize_params(fn, rows: list[dict], y, param_specs: dict, rng: random.Random) -> tuple[dict, float]:
    names = list(param_specs.keys())
    best_params = {name: param_specs[name][2] for name in names}
    best_f1 = f1_score(y, [safe_predict(fn, row, best_params) for row in rows], zero_division=0)

    for _ in range(N_RANDOM_SEARCH):
        candidate = {name: rng.uniform(param_specs[name][0], param_specs[name][1]) for name in names}
        f1 = f1_score(y, [safe_predict(fn, row, candidate) for row in rows], zero_division=0)
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
                f1 = f1_score(y, [safe_predict(fn, row, candidate) for row in rows], zero_division=0)
                if f1 > best_f1:
                    best_f1, best_params = f1, candidate
                    improved = True
        if not improved:
            break
    return best_params, best_f1


def evaluate_individual(
    code: str, generation: int, island: str, parent_ids: list[str], method: str,
    evolve_rows_by_match: dict[str, tuple[list[dict], list[int]]],
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
            code=code, param_specs=None, fitted_params_per_fold=None,
            fitness_f1=0.0, precision=0.0, recall=0.0, condition_count=0, penalized_fitness=0.0, error=str(e),
        )

    matches = list(evolve_rows_by_match.keys())
    fold_f1s, fold_precisions, fold_recalls, fitted_list = [], [], [], []
    try:
        for held_out in matches:
            train_rows, train_y = [], []
            for m in matches:
                if m == held_out:
                    continue
                rows, y = evolve_rows_by_match[m]
                train_rows.extend(rows)
                train_y.extend(y)

            rng = random.Random(RNG_SEED_PARAM_FIT)
            fitted_params, _train_f1 = optimize_params(fn, train_rows, train_y, param_specs, rng)
            fitted_list.append(fitted_params)

            test_rows, test_y = evolve_rows_by_match[held_out]
            error_counter = [0]
            preds = [safe_predict(fn, row, fitted_params, error_counter) for row in test_rows]
            fold_f1s.append(f1_score(test_y, preds, zero_division=0))
            fold_precisions.append(precision_score(test_y, preds, zero_division=0))
            fold_recalls.append(recall_score(test_y, preds, zero_division=0))
    except Exception as e:
        return Individual(
            id=ind_id, generation=generation, island=island, parent_ids=parent_ids, method=method,
            code=code, param_specs=param_specs, fitted_params_per_fold=None,
            fitness_f1=0.0, precision=0.0, recall=0.0, condition_count=0, penalized_fitness=0.0, error=str(e),
        )

    mean_f1 = sum(fold_f1s) / len(fold_f1s)
    mean_precision = sum(fold_precisions) / len(fold_precisions)
    mean_recall = sum(fold_recalls) / len(fold_recalls)
    condition_count = count_conditions(code)
    penalized_fitness = mean_f1 - LAMBDA_COMPLEXITY * condition_count

    return Individual(
        id=ind_id, generation=generation, island=island, parent_ids=parent_ids, method=method,
        code=code, param_specs=param_specs, fitted_params_per_fold=fitted_list,
        fitness_f1=mean_f1, precision=mean_precision, recall=mean_recall,
        condition_count=condition_count, penalized_fitness=penalized_fitness, error=None,
    )


def tournament_select(population: list[Individual], rng: random.Random, k: int = 3, exclude: set[str] = frozenset()) -> Individual:
    candidates = [ind for ind in population if ind.id not in exclude]
    pool = rng.sample(candidates, min(k, len(candidates)))
    return max(pool, key=lambda ind: ind.penalized_fitness)


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
                row["generation"] = generation
                row["origin_island"] = ind.island
                row["current_island"] = island_key
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def migrate(islands: dict[str, list[Individual]]) -> None:
    ring = ["A", "B", "C"]
    bests = {key: max(islands[key], key=lambda ind: ind.penalized_fitness) for key in ring}
    for i, key in enumerate(ring):
        next_key = ring[(i + 1) % len(ring)]
        worst_idx = min(range(len(islands[next_key])), key=lambda j: islands[next_key][j].penalized_fitness)
        islands[next_key][worst_idx] = bests[key]


def load_evolve_rows() -> dict[str, tuple[list[dict], list[int]]]:
    df = pd.read_csv(DATA_PATH)
    n_before = len(df)
    df = df[~df["sync_error_suspect"]].reset_index(drop=True)
    print(f"excluded {n_before - len(df)} sync_error_suspect rows -> {len(df)} rows remain")

    result = {}
    for match_id in EVOLVE_SET_MATCHES:
        match_df = df[df.match_id == match_id]
        result[match_id] = (to_records(match_df), list(match_df["label"].values))
    return result, df


def run_evolution(n_islands: int, island_size: int, max_generations: int, migration_interval: int, patience: int) -> None:
    load_dotenv()
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], max_retries=5)

    os.makedirs(OUT_DIR, exist_ok=True)
    log_path = f"{OUT_DIR}/population.jsonl"

    evolve_rows_by_match, _df = load_evolve_rows()

    with open(f"{OUT_DIR}/run_config.json", "w") as f:
        json.dump(
            {
                "run_id": RUN_ID, "model": MODEL, "n_islands": n_islands, "island_size": island_size,
                "max_generations": max_generations, "migration_interval": migration_interval, "patience": patience,
                "held_out_test_matches": HELD_OUT_TEST_MATCHES, "evolve_set_matches": EVOLVE_SET_MATCHES,
                "feature_columns": FEATURE_COLUMNS, "n_random_search": N_RANDOM_SEARCH,
                "lambda_complexity": LAMBDA_COMPLEXITY, "evo_rng_seed": EVO_RNG_SEED, "rng_seed_param_fit": RNG_SEED_PARAM_FIT,
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
            ind = evaluate_individual(code, 0, island_key, [], f"init_{island_key}", evolve_rows_by_match)
            print(f"  {ind.id} fitness_f1={ind.fitness_f1:.3f} conditions={ind.condition_count} penalized={ind.penalized_fitness:.3f} error={ind.error}")
            population.append(ind)
        islands[island_key] = population

    log_generation_snapshot(log_path, 0, islands)

    best_ever = max((ind for pop in islands.values() for ind in pop), key=lambda ind: ind.penalized_fitness)
    last_improved_gen = 0

    for generation in range(1, max_generations + 1):
        print(f"=== generation {generation} ===")
        new_islands: dict[str, list[Individual]] = {}
        for island_key in island_keys:
            population = islands[island_key]
            sorted_pop = sorted(population, key=lambda ind: ind.penalized_fitness, reverse=True)
            elites = sorted_pop[:1]  # 2->1: 多様性崩壊対策(3.4.4節の追加診断を参照)

            parent_m = tournament_select(population, evo_rng)
            code_m = mutate(client, parent_m)
            child_m = evaluate_individual(code_m, generation, island_key, [parent_m.id], "mutation", evolve_rows_by_match)

            # crossoverは既存個体同士を混ぜるだけで新しい遺伝子を持ち込めないため、
            # 独立な多様性源として「フレッシュな乱数個体」(初期生成と同じプロンプト、既存個体は一切参照しない)に置き換える。
            code_r = call_llm(client, ISLAND_PROMPTS[island_key])
            child_r = evaluate_individual(code_r, generation, island_key, [], "fresh_random", evolve_rows_by_match)

            print(f"  island {island_key}: mutation {child_m.id} f1={child_m.fitness_f1:.3f}(cond={child_m.condition_count}), fresh_random {child_r.id} f1={child_r.fitness_f1:.3f}(cond={child_r.condition_count})")
            new_islands[island_key] = elites + [child_m, child_r]

        islands = new_islands

        # ロギング修正: migrateより前にこの世代の状態を記録する(即座に上書きされる個体も1回は残す)
        log_generation_snapshot(log_path, generation, islands)

        if generation % migration_interval == 0 and n_islands == 3:
            migrate(islands)
            print("  migration applied")

        gen_best = max((ind for pop in islands.values() for ind in pop), key=lambda ind: ind.penalized_fitness)
        if gen_best.penalized_fitness > best_ever.penalized_fitness:
            best_ever = gen_best
            last_improved_gen = generation
            print(f"  new best: {gen_best.id} fitness_f1={gen_best.fitness_f1:.3f} penalized={gen_best.penalized_fitness:.3f}")

        if generation - last_improved_gen >= patience:
            print(f"  early stop: no improvement in last {patience} generations")
            break

    with open(f"{OUT_DIR}/best_individual.py", "w") as f:
        f.write(f"# id={best_ever.id} island={best_ever.island} fitness_f1={best_ever.fitness_f1:.3f} conditions={best_ever.condition_count} penalized={best_ever.penalized_fitness:.3f}\n")
        f.write(best_ever.code + "\n")

    print()
    print(f"best individual: {best_ever.id} (island {best_ever.island}) fitness_f1={best_ever.fitness_f1:.3f} penalized={best_ever.penalized_fitness:.3f}")
    print(f"outputs saved to {OUT_DIR}/")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--islands", type=int, default=3)
    parser.add_argument("--island-size", type=int, default=4)
    parser.add_argument("--generations", type=int, default=15)
    parser.add_argument("--migration-interval", type=int, default=3)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()

    run_evolution(
        n_islands=args.islands, island_size=args.island_size, max_generations=args.generations,
        migration_interval=args.migration_interval, patience=args.patience,
    )


if __name__ == "__main__":
    main()
