"""Island型進化計算 v3: 「新フィッティング」だけを効果検証するためのシンプルな設定。

documents/phase3c_evolution_v2_results.md の一連の診断を経て、
`scripts/compare_refit_methods.py`で3つの既存コードを新フィッティング(分位点グリッド+
座標降下法の全探索、evolve_skeleton_v2.pyで実装)で再フィッティングした結果:

    - A: 11特徴量・16条件ルール(v2, penaltyあり) held-out F1 0.370→0.368(変化なし)
    - B: 9特徴量・ind90構造(v1, OR of ANDs)     held-out F1 0.491→0.528(+0.037、決定木0.554まであと0.026)
    - C: 9特徴量・v1 seed0構造(AND連鎖)         held-out F1 0.467→0.365(悪化)

B(9特徴量・OR of ANDs構造)だけが新フィッティングで明確に改善した。A・C(11特徴量、AND連鎖)は
新フィッティングでも救えないことが確認できたため、この系統は追わない。

v3はBの系統(v1のisland A/B/C相当、9特徴量)に絞り、11特徴量・複雑さペナルティを外して
v1相当のシンプルな設定に戻した上で、証明された唯一の改善要因である「新フィッティング」
(evolve_skeleton_v2.pyのinfer_param_feature_mapping/build_candidate_grid/optimize_paramsを
そのまま再利用)だけを組み込む。これにより、進化計算(mutation/crossover相当のfresh_randomに
よる改良)自体もこれまでの弱いフィッティングではなく正しい評価のもとで回せるようにする。

v2から維持する変更(独立に効果検証済みのため):
    - 適応度: evolve側5試合の内側leave-one-out平均F1(単一試合への過適合を緩和)
    - エリート1体+fresh_random(v2で診断・修正した多様性崩壊対策)、移住間隔3世代
    - ロギング順序修正(migrateより前にlog_generation_snapshot)
    - APIリトライ

v2から外す変更:
    - 特徴量: 11→9(元の特徴量セットに戻す)
    - 複雑さペナルティ: 除去(LAMBDA_COMPLEXITY=0、選択は生のfitness_f1で行う)

uv run python scripts/evolve_skeleton_v3.py [--islands N] [--island-size N] [--generations N]
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
OUT_DIR = f"evolution_runs_v3/{RUN_ID}"

DATA_PATH = "data/tackling_features_v2.csv"  # 9特徴量のみ使用(11特徴量版の上位互換ファイルだが新規2列は参照しない)
FEATURE_COLUMNS = [
    "distance_m", "approach_angle_deg", "carrier_speed_mps", "opponent_speed_mps",
    "closing_speed_mps", "second_nearest_dist_m", "n_supporting_teammates",
    "dist_to_sideline_m", "dist_to_goal_line_m",
]

# 固定シード(random.Random(42))によるオリジナルの試合割り当て(evolve_skeleton.pyと同一)を継承。
HELD_OUT_TEST_MATCHES = ["J03WMX", "J03WOH"]
EVOLVE_SET_MATCHES = ["J03WN1", "J03WOY", "J03WQQ", "J03WR9", "J03WPY"]

RNG_SEED_PARAM_FIT = 0
EVO_RNG_SEED = 42
N_RANDOM_SEARCH = 700
N_REFINE_ROUNDS = 5
LAMBDA_COMPLEXITY = 0.0  # v3では複雑さペナルティを外す(選択は生のfitness_f1で行う)

_id_counter = itertools.count(1)

FEATURE_DESCRIPTIONS = FEATURE_DESCRIPTIONS_V1

# v1(scripts/evolve_skeleton.py)と同一の出力フォーマット指定(9特徴量、bool特徴量の追加指示なし)。
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


def infer_param_feature_mapping(code: str) -> dict[str, str]:
    """`f["feature"] <op> p["param"]` の形の比較をASTから機械的に検出し、
    パラメータ名 -> 比較されている生特徴量名 の対応を推定する(決定木が生特徴量の
    実データ値を候補にするのと同じことをするための下ごしらえ)。
    1つのパラメータが複数の異なる特徴量と比較されている場合は対応関係が曖昧なため除外する。
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {}
    mapping: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        operands = [node.left, *node.comparators]
        feat, param = None, None
        for op in operands:
            if isinstance(op, ast.Subscript) and isinstance(op.value, ast.Name) and isinstance(op.slice, ast.Constant):
                if op.value.id == "f":
                    feat = op.slice.value
                elif op.value.id == "p":
                    param = op.slice.value
        if feat is not None and param is not None:
            mapping.setdefault(param, set()).add(feat)
    return {p: next(iter(feats)) for p, feats in mapping.items() if len(feats) == 1}


def build_candidate_grid(
    param_specs: dict, param_feature_map: dict[str, str], rows: list[dict], n_percentile_points: int = 21
) -> dict[str, list[float]]:
    """各パラメータについて、対応する生特徴量の実データ分位点(5%刻み)から候補値リストを作る。
    対応する特徴量が推定できなかったパラメータは、PARAM_SPECSのlow-highを均等分割したグリッドにフォールバックする。
    """
    candidates: dict[str, list[float]] = {}
    for name, (low, high, _initial) in param_specs.items():
        feature = param_feature_map.get(name)
        values = None
        if feature is not None:
            values = sorted(r[feature] for r in rows if r.get(feature) is not None)
        if values:
            pts = []
            for i in range(n_percentile_points):
                pct = i / (n_percentile_points - 1)
                idx = int(pct * (len(values) - 1))
                v = values[idx]
                pts.append(min(high, max(low, v)))
            candidates[name] = sorted(set(pts))
        else:
            candidates[name] = [low + (high - low) * i / (n_percentile_points - 1) for i in range(n_percentile_points)]
    return candidates


def optimize_params(
    fn, rows: list[dict], y, param_specs: dict, rng: random.Random, candidates: dict[str, list[float]]
) -> tuple[dict, float]:
    """決定木の分岐選びと同じ考え方: 候補閾値は実データの分位点(build_candidate_gridで
    事前計算)のみから選ぶ。連続空間のランダムサーチ・±10%の局所摂動は行わない。
    1. ランダムサーチ: 候補グリッドから700回サンプリング
    2. 座標降下法: 各パラメータについて候補グリッドを全探索(決定木の分岐選びそのもの)し、
       改善がなくなるまで繰り返す
    """
    names = list(param_specs.keys())
    best_params = {name: param_specs[name][2] for name in names}
    best_f1 = f1_score(y, [safe_predict(fn, row, best_params) for row in rows], zero_division=0)

    for _ in range(N_RANDOM_SEARCH):
        candidate = {name: rng.choice(candidates[name]) for name in names}
        f1 = f1_score(y, [safe_predict(fn, row, candidate) for row in rows], zero_division=0)
        if f1 > best_f1:
            best_f1, best_params = f1, candidate

    for _ in range(N_REFINE_ROUNDS):
        improved = False
        for name in names:
            for val in candidates[name]:
                candidate = dict(best_params)
                candidate[name] = val
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
        param_feature_map = infer_param_feature_mapping(code)
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

            # 候補閾値はheld_out(このfoldの評価試合)を除いたtrain_rowsのみから計算する(リーク回避)
            candidates = build_candidate_grid(param_specs, param_feature_map, train_rows)
            rng = random.Random(RNG_SEED_PARAM_FIT)
            fitted_params, _train_f1 = optimize_params(fn, train_rows, train_y, param_specs, rng, candidates)
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
