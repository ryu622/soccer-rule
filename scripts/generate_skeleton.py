"""LLMに1v1デュエル判定タスクの「ルールの骨格(If-Then構造)」をコード生成させる。

research_plan.md 3.2節「LLMの役割」・3.3節「LLMコード生成」に対応。
- 複数回(SEEDS個)生成し、安定性を比較できるようにする
- 出力はPARAM_SPECS(閾値の探索範囲)とpredict_take_on関数のみに固定させ、
  数値そのものは後段のデータ最適化(fit_skeleton.py)でフィッティングする
- 生成結果は llm_skeletons/skeleton_seed{N}.py にそのまま保存する(再現性確認用)

uv run python scripts/generate_skeleton.py
"""

from __future__ import annotations

import os
import re

from anthropic import Anthropic
from dotenv import load_dotenv

MODEL = "claude-sonnet-5"
N_GENERATIONS = 3
OUT_DIR = "llm_skeletons"

FEATURE_DESCRIPTIONS = """\
以下は、サッカーの1v1局面(ボール保持者が守備者にプレッシャーをかけられている瞬間)における
トラッキングデータ由来の特徴量です。すべて、守備者との間合いが一定の閾値を下回った瞬間
(「プレッシャー開始」の瞬間)に計測されています。

- distance_m: ボール保持者と最も近い守備者との距離(m)。小さいほど間合いが詰まっている。
- approach_angle_deg: 最も近い守備者の進行方向と、守備者からボール保持者へ向かう直線とのなす角(度)。
  0度に近いほど守備者がボール保持者に正対して直進接近している。180度に近いほど守備者は離れる方向に動いている。
- carrier_speed_mps: ボール保持者の速度(m/s)。
- opponent_speed_mps: 最も近い守備者の速度(m/s)。
- closing_speed_mps: 間合いが詰まる速さ(m/s)。正の値は接近中、負の値は間合いが開いていることを意味する。
- second_nearest_dist_m: 2番目に近い守備者との距離(m)。大きいほど、最初の守備者をかわした後の空間的余裕が大きい。
- n_supporting_teammates: ボール保持者から半径15m以内かつ自陣側(後方)にいる味方選手の数。
  「後方サポート」の人数であり、多いほどリスクを取ってドリブルを仕掛けても後ろで拾ってもらえる安心感がある。
- dist_to_sideline_m: 最も近いタッチラインまでの距離(m)。小さいほどサイドに追い込まれている。
- dist_to_goal_line_m: 攻撃方向のゴールラインまでの距離(m)。小さいほど相手ゴールに近い(仕掛けるメリットが大きい)。
"""

TASK_DESCRIPTION = """\
このボール保持者が、次の瞬間に「仕掛ける(ドリブルで守備者を打開しようとする、TAKE_ON)」か
「仕掛けない(パスをする・後方に下げる・様子を見るなど、見送る)」かを判定する二値分類の
ルールを設計してください。

あなたの役割は、サッカーの戦術知識(間合い、進入角度、後方サポート、相手との相対的な勢い、
空間的余裕、ピッチ上の位置的リスクなど)に基づいて、複数の特徴量を組み合わせた
If-Then構造のルールの「骨格」を設計することです。個々の数値の閾値は後工程でデータから
最適化するため、あなたはその骨格(どの特徴量をどう組み合わせるか、どんな戦術概念で
名前付けするか)の設計に集中してください。
"""

PRIOR_FINDINGS = """\
このタスクにはすでに2つのベースラインを試した実績があります(同じ特徴量・同じ交差検証設定)。

- 決定木(max_depth=3, class_weight="balanced"): F1 = 0.554
- XGBoost(200本のアンサンブル): F1 = 0.634

決定木の分岐を調べたところ、7つある分岐のほとんどが`distance_m`(間合い)と
`closing_speed_mps`(間合いが詰まる速さ)の2特徴量だけで構成されており、
`n_supporting_teammates`(後方サポート人数)や`opponent_speed_mps`(相手の速度)といった
特徴量はほとんど分岐に使われていませんでした。一方XGBoostは9特徴量全てに重要度が
分散しており、これがXGBoostの優位性の一因と考えられます。

さらに重要な失敗パターンとして、過去に「AND条件を3〜4個直列に連結した骨格
(例: `A and B and (C or D) and not E`のように、最終判定の前提条件をAND鎖でどんどん
絞り込んでいく書き方)」を試したところ、F1が0.27〜0.47と決定木にすら及ばない結果に
なりました。原因は、AND鎖ではすべての条件が同時に真でなければ正例と判定されず、
どれか1つの条件の閾値が実データと少しでもズレているだけで再現率(recall)が急落する
ためです(このとき再現率は0.28〜0.60まで低下し、決定木の0.795を大きく下回りました)。

このAND鎖の失敗を踏まえ、今回は以下のいずれかの設計方針を採用してください:

(a) 「多数決/スコア方式」: 複数の戦術シグナル(bool)をそれぞれ用意し、真になった数
    (または重み付き合計スコア)を数え、その合計が閾値以上なら仕掛けると判定する。
    例えば6つのシグナルのうち4つ以上が真なら仕掛ける、といった形。
    1つの条件の閾値が多少ズレても他のシグナルでカバーできるため、AND鎖より頑健。

(b) 「浅いAND(高々2条件)+ORによる複数の経路」: 例えば
    `(条件Aかつ条件B) or (条件Cかつ条件D)` のように、独立した複数の"仕掛けるパターン"を
    ORで並列に用意する。1つのAND鎖に依存しすぎない構造にする。

いずれの場合も、単一の特徴量(例えば`distance_m`だけ)に依存する単純な閾値判定ではなく、
複数の特徴量を組み合わせた骨格にしてください。ただし3個以上の条件を単純にANDだけで
連結することは避けてください。
"""

OUTPUT_FORMAT = """\
出力は必ず以下の2つを含むPythonコードブロック1個のみとしてください。説明文やコメント以外の
自然言語は一切書かないでください。

1. `PARAM_SPECS`という名前の辞書。キーはパラメータ名(str)、値は `(low, high, initial)` の
   3要素タプル(float)。パラメータは3〜7個程度にしてください。パラメータ名は戦術的な意味が
   わかる名前にしてください(例: "tight_space_distance_m"、"support_min_count" など)。
   数値の初期値・範囲は、上記の特徴量の説明にある単位・常識的なサッカーのスケール感を踏まえて
   設定してください(例: 距離ならおおよそ0〜10m程度、角度ならおおよそ0〜180度、人数なら0〜5程度)。
   「多数決方式」を採用する場合は、閾値となる票数/スコアのパラメータ(例: "min_votes")も
   PARAM_SPECSに含めてください。

2. `predict_take_on(f: dict, p: dict) -> int` という名前の関数。
   - `f` は上記9つの特徴量を含む辞書(キー名は上記の通り)。
   - `p` は `PARAM_SPECS` のキーと同じキーを持つ辞書で、最適化後の実際のパラメータ値が入ります。
   - 戻り値は 1(仕掛ける)または 0(仕掛けない)。
   - 関数の内部では、まず戦術概念に対応する名前つきの中間変数(bool)をいくつか定義し
     (例: `is_tight_space = f["distance_m"] < p["tight_space_distance_m"]`)、
     PRIOR_FINDINGSで説明した(a)多数決/スコア方式 または (b)浅いAND+OR方式のどちらかで
     組み合わせて最終判定を行ってください。
   - `p`にないパラメータ名を使わないこと。`f`にないキーを参照しないこと。
   - `import`や外部ライブラリは使わず、標準のPython演算子・比較のみで書いてください。

出力例のフォーマット(内容はダミーです、多数決方式の例):

```python
PARAM_SPECS = {
    "example_distance_m": (1.0, 8.0, 3.0),
    "min_votes": (1.0, 5.0, 3.0),
}

def predict_take_on(f: dict, p: dict) -> int:
    is_example = f["distance_m"] < p["example_distance_m"]
    votes = int(is_example)  # 実際は複数のbool変数を足し合わせる
    return 1 if votes >= p["min_votes"] else 0
```
"""


def build_prompt() -> str:
    return f"{FEATURE_DESCRIPTIONS}\n{TASK_DESCRIPTION}\n{PRIOR_FINDINGS}\n{OUTPUT_FORMAT}"


def extract_code(text: str) -> str:
    match = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def main() -> None:
    load_dotenv()
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    os.makedirs(OUT_DIR, exist_ok=True)

    prompt = build_prompt()

    for seed in range(N_GENERATIONS):
        print(f"=== generating skeleton seed={seed} ===")
        response = client.messages.create(
            model=MODEL,
            max_tokens=2000,
            thinking={"type": "disabled"},
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        code = extract_code(text)

        out_path = f"{OUT_DIR}/skeleton_seed{seed}.py"
        with open(out_path, "w") as f:
            f.write(code + "\n")
        print(f"saved to {out_path}")
        print(code)
        print()


if __name__ == "__main__":
    main()
