"""進化計算で得られた個体のPARAM_SPECS(探索範囲)と実際にフィットされた値を突き合わせ、
探索範囲の端(閾値内)に張り付いているパラメータを機械的にリストアップする。

「端に張り付いている」パラメータは、その閾値が最も緩い(ほぼ常にTrue/False)方向に
最適化されている可能性が高く、コード上は複合条件の一部として書かれていても実質的に
判別へ寄与していない(装飾的)ことが多い。追加のLLM呼び出しは不要で、
population.jsonl(PARAM_SPECS・フィット値)だけから計算できる。

uv run python scripts/audit_pinning.py --run-id RUN_ID --individual-id ind90 [--params refit|inloop]
"""

from __future__ import annotations

import argparse
import glob
import json


def latest_run_dir() -> str:
    dirs = sorted(glob.glob("evolution_runs/*"))
    if not dirs:
        raise FileNotFoundError("evolution_runs/ にディレクトリが見つかりません")
    return dirs[-1]


def load_individual(run_dir: str, individual_id: str) -> dict:
    with open(f"{run_dir}/population.jsonl") as f:
        for line in f:
            row = json.loads(line)
            if row["id"] == individual_id:
                return row
    raise ValueError(f"{individual_id} が {run_dir}/population.jsonl に見つかりません")


def audit(param_specs: dict, fitted_params: dict, threshold_pct: float = 5.0) -> list[dict]:
    rows = []
    for name, (low, high, initial) in param_specs.items():
        span = high - low
        value = fitted_params[name]
        position = (value - low) / span
        edge_pct = min(position, 1 - position) * 100
        edge_side = "low" if position <= 0.5 else "high"
        pinned = edge_pct <= threshold_pct
        rows.append(
            {
                "param": name, "low": low, "high": high, "initial": initial, "fitted": value,
                "edge_pct": edge_pct, "edge_side": edge_side if pinned else None, "pinned": pinned,
            }
        )
    return rows


def print_audit(title: str, rows: list[dict]) -> None:
    print(f"=== {title} ===")
    n_pinned = sum(r["pinned"] for r in rows)
    print(f"{'param':<28} {'low':>6} {'high':>6} {'initial':>8} {'fitted':>10} {'edge%':>7} pinned")
    for r in rows:
        mark = f"YES({r['edge_side']})" if r["pinned"] else ""
        print(f"{r['param']:<28} {r['low']:>6.2f} {r['high']:>6.2f} {r['initial']:>8.2f} {r['fitted']:>10.3f} {r['edge_pct']:>6.1f}% {mark}")
    print(f"pinned: {n_pinned}/{len(rows)} (threshold={5.0}%)")
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--individual-id", required=True)
    parser.add_argument("--threshold-pct", type=float, default=5.0)
    args = parser.parse_args()

    run_dir = f"evolution_runs/{args.run_id}" if args.run_id else latest_run_dir()
    ind = load_individual(run_dir, args.individual_id)
    param_specs = ind["param_specs"]

    inloop_rows = audit(param_specs, ind["fitted_params"], args.threshold_pct)
    print_audit(f"{args.individual_id}: in-loop fit (evolve_train)", inloop_rows)

    try:
        held_out = json.load(open(f"{run_dir}/held_out_evaluation.json"))
        refit_rows = audit(param_specs, held_out["refit_params"], args.threshold_pct)
        print_audit(f"{args.individual_id}: refit (evolve_train+evolve_fitness, held-out評価用)", refit_rows)
    except FileNotFoundError:
        print("held_out_evaluation.json が見つからないため refit の監査はスキップ")


if __name__ == "__main__":
    main()
