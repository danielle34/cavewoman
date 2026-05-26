"""L4 Condition A output-length distribution.

Reproduces the L4-A length distribution table reported in the paper appendix on extraction audit.

The paper notes (one sentence) that L4-A was not externally enforced. This
script quantifies how often the soft 15-token budget was violated and by how
much, per model, on each dataset. It also handles L4-B for comparison.

Uses the `output_tokens` field already present in every JSONL row (counted by
each model's own tokenizer). Reports mean, median, p95 and violation rate
(fraction of items whose output_tokens > 15 for Condition A, or > 20 for
Condition B which uses max_new_tokens=20 to allow finishing under subword
tokenization).

Outputs:
    results/audits/l4_length_distribution.csv
    results/audits/l4_summary.json
"""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Dict, List

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "results" / "audits"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["qwen-2.5", "qwen-3.5", "deepseek-r1", "gemma-4", "gpt-4o", "gpt-5.4", "haiku-4.5", "sonnet-4.6"]
DATASETS = ["gsm8k", "boolq", "arc_easy", "commonsenseqa", "mmlu_stem"]
CONDITIONS = ["input", "output"]

BUDGET_A = 15
BUDGET_B = 20


def collect_l4(model: str, dataset: str, cond: str) -> List[int]:
    p = REPO / "results" / f"{model}_{cond}" / dataset / f"caveman_{model}_{dataset}_{cond}_L4.jsonl"
    if not p.exists():
        return []
    out = []
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            ot = r.get("output_tokens")
            if isinstance(ot, (int, float)):
                out.append(int(ot))
    return out


def summarize(toks: List[int], budget: int) -> Dict:
    if not toks:
        return {"n": 0, "mean": None, "median": None, "p95": None, "max": None, "violation_rate": None, "mean_overshoot": None}
    n = len(toks)
    sorted_t = sorted(toks)
    p95 = sorted_t[int(0.95 * (n, 1))]
    over = [t for t in toks if t > budget]
    return {
        "n": n,
        "mean": sum(toks) / n,
        "median": statistics.median(toks),
        "p95": p95,
        "max": max(toks),
        "violation_rate": len(over) / n,
        "mean_overshoot": (sum(over) / len(over)) if over else 0.0,
    }


def main():
    rows: List[Dict] = []
    for m in MODELS:
        for d in DATASETS:
            for c in CONDITIONS:
                toks = collect_l4(m, d, c)
                budget = BUDGET_A if c == "input" else BUDGET_B
                stats = summarize(toks, budget)
                rows.append({
                    "model": m,
                    "dataset": d,
                    "condition": c,
                    "budget": budget,
                    **stats,
                })
                if stats["n"] > 0:
                    print(f"[L4] {m:8s} {d:14s} {c:6s} budget={budget} n={stats['n']:5d} "
                          f"mean={stats['mean']:.1f} median={stats['median']:.0f} "
                          f"p95={stats['p95']:.0f} max={stats['max']:.0f} "
                          f"violation_rate={stats['violation_rate']:.3f}")

    out_csv = OUT_DIR / "l4_length_distribution.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "dataset", "condition", "budget", "n", "mean", "median", "p95", "max", "violation_rate", "mean_overshoot"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Per-condition aggregates
    a_rows = [r for r in rows if r["condition"] == "input" and r["n"]]
    b_rows = [r for r in rows if r["condition"] == "output" and r["n"]]

    def grand(group):
        total_n = sum(r["n"] for r in group)
        if not total_n:
            return None
        weighted_mean = sum(r["mean"] * r["n"] for r in group) / total_n
        weighted_violation = sum(r["violation_rate"] * r["n"] for r in group) / total_n
        return {
            "n_cells": len(group),
            "n_items": total_n,
            "weighted_mean_output_tokens": weighted_mean,
            "weighted_violation_rate": weighted_violation,
            "max_violation_rate_cell": max(group, key=lambda r: r["violation_rate"]),
        }

    summary = {
        "L4_input (budget=15, soft)": grand(a_rows),
        "L4_output (budget=20, soft via prompt)": grand(b_rows),
    }
    (OUT_DIR / "l4_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print("\n=== L4 grand ===")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
