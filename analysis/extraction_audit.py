"""Extraction-rate audit.

Reproduces the per-cell extraction-rate audit reported in the paper appendix on extraction audit.

For every (model, dataset, condition, level) cell, compute the fraction of items
whose answer was successfully extracted by the regex extractor. The paper uses
`predicted_answer is not None` and a None/empty `predicted_answer` indicates a
parse failure under the existing extractor.

Reports:
    results/audits/cell_table.csv
    results/audits/per_model_dataset_condition.csv
    results/audits/L0_vs_L1_gap.csv  (the headline gap)
    results/audits/_summary.json
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "results" / "audits"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["qwen-2.5", "qwen-3.5", "deepseek-r1", "gemma-4", "gpt-4o", "gpt-5.4", "haiku-4.5", "sonnet-4.6"]
DATASETS = ["gsm8k", "boolq", "arc_easy", "commonsenseqa", "mmlu_stem"]
CONDITIONS = ["input", "output"]
LEVELS = ["L0", "L1", "L2", "L3", "L4"]


def parse_rate(jsonl_path: Path) -> Dict:
    if not jsonl_path.exists():
        return {"n": 0, "parsed": 0, "rate": None}
    n = 0
    parsed = 0
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            n += 1
            pa = r.get("predicted_answer")
            if pa is not None and str(pa).strip() != "":
                parsed += 1
    return {"n": n, "parsed": parsed, "rate": parsed / n if n else None}


def main():
    rows: List[Dict] = []
    for m in MODELS:
        for d in DATASETS:
            for c in CONDITIONS:
                for lvl in LEVELS:
                    p = REPO / "results" / f"{m}_{c}" / d / f"caveman_{m}_{d}_{c}_{lvl}.jsonl"
                    stats = parse_rate(p)
                    rows.append({
                        "model": m,
                        "dataset": d,
                        "condition": c,
                        "level": lvl,
                        "n": stats["n"],
                        "parsed": stats["parsed"],
                        "parse_rate": stats["rate"],
                    })

    cell_path = OUT_DIR / "cell_table.csv"
    with cell_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "dataset", "condition", "level", "n", "parsed", "parse_rate"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Headline L0 vs L1 gap
    gap_rows = []
    flagged = []
    for m in MODELS:
        for d in DATASETS:
            for c in CONDITIONS:
                l0 = next((r for r in rows if r["model"] == m and r["dataset"] == d and r["condition"] == c and r["level"] == "L0"), None)
                l1 = next((r for r in rows if r["model"] == m and r["dataset"] == d and r["condition"] == c and r["level"] == "L1"), None)
                if l0 and l1 and l0["parse_rate"] is not None and l1["parse_rate"] is not None:
                    gap = l1["parse_rate"], l0["parse_rate"]
                    gap_rows.append({
                        "model": m, "dataset": d, "condition": c,
                        "L0_parse_rate": l0["parse_rate"],
                        "L1_parse_rate": l1["parse_rate"],
                        "L1_minus_L0": gap,
                    })
                    if l0["parse_rate"] < 0.95:
                        flagged.append({"model": m, "dataset": d, "condition": c, "L0_parse_rate": l0["parse_rate"]})

    gap_path = OUT_DIR / "L0_vs_L1_gap.csv"
    with gap_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "dataset", "condition", "L0_parse_rate", "L1_parse_rate", "L1_minus_L0"])
        w.writeheader()
        for r in gap_rows:
            w.writerow(r)

    summary = {
        "n_cells": len(rows),
        "flagged_L0_parse_below_0.95": flagged,
        "n_flagged": len(flagged),
        "mean_L0_parse_rate": sum(r["L0_parse_rate"] for r in gap_rows) / max(1, len(gap_rows)),
        "mean_L1_parse_rate": sum(r["L1_parse_rate"] for r in gap_rows) / max(1, len(gap_rows)),
        "biggest_L0_to_L1_gap_cells": sorted(gap_rows, key=lambda x: -x["L1_minus_L0"])[:10],
    }
    (OUT_DIR / "_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"[extraction] wrote {len(rows)} rows to {cell_path}")
    print(f"[extraction] {len(flagged)} (model,ds,cond) cells with L0 parse_rate < 0.95")
    print(f"[extraction] mean L0 parse rate = {summary['mean_L0_parse_rate']:.4f}")
    print(f"[extraction] mean L1 parse rate = {summary['mean_L1_parse_rate']:.4f}")
    print("[extraction] biggest L0→L1 parse-rate gaps:")
    for r in summary["biggest_L0_to_L1_gap_cells"][:5]:
        print(f"   {r['model']:8s} {r['dataset']:14s} {r['condition']:6s}: "
              f"L0={r['L0_parse_rate']:.3f} L1={r['L1_parse_rate']:.3f} gap={r['L1_minus_L0']:+.3f}")


if __name__ == "__main__":
    main()
