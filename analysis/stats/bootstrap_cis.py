"""Bootstrap 95% CIs on per-cell C2 / C3 / bidir-NLI rates.

Loads the existing per-(model, dataset, condition, level) `_with_entailment`
JSONL files (already on disk), computes the per-cell C1/C2/C3/C4 counts, and
runs 1000-resample bootstrap CIs for each cell. Also produces an L1-B
grand-mean CI matching the paper's 52% headline.

Outputs:
    results/revision/bootstrap_cis/per_cell.csv
    results/revision/bootstrap_cis/L1B_grand.json
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import List, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
REPO = _REPO_ROOT
import sys as _sys
_sys.path.insert(0, str(_REPO_ROOT / "src"))
from release_paths import results_root  # noqa: E402
_RESULTS = results_root(_REPO_ROOT / "results")

OUT_DIR = _RESULTS / "revision" / "bootstrap_cis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# The eight-model panel the paper reports significance over.
MODELS = ["qwen2.5", "qwen3.5", "deepseek", "gemma4", "gpt4o", "gpt5.4", "haiku", "sonnet"]
DATASETS = ["gsm8k", "boolq", "arc_easy", "commonsenseqa", "mmlu_stem"]
CONDITIONS = ["input", "output"]
LEVELS = ["L1", "L2", "L3"]
N_BOOT = 1000
RNG_SEED = 4242


def load_cells(model: str, dataset: str, cond: str, lvl: str) -> List[Tuple[bool, bool]]:
    p = _RESULTS / f"{model}_{cond}" / dataset / f"caveman_{model}_{dataset}_{cond}_{lvl}_with_entailment.jsonl"
    if not p.exists():
        return []
    rows = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            correct = bool(r.get("correct"))
            bidir = bool(r.get("bidirectional_entailment", False))
            rows.append((correct, bidir))
    return rows


def bootstrap_ci(values: List[float], n_boot: int = N_BOOT, alpha: float = 0.05) -> Tuple[float, float, float]:
    if not values:
        return (float("nan"), float("nan"), float("nan"))
    n = len(values)
    mean = sum(values) / n
    rng = random.Random(RNG_SEED)
    samples = []
    for _ in range(n_boot):
        s = [values[rng.randrange(n)] for _ in range(n)]
        samples.append(sum(s) / n)
    samples.sort()
    lo = samples[int(n_boot * (alpha / 2))]
    hi = samples[int(n_boot * (1 - alpha / 2))]
    return (mean, lo, hi)


def main():
    rows = []
    l1b_all_C2 = []  # paper-style headline: pooled L1-B C2 across all models/datasets
    l1b_all_C3 = []

    for m in MODELS:
        for d in DATASETS:
            for c in CONDITIONS:
                for lvl in LEVELS:
                    items = load_cells(m, d, c, lvl)
                    if not items:
                        continue
                    n = len(items)
                    # C2 indicator per item: correct AND not bidir
                    c2 = [1.0 if (correct and not bidir) else 0.0 for correct, bidir in items]
                    c3 = [1.0 if (not correct and bidir) else 0.0 for correct, bidir in items]
                    c1 = [1.0 if (correct and bidir) else 0.0 for correct, bidir in items]
                    c4 = [1.0 if (not correct and not bidir) else 0.0 for correct, bidir in items]
                    bidir_rate = [1.0 if bidir else 0.0 for _, bidir in items]
                    correct_rate = [1.0 if correct else 0.0 for correct, _ in items]

                    c2_m, c2_lo, c2_hi = bootstrap_ci(c2)
                    c3_m, c3_lo, c3_hi = bootstrap_ci(c3)
                    rows.append({
                        "model": m, "dataset": d, "condition": c, "level": lvl,
                        "n": n,
                        "C1": sum(c1) / n, "C2": c2_m, "C3": c3_m, "C4": sum(c4) / n,
                        "C2_lo": c2_lo, "C2_hi": c2_hi,
                        "C3_lo": c3_lo, "C3_hi": c3_hi,
                        "bidir_rate": sum(bidir_rate) / n,
                        "accuracy": sum(correct_rate) / n,
                    })
                    if c == "output" and lvl == "L1":
                        l1b_all_C2.extend(c2)
                        l1b_all_C3.extend(c3)
                    print(f"[boot] {m:8s} {d:14s} {c:6s} {lvl} n={n:5d} "
                          f"C2={c2_m:.4f} [{c2_lo:.4f}, {c2_hi:.4f}]")

    out_csv = OUT_DIR / "per_cell.csv"
    if not rows:
        raise SystemExit(
            "no per-cell rows found. Check that results_root points at a tree of\n"
            "<tag>_<condition>/<dataset>/caveman_*.jsonl files with entailment\n"
            "sidecars, and that the model tags match the directory names."
        )
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # L1-B grand pooled CI
    headline = bootstrap_ci(l1b_all_C2)
    c3_grand = bootstrap_ci(l1b_all_C3)
    summary = {
        "n_total_L1B": len(l1b_all_C2),
        "L1B_grand_C2_mean": headline[0],
        "L1B_grand_C2_95CI": [headline[1], headline[2]],
        "L1B_grand_C3_mean": c3_grand[0],
        "L1B_grand_C3_95CI": [c3_grand[1], c3_grand[2]],
        "n_boot": N_BOOT,
        "seed": RNG_SEED,
    }
    (OUT_DIR / "L1B_grand.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n=== L1-B grand headline ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
