"""Benjamini-Hochberg FDR correction for the Wilcoxon test family.

Re-runs paired Wilcoxon signed-rank tests for accuracy at each (model, dataset,
condition, level) cell vs the within-model, within-condition L0 baseline, then
applies BH-FDR at alpha=0.05 across the full test family.

Outputs:
    results/revision/fdr/wilcoxon_corrected.csv
    results/revision/fdr/_summary.json
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
REPO = _REPO_ROOT
import sys as _sys
_sys.path.insert(0, str(_REPO_ROOT / "src"))
from release_paths import results_root  # noqa: E402
_RESULTS = results_root(_REPO_ROOT / "results")

OUT_DIR = _RESULTS / "revision" / "fdr"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# The eight-model panel the paper reports significance over.
MODELS = ["qwen2.5", "qwen3.5", "deepseek", "gemma4", "gpt4o", "gpt5.4", "haiku", "sonnet"]
DATASETS = ["gsm8k", "boolq", "arc_easy", "commonsenseqa", "mmlu_stem"]
CONDITIONS = ["input", "output"]
LEVELS = ["L1", "L2", "L3", "L4"]


def load_correct(model: str, dataset: str, cond: str, lvl: str) -> Dict[int, int]:
    p = _RESULTS / f"{model}_{cond}" / dataset / f"caveman_{model}_{dataset}_{cond}_{lvl}.jsonl"
    if not p.exists():
        return {}
    out = {}
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out[int(r["idx"])] = 1 if r.get("correct") else 0
    return out


def wilcoxon_signed_rank(diff: List[int]) -> Tuple[float, float]:
    """Two-sided Wilcoxon signed-rank test on integer differences.

    Returns (W_statistic, p_value). Uses scipy.stats.wilcoxon for correctness.
    """
    nz = [d for d in diff if d != 0]
    if not nz:
        return (0.0, 1.0)
    try:
        from scipy.stats import wilcoxon  # type: ignore
        res = wilcoxon(nz, zero_method="wilcox", alternative="two-sided", correction=False)
        return (float(res.statistic), float(res.pvalue))
    except Exception as e:
        return (0.0, float("nan"))


def bh_fdr(pvals: List[float], alpha: float = 0.05) -> List[bool]:
    """Return rejected[i] = True if p_i passes BH-FDR at the given alpha."""
    n = len(pvals)
    order = sorted(range(n), key=lambda i: (pvals[i] if pvals[i] == pvals[i] else 1.0))
    ranks = [0] * n
    for rk, idx in enumerate(order, start=1):
        ranks[idx] = rk
    # Find largest rank k with p_(k) <= k/n * alpha
    sorted_p = sorted(pvals)
    k_max = 0
    for k in range(1, n + 1):
        threshold = k * alpha / n
        if sorted_p[k - 1] <= threshold:
            k_max = k
    p_cutoff = sorted_p[k_max - 1] if k_max else -1.0
    return [pv <= p_cutoff for pv in pvals]


def main():
    cells = []
    for m in MODELS:
        for d in DATASETS:
            for c in CONDITIONS:
                l0 = load_correct(m, d, c, "L0")
                if not l0:
                    continue
                for lvl in LEVELS:
                    lx = load_correct(m, d, c, lvl)
                    if not lx:
                        continue
                    common = sorted(set(l0.keys()) & set(lx.keys()))
                    diffs = [lx[i] - l0[i] for i in common]
                    W, p = wilcoxon_signed_rank(diffs)
                    cells.append({
                        "model": m, "dataset": d, "condition": c, "level": lvl,
                        "n": len(diffs),
                        "L0_acc": sum(l0[i] for i in common) / len(common),
                        "Lx_acc": sum(lx[i] for i in common) / len(common),
                        "delta_acc": (sum(lx[i] for i in common) - sum(l0[i] for i in common)) / len(common),
                        "W": W,
                        "p_uncorrected": p,
                    })

    pvals = [c["p_uncorrected"] for c in cells]
    rejected = bh_fdr(pvals, alpha=0.05)
    for c, r in zip(cells, rejected):
        c["bh_significant"] = bool(r)

    out_csv = OUT_DIR / "wilcoxon_corrected.csv"
    if not cells:
        raise SystemExit(
            "no test cells found. Check that results_root points at a tree of\n"
            "<tag>_<condition>/<dataset>/caveman_*.jsonl files with entailment\n"
            "sidecars, and that the model tags match the directory names."
        )
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(cells[0].keys()))
        w.writeheader()
        for c in cells:
            w.writerow(c)

    n_total = len(cells)
    n_sig_uncorr = sum(1 for c in cells if c["p_uncorrected"] <= 0.05)
    n_sig_bh = sum(1 for c in cells if c["bh_significant"])
    summary = {
        "n_tests": n_total,
        "alpha": 0.05,
        "n_significant_uncorrected_at_0.05": n_sig_uncorr,
        "n_significant_BH_FDR_at_0.05": n_sig_bh,
        "delta_lost_by_correction": n_sig_uncorr - n_sig_bh,
    }
    (OUT_DIR / "_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
