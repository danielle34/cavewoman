"""Per-item adjudicated correctness labels + Wilcoxon/BH significance rerun.

Stage 1, labels: for every (model, dataset, condition, level) cell, join the
production per-item JSONL with the recovery-judge output:

    adj_correct = regex_correct OR (judge verdict == CORRECT)

Judge records are deduped by idx keeping the last parseable verdict. Failed
items never judged (sample-tier remainder) stay incorrect, conservative, and
identical in expectation to the Wilson-CI treatment in adjudicated_accuracy.py.
Writes one gzipped CSV: results/revision/adjudicated/item_labels.csv.gz

Stage 2, significance: paired Wilcoxon signed-rank per (model, dataset,
condition, level>L0) comparing adjudicated per-item correctness at Lx vs L0
(same idx), Benjamini-Hochberg across the whole family (alpha=.05). Writes
results/revision/adjudicated/wilcoxon_adjudicated.csv and, if the original
raw-label FDR table exists, a star-change comparison
(wilcoxon_star_changes.csv).

CPU only.
"""
from __future__ import annotations

import csv
import gzip
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
import sys as _sys
_sys.path.insert(0, str(_REPO_ROOT / "src"))
from release_paths import results_root  # noqa: E402
_RESULTS = results_root(_REPO_ROOT / "results")

RESULTS = _RESULTS
JUDGE = RESULTS / "judge_qwen35"
OUTDIR = RESULTS / "revision" / "adjudicated"

sys.path.insert(0, str(_REPO_ROOT / "analysis" / "judge"))
import run_judge  # noqa: E402

# The eight-model panel; Kimi-K2.6 has no visible text to adjudicate.
MODELS = ["qwen2.5", "qwen3.5", "deepseek", "gemma4", "gpt4o", "gpt5.4", "haiku", "sonnet"]
DATASETS = ["gsm8k", "boolq", "arc_easy", "commonsenseqa", "mmlu_stem"]
CONDITIONS = ["output", "input"]
LEVELS = ["L0", "L1", "L2", "L3", "L4"]
ALPHA = 0.05


def recovery_path(m, d, c, lv):
    if lv == "L0":
        return JUDGE / f"{m}_{d}_{c}_recovery.jsonl"
    return JUDGE / f"{m}_{d}_{c}_recovery_{lv}.jsonl"


def judged_correct_idx(m, d, c, lv) -> set:
    p = recovery_path(m, d, c, lv)
    if not p.exists():
        return set()
    best = {}
    for r in run_judge.load_jsonl(p):
        idx = r.get("idx")
        prev = best.get(idx)
        if prev is None or r.get("judge_verdict") or not prev.get("judge_verdict"):
            best[idx] = r
    return {i for i, r in best.items() if r.get("judge_verdict") == "CORRECT"}


def bh_correct(pvals: list[float], alpha: float) -> list[bool]:
    n = len(pvals)
    order = np.argsort(pvals)
    passed = [False] * n
    max_k = -1
    for rank, i in enumerate(order, start=1):
        if pvals[i] <= alpha * rank / n:
            max_k = rank
    for rank, i in enumerate(order, start=1):
        if rank <= max_k:
            passed[i] = True
    return passed


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    labels = {}   # (m,d,c,lv) -> {idx: (raw, adj)}
    for m in MODELS:
        for d in DATASETS:
            for c in CONDITIONS:
                for lv in LEVELS:
                    src = run_judge.jsonl_path(str(RESULTS), m, c, d, lv)
                    if not src.exists():
                        continue
                    jc = judged_correct_idx(m, d, c, lv)
                    cell = {}
                    for r in run_judge.load_jsonl(src):
                        idx = r.get("idx")
                        if idx is None:
                            continue
                        raw = bool(r.get("correct"))
                        adj = raw or (r.get("predicted_answer") is None and idx in jc)
                        cell[idx] = (raw, adj)
                    labels[(m, d, c, lv)] = cell
        print(f"[labels] {m} done", flush=True)

    with gzip.open(OUTDIR / "item_labels.csv.gz", "wt", newline="",
                   encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "dataset", "condition", "level", "idx", "raw_correct", "adj_correct"])
        for (m, d, c, lv), cell in labels.items():
            for idx, (raw, adj) in cell.items():
                w.writerow([m, d, c, lv, idx, int(raw), int(adj)])
    print(f"[labels] wrote {OUTDIR/'item_labels.csv.gz'}", flush=True)

    # Stage 2: Wilcoxon Lx vs L0 on adjudicated labels
    rows = []
    for (m, d, c, lv), cell in labels.items():
        if lv == "L0":
            continue
        l0 = labels.get((m, d, c, "L0"))
        if not l0:
            continue
        common = sorted(set(cell) & set(l0))
        if len(common) < 10:
            continue
        adj_x = np.array([cell[i][1] for i in common], dtype=float)
        adj_0 = np.array([l0[i][1] for i in common], dtype=float)
        diff = adj_x - adj_0
        if np.all(diff == 0):
            p = 1.0
        else:
            try:
                p = float(wilcoxon(adj_x, adj_0, zero_method="wilcox").pvalue)
            except ValueError:
                p = 1.0
        rows.append({
            "model": m, "dataset": d, "condition": c, "level": lv,
            "n": len(common),
            "adj_acc_L0": round(float(adj_0.mean()), 4),
            "adj_acc_Lx": round(float(adj_x.mean()), 4),
            "delta": round(float(diff.mean()), 4),
            "p_raw": p,
        })
    passed = bh_correct([r["p_raw"] for r in rows], ALPHA)
    for r, ok in zip(rows, passed):
        r["significant_BH05"] = int(ok)

    if not rows:
        raise SystemExit(
            "no cells found. This script needs judge verdicts in\n"
            "results/judge_qwen35/ alongside the per-cell generation files."
        )
    with open(OUTDIR / "wilcoxon_adjudicated.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    n_sig = sum(passed)
    print(f"[wilcoxon] {len(rows)} cells, {n_sig} significant after BH at {ALPHA}", flush=True)

    # Star-change comparison vs the original raw-label family, if present
    orig = RESULTS / "revision" / "fdr" / "wilcoxon_corrected.csv"
    if orig.exists():
        with open(orig, encoding="utf-8") as f:
            old = {(r["model"], r["dataset"], r["condition"], r["level"]):
                   r for r in csv.DictReader(f)}
        changes = []
        for r in rows:
            k = (r["model"], r["dataset"], r["condition"], r["level"])
            o = old.get(k)
            if not o:
                continue
            sig_cols = [cname for cname in o if "signif" in cname.lower() or "reject" in cname.lower()]
            old_sig = None
            if sig_cols:
                v = o[sig_cols[0]].strip().lower()
                old_sig = v in ("1", "true", "yes")
            if old_sig is not None and old_sig != bool(r["significant_BH05"]):
                changes.append({**{kk: r[kk] for kk in ("model", "dataset", "condition", "level", "delta")},
                                "was_significant": old_sig,
                                "now_significant": bool(r["significant_BH05"])})
        with open(OUTDIR / "wilcoxon_star_changes.csv", "w", newline="", encoding="utf-8") as f:
            if changes:
                w = csv.DictWriter(f, fieldnames=list(changes[0].keys()))
                w.writeheader(); w.writerows(changes)
        print(f"[stars] {len(changes)} cells changed significance vs raw family "
              f"-> wilcoxon_star_changes.csv", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
