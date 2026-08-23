"""Length-controlled bidirectional NLI.

For each (model, dataset, item) at L1, L2, L3 (Condition B), tokenize the L0
output and truncate it to match the L1/L2/L3 output token length, then run
bidirectional DeBERTa-v3 NLI on (truncated_L0, Lx). The C2 rate computed on
these length-matched pairs isolates propositional drift from verbosity
asymmetry.

Uses whitespace-token length as the bridging unit since the L0 token count is
not stored per-row; this is a coarse but consistent approximation.

Outputs:
    results/revision/length_control/<model>__<ds>__<cond>__<lvl>.jsonl
    results/revision/length_control/_grand_summary.json
"""
from __future__ import annotations

import os
import sys as _sys
from glob import glob as _glob

# Some torch builds need cuSPARSELt on LD_LIBRARY_PATH before the NLI model
# loads. Resolved from the active env; no-op if absent.
_cusparselt = _glob(
    os.path.join(_sys.prefix, "lib", "python*", "site-packages",
                 "nvidia", "cusparselt", "lib")
)
if _cusparselt:
    os.environ["LD_LIBRARY_PATH"] = (
        _cusparselt[0] + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    )

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

REPO = Path(__file__).resolve().parents[2]
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
import sys as _sys
_sys.path.insert(0, str(_REPO_ROOT / "src"))
from release_paths import results_root  # noqa: E402
_RESULTS = results_root(_REPO_ROOT / "results")

sys.path.insert(0, str(_REPO_ROOT / "src"))

from entailment_scorer import load_nli_model, batch_score_entailment  # noqa: E402

OUT_DIR = _RESULTS / "revision" / "length_control"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# The eight-model panel.
MODELS = ["qwen2.5", "qwen3.5", "deepseek", "gemma4", "gpt4o", "gpt5.4", "haiku", "sonnet"]
DATASETS = ["gsm8k", "boolq", "arc_easy", "commonsenseqa", "mmlu_stem"]
CONDITIONS = ["input", "output"]
LEVELS = ["L1", "L2", "L3"]


def load_jsonl(p: Path) -> List[Dict]:
    if not p.exists():
        return []
    with p.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_l0(model: str, dataset: str, cond: str) -> Dict[int, str]:
    p = _RESULTS / f"{model}_{cond}" / dataset / f"caveman_{model}_{dataset}_{cond}_L0.jsonl"
    out = {}
    for r in load_jsonl(p):
        out[int(r["idx"])] = (r.get("output") or "").strip()
    return out


def truncate_to_length(text: str, n_words: int) -> str:
    if n_words <= 0:
        return ""
    toks = text.split()
    if len(toks) <= n_words:
        return text
    return " ".join(toks[:n_words])


def score_cell(nli, m, d, c, lvl, batch_size) -> Dict:
    src = _RESULTS / f"{m}_{c}" / d / f"caveman_{m}_{d}_{c}_{lvl}.jsonl"
    rows = load_jsonl(src)
    if not rows:
        return {"model": m, "dataset": d, "condition": c, "level": lvl, "n": 0, "skipped": True}
    l0 = load_l0(m, d, c)
    if not l0:
        return {"model": m, "dataset": d, "condition": c, "level": lvl, "n": 0, "skipped": True}

    pairs: List[Tuple[str, str]] = []
    meta: List[Tuple[int, bool, int, int]] = []  # idx, correct, n_words_lx, n_words_l0
    for r in rows:
        idx = int(r["idx"])
        lx_out = (r.get("output") or "").strip()
        l0_out = l0.get(idx, "")
        lx_len = len(lx_out.split())
        l0_full_len = len(l0_out.split())
        l0_trunc = truncate_to_length(l0_out, lx_len)
        pairs.append((lx_out, l0_trunc))   # Lx -> truncated L0
        pairs.append((l0_trunc, lx_out))   # truncated L0 -> Lx
        meta.append((idx, bool(r.get("correct")), lx_len, l0_full_len))

    t0 = time.time()
    results = batch_score_entailment(nli, pairs, batch_size=batch_size)
    elapsed = time.time() - t0

    n_bidir = 0
    n_correct = 0
    n_c2 = 0
    out_path = OUT_DIR / f"{m}__{d}__{c}__{lvl}.jsonl"
    with out_path.open("w", encoding="utf-8") as fout:
        for i, (idx, correct, lx_len, l0_len) in enumerate(meta):
            lx_to_l0 = results[2 * i]
            l0_to_lx = results[2 * i + 1]
            bidir = (lx_to_l0["label"] == "entailment") and (l0_to_lx["label"] == "entailment")
            if bidir:
                n_bidir += 1
            if correct:
                n_correct += 1
                if not bidir:
                    n_c2 += 1
            fout.write(json.dumps({
                "idx": idx,
                "correct": correct,
                "lx_n_words": lx_len,
                "l0_n_words_full": l0_len,
                "length_controlled_bidir": bool(bidir),
            }) + "\n")

    n = len(meta)
    return {
        "model": m,
        "dataset": d,
        "condition": c,
        "level": lvl,
        "n": n,
        "lc_bidir_rate": n_bidir / n,
        "lc_C2_rate": n_c2 / n,
        "accuracy": n_correct / n,
        "elapsed_s": elapsed,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default=None,
                       help="Path to the results tree "
                            "(default: <repo>/results, or $CAVEWOMAN_RESULTS).")
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--conditions", nargs="+", default=CONDITIONS)
    ap.add_argument("--levels", nargs="+", default=LEVELS)
    ap.add_argument("--batch_size", type=int, default=128)
    args = ap.parse_args()

    nli = load_nli_model()
    summaries: List[Dict] = []
    for m in args.models:
        for d in args.datasets:
            for c in args.conditions:
                for lvl in args.levels:
                    try:
                        s = score_cell(nli, m, d, c, lvl, args.batch_size)
                        summaries.append(s)
                        if not s.get("skipped"):
                            print(f"[lc] {m}/{d}/{c}/{lvl} n={s['n']} "
                                  f"lc_C2={s['lc_C2_rate']:.4f} "
                                  f"lc_bidir={s['lc_bidir_rate']:.4f} "
                                  f"t={s['elapsed_s']:.1f}s", flush=True)
                    except Exception as e:
                        print(f"[ERROR] {m}/{d}/{c}/{lvl}: {type(e).__name__}: {e}", flush=True)

    l1b = [s for s in summaries if not s.get("skipped") and s["condition"] == "output" and s["level"] == "L1"]
    n_tot = sum(s["n"] for s in l1b)
    n_c2 = sum(int(s["lc_C2_rate"] * s["n"]) for s in l1b)
    grand = {
        "all_cells": summaries,
        "L1B_grand": {
            "n_cells": len(l1b),
            "n_total": n_tot,
            "lc_C2_rate": n_c2 / n_tot if n_tot else None,
        },
    }
    (OUT_DIR / "_grand_summary.json").write_text(json.dumps(grand, indent=2), encoding="utf-8")
    print(f"\n=== LENGTH-CONTROLLED L1-B GRAND: C2={grand['L1B_grand']['lc_C2_rate']} ===", flush=True)


if __name__ == "__main__":
    main()
