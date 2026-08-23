"""Register floor: L0-A vs L0-B entailment across the model panel.

Measures how far the two channels' unconstrained outputs already sit apart
before any compression is applied:

  * Full 8-model panel (adds gemma4, which R1 omitted); Kimi excluded paper-wide.
  * Parse-rate gate: exclude (model, dataset) cells whose Condition-A (input) L0
    answer-extraction rate < 0.95 from the floor, and flag them separately. Same
    rule the paper uses before any "compressed exceeds unconstrained" claim
    (cavewoman.tex S "Answer-extraction"), so e.g. Sonnet L0-A CSQA (parse 0.037)
    does not contaminate the floor with unparseable text.
  * Headline denominator: 6 non-reasoning models (excl. DeepSeek-R1, Qwen3.5);
    8-model aggregate reported alongside.

The judge is reused VERBATIM from scripts/entailment_scorer.py:
  cross-encoder/nli-deberta-v3-base, argmax label == 'entailment' per direction,
  bidirectional = both directions argmax to 'entailment'. Loaded on CPU.

For each (model, dataset, item) present in BOTH L0-A and L0-B we score the pair
(L0-A text, L0-B text) in both directions. The "floor" is the non-entailment
rate (1 - bidirectional entailment rate). We also report the correct-both
non-entailment rate, the direct analogue of the 52% C2 headline.

Outputs (results/revision/l0_noise_floor/):
    <model>__<dataset>__pairs.jsonl
    <model>__<dataset>__summary.json
    pairs_all.csv            (model, dataset, item, entail_AtoB, entail_BtoA, bidirectional_pass)
    summary_table.csv        (per-cell + per-model + per-dataset + panel rollups)
    _grand_summary.json
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
import csv
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

from entailment_scorer import NLI_MODEL_NAME, batch_score_entailment  # noqa: E402

# 8-model panel (paper S models). Kimi-K2.6 is excluded paper-wide.
PANEL_8 = ["qwen2.5", "qwen3.5", "deepseek", "gemma4", "gpt4o", "gpt5.4", "haiku", "sonnet"]
# Reasoning models grouped out of the divergence headline.
REASONING = {"qwen3.5", "deepseek"}
NON_REASONING_6 = [m for m in PANEL_8 if m not in REASONING]
DATASETS = ["gsm8k", "boolq", "arc_easy", "commonsenseqa", "mmlu_stem"]

PARSE_GATE = 0.95  # exclude a cell when L0-A parse rate is below this

OUT_DIR = _RESULTS / "revision" / "l0_noise_floor"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_nli_cpu():
    """Load the same cross-encoder judge as the headline, pinned to CPU."""
    from sentence_transformers import CrossEncoder
    print(f"[nli] Loading {NLI_MODEL_NAME} on CPU ...", flush=True)
    model = CrossEncoder(NLI_MODEL_NAME, device="cpu")
    try:
        n_params = sum(p.numel() for p in model.model.parameters())
        print(f"[nli] Loaded: ~{n_params / 1e6:.1f}M parameters", flush=True)
    except Exception:
        print("[nli] Loaded.", flush=True)
    return model


def load_l0(model: str, dataset: str, condition: str) -> Dict[int, Dict]:
    fp = _RESULTS / f"{model}_{condition}" / dataset / f"caveman_{model}_{dataset}_{condition}_L0.jsonl"
    if not fp.exists():
        return {}
    rows: Dict[int, Dict] = {}
    with fp.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows[int(r["idx"])] = r
    return rows


def parse_rate(rows: Dict[int, Dict]) -> Tuple[int, int, float]:
    """Fraction of rows with a non-empty predicted_answer (matches extraction_audit.py)."""
    n = len(rows)
    parsed = sum(
        1 for r in rows.values()
        if r.get("predicted_answer") is not None and str(r.get("predicted_answer")).strip() != ""
    )
    return n, parsed, (parsed / n if n else 0.0)


def score_cell(nli, model: str, dataset: str, batch_size: int) -> Dict:
    a = load_l0(model, dataset, "input")   # Condition A
    b = load_l0(model, dataset, "output")  # Condition B
    common = sorted(set(a.keys()) & set(b.keys()))

    a_n, a_parsed, a_rate = parse_rate(a)
    included = a_rate >= PARSE_GATE

    if not common:
        return {"model": model, "dataset": dataset, "n_pairs": 0, "skipped": True,
                "A_parse_rate": a_rate, "included": False}

    pairs: List[Tuple[str, str]] = []
    meta: List[Tuple[int, bool, bool]] = []  # idx, correct_A, correct_B
    for idx in common:
        ra, rb = a[idx], b[idx]
        oa = (ra.get("output") or "").strip()
        ob = (rb.get("output") or "").strip()
        # entail_AtoB: premise=A, hypothesis=B ; entail_BtoA: premise=B, hypothesis=A
        pairs.append((oa, ob))
        pairs.append((ob, oa))
        meta.append((idx, bool(ra.get("correct")), bool(rb.get("correct"))))

    t0 = time.time()
    results = batch_score_entailment(nli, pairs, batch_size=batch_size)
    elapsed = time.time() - t0

    rows_out = []
    n_bidir = n_correct_both = n_correct_both_bidir = n_either = 0
    for i, (idx, ca, cb) in enumerate(meta):
        a_to_b = results[2 * i]
        b_to_a = results[2 * i + 1]
        e_ab = a_to_b["label"] == "entailment"
        e_ba = b_to_a["label"] == "entailment"
        bidir = e_ab and e_ba
        n_bidir += int(bidir)
        if ca and cb:
            n_correct_both += 1
            n_correct_both_bidir += int(bidir)
        if ca or cb:
            n_either += 1
        rows_out.append({
            "idx": idx, "correct_A": ca, "correct_B": cb,
            "entail_AtoB": bool(e_ab), "entail_BtoA": bool(e_ba),
            "bidirectional_pass": bool(bidir),
            "A_to_B": a_to_b, "B_to_A": b_to_a,
        })

    with (OUT_DIR / f"{model}__{dataset}__pairs.jsonl").open("w", encoding="utf-8") as fout:
        for r in rows_out:
            fout.write(json.dumps(r) + "\n")

    n = len(meta)
    summary = {
        "model": model, "dataset": dataset, "reasoning": model in REASONING,
        "included": included, "A_parse_rate": a_rate, "A_n": a_n, "A_parsed": a_parsed,
        "n_pairs": n,
        "n_bidir": n_bidir,
        "non_entailment_rate": 1.0 - n_bidir / n if n else None,
        "n_correct_both": n_correct_both,
        "non_entailment_rate_correct_both": (1.0 - n_correct_both_bidir / n_correct_both)
        if n_correct_both else None,
        "n_correct_both_bidir": n_correct_both_bidir,
        "elapsed_s": elapsed,
    }
    (OUT_DIR / f"{model}__{dataset}__summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    flag = "INCL" if included else f"EXCL(A_parse={a_rate:.3f})"
    print(f"[{model}/{dataset}] {flag} n={n} "
          f"floor={summary['non_entailment_rate']:.4f} "
          f"floor(corr-both)={summary['non_entailment_rate_correct_both'] if n_correct_both else float('nan'):.4f} "
          f"elapsed={elapsed:.1f}s", flush=True)
    return {**summary, "_rows": rows_out}


def rollup(cells: List[Dict], models: List[str], gated: bool = False) -> Dict:
    """Aggregate cells over the given model set (pooled over items).

    gated=False : all cells (matches the un-gated 52% C2 headline denominator).
    gated=True  : only cells with L0-A parse rate >= PARSE_GATE (robustness check).
    """
    sel = [c for c in cells if c["model"] in models and c.get("n_pairs")
           and (c.get("included") if gated else True)]
    n = sum(c["n_pairs"] for c in sel)
    nb = sum(c["n_bidir"] for c in sel)
    ncb = sum(c["n_correct_both"] for c in sel)
    ncbb = sum(c["n_correct_both_bidir"] for c in sel)
    return {
        "models": models, "gated": gated, "n_cells": len(sel), "n_pairs": n,
        "non_entailment_rate": (1.0 - nb / n) if n else None,
        "n_correct_both": ncb,
        "non_entailment_rate_correct_both": (1.0 - ncbb / ncb) if ncb else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default=None,
                       help="Path to the results tree "
                            "(default: <repo>/results, or $CAVEWOMAN_RESULTS).")
    ap.add_argument("--models", nargs="+", default=PANEL_8)
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    nli = load_nli_cpu()

    cells: List[Dict] = []
    for m in args.models:
        for d in args.datasets:
            try:
                cells.append(score_cell(nli, m, d, args.batch_size))
            except Exception as e:
                print(f"[ERROR] {m}/{d}: {type(e).__name__}: {e}", flush=True)

    # Per-pair CSV (across all scored cells; includes A-parse flag for filtering).
    with (OUT_DIR / "pairs_all.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "dataset", "item", "entail_AtoB", "entail_BtoA",
                    "bidirectional_pass", "correct_A", "correct_B", "cell_included"])
        for c in cells:
            for r in c.get("_rows", []):
                w.writerow([c["model"], c["dataset"], r["idx"], int(r["entail_AtoB"]),
                            int(r["entail_BtoA"]), int(r["bidirectional_pass"]),
                            int(r["correct_A"]), int(r["correct_B"]), int(c["included"])])

    # Strip heavy rows before summarising / serialising.
    lean = [{k: v for k, v in c.items() if k != "_rows"} for c in cells]

    # Primary = un-gated (matches the 52% headline, which applies no parse gate).
    # Gated = parse>=0.95 robustness check (collapses mostly to GSM8K; see notes).
    headline = rollup(cells, NON_REASONING_6, gated=False)
    headline_gated = rollup(cells, NON_REASONING_6, gated=True)
    panel = rollup(cells, PANEL_8, gated=False)
    panel_gated = rollup(cells, PANEL_8, gated=True)
    per_model = {m: rollup(cells, [m], gated=False) for m in PANEL_8}
    per_dataset = {
        d: rollup([c for c in cells if c["dataset"] == d], NON_REASONING_6, gated=False)
        for d in DATASETS
    }
    excluded = [{"model": c["model"], "dataset": c["dataset"], "A_parse_rate": c["A_parse_rate"]}
                for c in lean if not c.get("included")]

    grand = {
        "judge": NLI_MODEL_NAME, "device": "cpu", "parse_gate": PARSE_GATE,
        "panel_8": PANEL_8, "non_reasoning_6": NON_REASONING_6,
        "note": ("Low L0-A parse rates are extraction-format misses (e.g. 'The answer "
                 "is **A**'), not garbage text; the un-gated rate is the headline-"
                 "comparable floor. Gated rate is a GSM8K-dominated robustness check."),
        "headline_6_non_reasoning_ungated": headline,
        "headline_6_non_reasoning_gated": headline_gated,
        "aggregate_8_model_ungated": panel,
        "aggregate_8_model_gated": panel_gated,
        "per_model_ungated": per_model,
        "per_dataset_6_non_reasoning_ungated": per_dataset,
        "excluded_low_parse_cells": excluded,
        "n_excluded_cells": len(excluded),
        "per_cell": lean,
    }
    (OUT_DIR / "_grand_summary.json").write_text(json.dumps(grand, indent=2), encoding="utf-8")

    # Summary CSV: cells, then per-model, per-dataset, and panel rollups.
    with (OUT_DIR / "summary_table.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scope", "key", "n_pairs", "non_entailment_rate",
                    "non_entailment_rate_correct_both", "n_correct_both", "included/A_parse"])
        for c in lean:
            w.writerow(["cell", f"{c['model']}/{c['dataset']}", c.get("n_pairs"),
                        c.get("non_entailment_rate"), c.get("non_entailment_rate_correct_both"),
                        c.get("n_correct_both"),
                        "INCL" if c.get("included") else f"EXCL({c.get('A_parse_rate'):.3f})"])
        for m, r in per_model.items():
            w.writerow(["per_model", m, r["n_pairs"], r["non_entailment_rate"],
                        r["non_entailment_rate_correct_both"], r["n_correct_both"], ""])
        for d, r in per_dataset.items():
            w.writerow(["per_dataset(6nr)", d, r["n_pairs"], r["non_entailment_rate"],
                        r["non_entailment_rate_correct_both"], r["n_correct_both"], ""])
        for name, r in [("6_non_reasoning_ungated", headline),
                        ("6_non_reasoning_gated", headline_gated),
                        ("8_model_ungated", panel),
                        ("8_model_gated", panel_gated)]:
            w.writerow(["panel", name, r["n_pairs"], r["non_entailment_rate"],
                        r["non_entailment_rate_correct_both"], r["n_correct_both"], ""])

    def line(label, r):
        cb = r["non_entailment_rate_correct_both"]
        return (f"{label:30s} floor={r['non_entailment_rate']:.4f}  "
                f"floor(correct-both)={cb:.4f}  n_pairs={r['n_pairs']} cells={r['n_cells']}"
                if r["non_entailment_rate"] is not None else f"{label:30s} (no pairs)")

    print("\n=== L0-A vs L0-B NOISE FLOOR ===", flush=True)
    print(f"low-parse cells flagged (A_parse<{PARSE_GATE}): {len(excluded)} of 40", flush=True)
    print("PRIMARY (un-gated, matches 52% headline denominator):", flush=True)
    print("  " + line("6 non-reasoning (headline)", headline), flush=True)
    print("  " + line("8-model aggregate", panel), flush=True)
    print("ROBUSTNESS (parse>=0.95, GSM8K-dominated):", flush=True)
    print("  " + line("6 non-reasoning", headline_gated), flush=True)
    print("  " + line("8-model aggregate", panel_gated), flush=True)


if __name__ == "__main__":
    main()
