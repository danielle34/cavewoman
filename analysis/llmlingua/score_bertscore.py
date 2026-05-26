"""Score BERTScore on LLMLingua and Cavewoman matched-comparison outputs.

Matches the paper's BERTScore protocol exactly:
- model: roberta-large
- 300-item subsample per (model x dataset x method) cell (CPU OOM bound)
- random seed fixed across methods for parity

Writes:
  results/llmlingua/bertscore_per_cell.csv     (one row per cell)
  results/llmlingua/bertscore_summary.csv      (method-level aggregate)
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd
from bert_score import score as bertscore

SEED = 0
SUBSAMPLE = 300
MODEL_TYPE = "roberta-large"
NUM_LAYERS = 17                       # default for roberta-large in bert_score

RESULTS = Path("../../results")
LLM_DIR = RESULTS / "llmlingua" / "inference"
LL_MODELS = ["gpt-4o", "sonnet-4.6", "qwen-2.5"]
LL_DATASETS = ["gsm8k", "boolq", "arc_easy"]

OUT_PER_CELL = RESULTS / "llmlingua" / "bertscore_per_cell.csv"
OUT_SUMMARY  = RESULTS / "llmlingua" / "bertscore_summary.csv"


def load_pairs_from_scored(path: Path) -> list[tuple[str, str]]:
    """Return list of (l0_output, lx_output) pairs from a scored JSONL.

    Skips rows with missing fields or empty strings. Used for LLMLingua
    scored files (τ=0.5 and τ=0.8)."""
    pairs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            l0 = (rec.get("l0_output") or "").strip()
            lx = (rec.get("output") or "").strip()
            if not l0 or not lx:
                continue
            pairs.append((l0, lx))
    return pairs


def load_caveman_pairs(model: str, dataset: str, level: str) -> list[tuple[str, str]]:
    """Cavewoman L1/L2 input-channel pairs. We pull L0 outputs and Lx outputs
    from the per-item entailment JSONLs (which include both side-by-side)."""
    run_dir = RESULTS / f"{model}_input" / dataset
    lx_path = run_dir / f"caveman_{model}_{dataset}_input_{level}_with_entailment.jsonl"
    l0_path = run_dir / f"caveman_{model}_{dataset}_input_L0_with_embeddings.jsonl"
    if not (lx_path.exists() and l0_path.exists()):
        return []
    # Index L0 by item id.
    l0_idx = {}
    with open(l0_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            l0_idx[r.get("idx", r.get("id"))] = (r.get("output") or "").strip()
    pairs = []
    with open(lx_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            i = r.get("idx", r.get("id"))
            l0 = l0_idx.get(i, "")
            lx = (r.get("output") or "").strip()
            if not l0 or not lx:
                continue
            pairs.append((l0, lx))
    return pairs


def subsample(pairs: list[tuple[str, str]], n: int, seed: int) -> list[tuple[str, str]]:
    if len(pairs) <= n:
        return pairs
    rnd = random.Random(seed)
    return rnd.sample(pairs, n)


def score_pairs(pairs: list[tuple[str, str]]) -> float:
    if not pairs:
        return float("nan")
    refs = [p[0] for p in pairs]
    cands = [p[1] for p in pairs]
    P, R, F1 = bertscore(cands, refs,
                         model_type=MODEL_TYPE,
                         num_layers=NUM_LAYERS,
                         device="cpu", batch_size=8, verbose=False)
    return float(F1.mean().item())


def main():
    random.seed(SEED)
    rows = []

    print(f"BERTScore (model_type={MODEL_TYPE}) on CPU, "
          f"subsample={SUBSAMPLE}/cell, seed={SEED}")

    # ---- LLMLingua τ=0.5: 3 models x 3 datasets
    for model in LL_MODELS:
        for ds in LL_DATASETS:
            p = LLM_DIR / model / ds / f"{ds}_LLMLingua_scored.jsonl"
            if not p.exists():
                print(f"  [skip] {p} missing")
                continue
            pairs = subsample(load_pairs_from_scored(p), SUBSAMPLE, SEED)
            print(f"  scoring LLMLingua τ=0.5 / {model} / {ds} … "
                  f"n={len(pairs)}")
            f1 = score_pairs(pairs)
            rows.append(dict(method="LLMLingua (τ=0.5)", model=model, dataset=ds,
                             n=len(pairs), bertscore_f1=f1))

    # ---- LLMLingua τ=0.8: Qwen only, 3 datasets
    for ds in LL_DATASETS:
        p = LLM_DIR / "qwen-2.5" / ds / f"{ds}_LLMLingua_t0.8_scored.jsonl"
        if not p.exists():
            print(f"  [skip] {p} missing")
            continue
        pairs = subsample(load_pairs_from_scored(p), SUBSAMPLE, SEED)
        print(f"  scoring LLMLingua τ=0.8 / qwen-2.5 / {ds} … n={len(pairs)}")
        f1 = score_pairs(pairs)
        rows.append(dict(method="LLMLingua (τ=0.8)", model="qwen-2.5", dataset=ds,
                         n=len(pairs), bertscore_f1=f1))

    # ---- Cavewoman L1 and L2 Cond A on the same 3x3
    for method, level in [("Cavewoman (POS filter)", "L1"),
                          ("Cavewoman (POS filter)", "L2")]:
        for model in LL_MODELS:
            for ds in LL_DATASETS:
                pairs = subsample(load_caveman_pairs(model, ds, level),
                                  SUBSAMPLE, SEED)
                if not pairs:
                    print(f"  [skip] caveman {model}/{ds}/{level}")
                    continue
                print(f"  scoring Cavewoman {level}-A / {model} / {ds} … "
                      f"n={len(pairs)}")
                f1 = score_pairs(pairs)
                rows.append(dict(method=method, model=model, dataset=ds,
                                 n=len(pairs), bertscore_f1=f1,
                                 level=level))

    df = pd.DataFrame(rows)
    df.to_csv(OUT_PER_CELL, index=False)
    print(f"[wrote] {OUT_PER_CELL}  ({len(df)} rows)")

    summary = (df.groupby([c for c in ["method", "level"] if c in df.columns])
                 ["bertscore_f1"].mean().reset_index())
    summary.to_csv(OUT_SUMMARY, index=False)
    print(f"[wrote] {OUT_SUMMARY}")
    print(summary)


if __name__ == "__main__":
    main()
