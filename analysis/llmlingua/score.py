"""Step 4, bidirectional NLI + cosine vs L0 baseline for LLMLingua outputs (single level).

For each row, the "Lx" output is the LLMLingua-compressed-prompt generation, and
the "L0" output is the existing Condition-A L0 generation from the same
(model, dataset, idx).

Writes:
    results/llmlingua/inference/<model>/<dataset>/<dataset>_LLMLingua_scored.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from entailment_scorer import load_nli_model, batch_score_entailment  # noqa: E402

INF_BASE = REPO / "results" / "llmlingua" / "inference"
BASELINE_BASE = REPO / "results"

MODELS = ["gpt-4o", "sonnet-4.6", "qwen-2.5"]
DATASETS = ["gsm8k", "boolq", "arc_easy"]
# Levels we score. Primary LLMLingua run is τ=0.5 across all 3 models.
# Rate-sensitivity run (LLMLingua_t0.8) only exists for qwen-2.5, score_one
# silently skips (model, dataset, level) cells whose inference file is missing.
LEVELS = ["LLMLingua", "LLMLingua_t0.8"]

NLI_BATCH_SIZE = 32


def load_baseline_l0(model_tag: str, dataset: str) -> Dict[int, str]:
    """Return {idx: l0_output_text} from the matching Condition-A L0 JSONL."""
    base_dir = BASELINE_BASE / f"{model_tag}_input" / dataset
    candidate = base_dir / f"caveman_{model_tag}_{dataset}_input_L0_with_embeddings.jsonl"
    if not candidate.exists():
        candidate = base_dir / f"caveman_{model_tag}_{dataset}_input_L0.jsonl"
    if not candidate.exists():
        # Sonnet's baseline may not exist yet (paper run didn't include Sonnet);
        # caller skips this cell.
        return {}
    out = {}
    with candidate.open() as f:
        for line in f:
            rec = json.loads(line)
            out[int(rec["idx"])] = rec.get("output", "") or ""
    return out


def load_inference(model_tag: str, dataset: str, level: str) -> List[Dict]:
    p = INF_BASE / model_tag / dataset / f"{dataset}_{level}.jsonl"
    rows = []
    if not p.exists():
        return rows
    with p.open() as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def compute_cosine(cosine_model, lx_texts, l0_texts, batch_size=256):
    lx_emb = cosine_model.encode(lx_texts, batch_size=batch_size,
                                  normalize_embeddings=True, show_progress_bar=False)
    l0_emb = cosine_model.encode(l0_texts, batch_size=batch_size,
                                  normalize_embeddings=True, show_progress_bar=False)
    return (np.asarray(lx_emb) * np.asarray(l0_emb)).sum(axis=1).tolist()


def score_one(model_tag: str, dataset: str, level: str, nli, cosine_model) -> None:
    rows = load_inference(model_tag, dataset, level)
    if not rows:
        print(f"[{model_tag} {dataset} {level}] no inference rows, skipping", flush=True)
        return
    l0_map = load_baseline_l0(model_tag, dataset)
    if not l0_map:
        print(f"[{model_tag} {dataset} {level}] no L0 baseline found, skipping",
              flush=True)
        return

    paired = []
    skipped = 0
    for r in rows:
        idx = int(r["idx"])
        lx = r.get("output", "") or ""
        l0 = l0_map.get(idx, "")
        if not lx or not l0:
            skipped += 1
            continue
        paired.append((r, lx, l0))
    print(f"[{model_tag} {dataset} {level}] paired={len(paired)} skipped={skipped}", flush=True)
    if not paired:
        return

    lx_texts = [t[1] for t in paired]
    l0_texts = [t[2] for t in paired]

    t0 = time.time()
    cosine_scores = compute_cosine(cosine_model, lx_texts, l0_texts)
    print(f"  cosine done in {time.time()-t0:.1f}s", flush=True)

    fwd_pairs = list(zip(lx_texts, l0_texts))
    bwd_pairs = list(zip(l0_texts, lx_texts))

    t1 = time.time()
    fwd_out = batch_score_entailment(nli, fwd_pairs, batch_size=NLI_BATCH_SIZE)
    bwd_out = batch_score_entailment(nli, bwd_pairs, batch_size=NLI_BATCH_SIZE)
    print(f"  NLI both directions done in {time.time()-t1:.1f}s", flush=True)

    out_path = INF_BASE / model_tag / dataset / f"{dataset}_{level}_scored.jsonl"
    with out_path.open("w") as f:
        for (r, _, l0), cos_s, fwd, bwd in zip(paired, cosine_scores, fwd_out, bwd_out):
            new_rec = dict(r)
            new_rec["l0_output"] = l0
            new_rec["cosine_similarity_l0"] = round(float(cos_s), 6)
            new_rec["lx_entails_l0_label"] = fwd["label"]
            new_rec["lx_entails_l0_prob"] = round(float(fwd["entailment_prob"]), 6)
            new_rec["l0_entails_lx_label"] = bwd["label"]
            new_rec["l0_entails_lx_prob"] = round(float(bwd["entailment_prob"]), 6)
            new_rec["bidirectional_entailment"] = bool(
                fwd["label"] == "entailment" and bwd["label"] == "entailment"
            )
            f.write(json.dumps(new_rec, ensure_ascii=False) + "\n")
    print(f"  wrote {out_path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--levels", nargs="+", default=LEVELS)
    args = ap.parse_args()

    print("Loading NLI model ...", flush=True)
    nli = load_nli_model()
    print("Loading cosine model ...", flush=True)
    from sentence_transformers import SentenceTransformer
    cosine_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    for mt in args.models:
        for ds in args.datasets:
            for lv in args.levels:
                score_one(mt, ds, lv, nli, cosine_model)


if __name__ == "__main__":
    main()
