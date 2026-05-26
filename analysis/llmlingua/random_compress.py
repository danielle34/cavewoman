"""Step 2 (random-deletion baseline), drop tokens uniformly at random at target τ.

Companion to analysis/llmlingua/compress.py. Same input/output shape, same
downstream pipeline, but the compression decision is uniform-random deletion
instead of LLMLingua-2's learned BERT classifier. This is the "any compression
produces the dissociation?" baseline: if Random produces similar C2 cells to
LLMLingua at matched τ, then the dissociation is a generic property of token
deletion and not specific to information-aware compression methods. If Random
produces meaningfully different cells, the choice of which tokens to drop
matters.

Implementation:
  , Tokenize on whitespace (rough word-level).
  , For target τ, keep round(τ * N) tokens chosen uniformly without replacement.
  , Use a deterministic per-(dataset, idx, tau) seed so this is reproducible.
  , Preserve original token order in the reconstructed prompt.

This runs locally in <1 min, no model loading, no GPU. Output schema matches
LLMLingua compressed JSONL so the same inference / scoring code works on it.

Outputs:
    results/random/compressed/<ds>/<ds>_random_compressed_t0.5.jsonl
    results/random/compressed/<ds>/<ds>_random_compressed_t0.8.jsonl
    (level field: "Random_t0.5" or "Random_t0.8")

Usage:
    python analysis/llmlingua/random_compress.py
    # or limit to one dataset / tau:
    python analysis/llmlingua/random_compress.py --datasets gsm8k --taus 0.5
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from dataset_loader_multi import load_dataset_caveman  # noqa: E402

OUT_BASE = REPO / "results" / "random" / "compressed"
DATASETS = ["gsm8k", "boolq", "arc_easy", "commonsenseqa", "mmlu_stem"]
TAUS = [0.5, 0.8]


def _seed_for(dataset: str, idx: int, tau: float) -> int:
    """Stable seed per cell so this is reproducible end-to-end."""
    key = f"{dataset}|{idx}|{tau:.3f}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big")


def _random_drop(text: str, tau: float, rng: random.Random) -> tuple[str, int, int]:
    """Keep round(tau * N) whitespace-tokens chosen uniformly without replacement, original order."""
    tokens = text.split()
    n = len(tokens)
    if n == 0:
        return text, 0, 0
    keep = max(1, round(tau * n))
    keep = min(keep, n)
    keep_idxs = sorted(rng.sample(range(n), keep))
    kept = [tokens[i] for i in keep_idxs]
    return " ".join(kept), n, keep


def process(dataset: str, tau: float, limit: Optional[int]) -> Path:
    items = load_dataset_caveman(dataset)
    if limit:
        items = items[:limit]
    tau_tag = f"t{tau:g}"   # 0.5 -> "t0.5",  0.8 -> "t0.8"
    level = f"Random_{tau_tag}"
    out_dir = OUT_BASE / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{dataset}_random_compressed_{tau_tag}.jsonl"
    n_written = 0
    n_skipped = 0
    with out_path.open("w") as f:
        for it in items:
            idx = it["idx"]
            qtext = it["question_formatted"]
            gold = it["answer_gt"]
            atype = it["answer_type"]
            if not qtext or not qtext.strip():
                n_skipped += 1
                continue
            rng = random.Random(_seed_for(dataset, idx, tau))
            compressed, orig_tok, comp_tok = _random_drop(qtext, tau, rng)
            actual_tau = (comp_tok / orig_tok) if orig_tok > 0 else 1.0
            rec = {
                "dataset": dataset,
                "idx": idx,
                "level": level,
                "answer_type": atype,
                "gold_answer": gold,
                "target_tau": tau,
                "actual_tau": round(actual_tau, 4),
                "original_tokens": orig_tok,
                "compressed_tokens": comp_tok,
                "original_prompt": qtext,
                "compressed_prompt": compressed,
                "error": None,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_written += 1
    print(f"[{dataset} τ={tau}] wrote {n_written} rows ({n_skipped} skipped) -> {out_path}",
          flush=True)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--taus", nargs="+", type=float, default=TAUS)
    ap.add_argument("--limit", type=int, default=None, help="Debug only")
    args = ap.parse_args()
    print(f"Random-deletion compression: datasets={args.datasets} taus={args.taus}")
    for ds in args.datasets:
        for tau in args.taus:
            process(ds, tau, args.limit)


if __name__ == "__main__":
    main()
