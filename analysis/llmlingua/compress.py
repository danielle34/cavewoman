"""Step 2, compress GSM8K + BoolQ + ARC-Easy prompts with LLMLingua-2.

Design: LLMLingua is its own SINGLE compression level (named "LLMLingua"),
not split into matched L1/L2 ratios. The compressed prompts are what we
feed to the downstream LLMs as if LLMLingua were just another constraint
level peer to caveman's L0..L4.

Notes on the rate parameter (verified against the LLMLingua-2 paper, Sec. 4.2):
    `rate` passed to PromptCompressor.compress_prompt is τ = fraction KEPT
    (NOT fraction removed). τ=0.5 ⇒ keep 50%. Smaller τ ⇒ more aggressive
    compression.

We use τ = 0.5, LLMLingua-2's published default rate (the value used in
many of their reported benchmarks). This is the "what does LLMLingua do
out of the box" setting.

Outputs:
    results/llmlingua/compressed/<dataset>/<dataset>_compressed.jsonl
        one row per (dataset, idx) with:
            idx, dataset, level=LLMLingua, target_tau, actual_tau,
            original_tokens, compressed_tokens,
            original_prompt, compressed_prompt,
            gold_answer, answer_type
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from dataset_loader_multi import load_dataset_caveman  # noqa: E402

OUT_BASE = REPO / "results" / "llmlingua" / "compressed"

DATASETS = ["gsm8k", "boolq", "arc_easy"]
DEFAULT_LEVEL_NAME = "LLMLingua"
DEFAULT_TAU = 0.5


def _level_and_filename(tau_tag: Optional[str]) -> tuple[str, str]:
    """Resolve (level_name, output_filename) from optional tau_tag.
    tau_tag=None -> uses default ("LLMLingua", "<ds>_compressed.jsonl")
    tau_tag='t0.8' -> ("LLMLingua_t0.8", "<ds>_compressed_t0.8.jsonl")
    """
    if not tau_tag:
        return DEFAULT_LEVEL_NAME, "{ds}_compressed.jsonl"
    return f"{DEFAULT_LEVEL_NAME}_{tau_tag}", "{ds}_compressed_" + tau_tag + ".jsonl"

# Graceful shutdown on SIGTERM / SIGUSR1 (SLURM walltime trap).
_STOP = {"flag": False}


def _stop_handler(signum, frame):
    print(f"[compress] caught signal {signum}; finishing current item then exiting cleanly",
          flush=True)
    _STOP["flag"] = True


signal.signal(signal.SIGTERM, _stop_handler)
signal.signal(signal.SIGUSR1, _stop_handler)


def load_compressor():
    from llmlingua import PromptCompressor

    print("Loading LLMLingua-2 (microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank)...",
          flush=True)
    return PromptCompressor(
        model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
        use_llmlingua2=True,
        device_map="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
    )


def read_existing_ids(path: Path) -> set:
    if not path.exists():
        return set()
    done = set()
    with path.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
                done.add(rec["idx"])
            except Exception:
                continue
    return done


def compress_one(compressor, text: str, target_tau: float) -> Dict:
    out = compressor.compress_prompt(
        text,
        rate=target_tau,
        force_tokens=["\n", "?", ".", ":"],
        drop_consecutive=True,
    )
    return out


def process_dataset(dataset: str, compressor, target_tau: float,
                     limit: Optional[int], tau_tag: Optional[str]) -> None:
    items = load_dataset_caveman(dataset)
    if limit:
        items = items[:limit]
    level_name, filename_tmpl = _level_and_filename(tau_tag)
    out_path = OUT_BASE / dataset / filename_tmpl.format(ds=dataset)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    already = read_existing_ids(out_path)
    print(f"[{dataset}] {len(items)} items, {len(already)} already written  "
          f"(level={level_name}, file={out_path.name})",
          flush=True)

    t_start = time.time()
    written = 0
    with out_path.open("a") as f:
        for it in items:
            if _STOP["flag"]:
                print(f"[{dataset}] stopping early due to signal", flush=True)
                break
            idx = it["idx"]
            qtext = it["question_formatted"]
            gold = it["answer_gt"]
            atype = it["answer_type"]
            if not qtext or not qtext.strip():
                continue
            if idx in already:
                continue
            try:
                out = compress_one(compressor, qtext, target_tau)
                compressed = out["compressed_prompt"]
                orig_tok = int(out["origin_tokens"])
                comp_tok = int(out["compressed_tokens"])
                actual_tau = comp_tok / orig_tok if orig_tok > 0 else 1.0
                err = None
            except Exception as e:
                compressed = qtext
                orig_tok = len(qtext.split())
                comp_tok = orig_tok
                actual_tau = 1.0
                err = f"{type(e).__name__}: {e}"
                print(f"  [error] {dataset} idx={idx}: {err}", flush=True)
            rec = {
                "dataset": dataset,
                "idx": idx,
                "level": level_name,
                "answer_type": atype,
                "gold_answer": gold,
                "target_tau": target_tau,
                "actual_tau": round(actual_tau, 4),
                "original_tokens": orig_tok,
                "compressed_tokens": comp_tok,
                "original_prompt": qtext,
                "compressed_prompt": compressed,
                "error": err,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            written += 1
            if written % 200 == 0:
                elapsed = time.time(), t_start
                rate = written / elapsed if elapsed > 0 else 0
                print(f"  [{dataset}] written={written}  rate={rate:.1f}/s  elapsed={elapsed:.0f}s",
                      flush=True)

    print(f"[{dataset}] done, {written} new rows -> {out_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU,
                        help="Fraction of tokens to KEEP (LLMLingua-2's rate parameter). Default 0.5 (paper default).")
    parser.add_argument("--tau_tag", type=str, default=None,
                        help="Optional suffix like 't0.8' to keep multi-rate runs from clobbering each other. "
                             "When set, outputs <ds>_compressed_<tag>.jsonl and level='LLMLingua_<tag>'.")
    args = parser.parse_args()

    print(f"Compression target: tau = {args.tau} (keep {args.tau*100:.0f}%)  tau_tag={args.tau_tag!r}")
    compressor = load_compressor()

    for ds in args.datasets:
        process_dataset(ds, compressor, args.tau, args.limit, args.tau_tag)


if __name__ == "__main__":
    main()
