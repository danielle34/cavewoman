"""Long-form summarization runner: XSum / CNN-DailyMail, Condition B, local models.

The model summarizes each article under the L0-L4 register ladder. Only the
output channel applies here, since the article is the input.

Outputs (resume-safe, appends per cell):
  results/longform/<tag>_output/<dataset>/caveman_<tag>_<dataset>_output_<Lx>.jsonl

Usage:
  python experiments/longform/run_summarisation.py \
      --model_name qwen2.5 --model_path ~/models/Qwen2.5-VL-7B \
      --dataset xsum --levels L0,L1,L2,L3,L4 --n 200

  # second draw for the sampling floor
  python experiments/longform/run_summarisation.py ... \
      --levels L0 --temperature 0.7 --suffix _paired

Add --disable_thinking for Qwen3-family models.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
REPO = _REPO_ROOT
sys.path.insert(0, str(_REPO_ROOT / "src"))

DOC_CHAR_CAP = 6000  # articles longer than this are truncated; flagged per row
# The main panel's {400,300,200,150,20} ladder halved, except L4, where the
# 15-token target fixes the cap at 20.
MAX_NEW_TOKENS = {"L0": 200, "L1": 150, "L2": 100, "L3": 75, "L4": 20}

PROMPTS = {
    "L0": (
        "Summarize the article below in one to three full, grammatical "
        "sentences. Write fluent English prose."
    ),
    "L1": (
        "Summarize the article below under a TELEGRAPHIC constraint.\n\n"
        "Rules:\n"
        "  - DO NOT use any function words. No articles (the, a, an). "
        "No conjunctions (and, but, or, so). No prepositions (of, in, to, "
        "for, at, with, from, by, on, per).\n"
        "  - DO use nouns, main verbs, and numbers.\n"
        "  - One to three telegraphic lines."
    ),
    "L2": (
        "Summarize the article below under a KEYWORD-ONLY constraint.\n\n"
        "Rules:\n"
        "  - Use ONLY nouns and main verbs. No grammar, no full sentences.\n"
        "  - Output fragments, short labels, or list items.\n"
        "  - Numbers are allowed."
    ),
    # L3/L4 reword the short-answer Condition-B prompts for summarization.
    "L3": (
        "Summarize the article below under a NOUN-PHRASE SKELETON constraint.\n\n"
        "Rules:\n"
        "  - NO verbs of any kind. None.\n"
        "  - Use only nominal fragments: nouns, noun compounds, and numbers.\n"
        "  - Each line is a noun phrase naming an entity, quantity, or claim."
    ),
    "L4": (
        "Summarize the article below under a HARD TOKEN BUDGET.\n\n"
        "Rules:\n"
        "  - Your ENTIRE response must be 15 tokens or fewer.\n"
        "  - Prefer the raw content over prose."
    ),
}


def completed_idx(path: Path) -> set:
    done = set()
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["idx"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--levels", default="L0,L1,L2")
    ap.add_argument("--dataset", default="xsum", choices=["xsum","cnndm"])
    ap.add_argument("--output_root", default=str(REPO / "results" / "longform"))
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="Decoding temperature (0.7 for paired-L0 noise-floor draws).")
    ap.add_argument("--suffix", default="",
                    help="Filename suffix, e.g. _paired for the T>0 second draw.")
    ap.add_argument("--disable_thinking", action="store_true",
                    help="Suppress the chat template's reasoning block. Use this "
                         "on Qwen3-family models, or the reasoning trace eats the "
                         "summary budget.")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    from datasets import load_dataset
    if args.dataset == "cnndm":
        ds = load_dataset("abisee/cnn_dailymail", "3.0.0", split="test")
        DOC_KEY, SUM_KEY = "article", "highlights"
    else:
        ds = load_dataset("EdinburghNLP/xsum", split="test", trust_remote_code=True)
        DOC_KEY, SUM_KEY = "document", "summary"
    rng = random.Random(args.seed)
    sample = sorted(rng.sample(range(len(ds)), args.n))

    tokenizer = model = None
    if not args.dry_run:
        from model_loader import load_model, run_inference
        print(f"[load] {args.model_path}", flush=True)
        tokenizer, model = load_model(args.model_path, model_type="auto")

    tag = args.model_name
    out_dir = Path(args.output_root) / f"{tag}_output" / args.dataset
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    levels = [x.strip() for x in args.levels.split(",") if x.strip()]
    for level in levels:
        out_path = out_dir / f"caveman_{tag}_{args.dataset}_output_{level}{args.suffix}.jsonl"
        done = completed_idx(out_path)
        sp = PROMPTS[level]
        max_new = MAX_NEW_TOKENS[level]
        if not args.dry_run:
            (out_dir / f"run_config_{level}.json").write_text(json.dumps({
                "model_name": tag, "model_path": args.model_path,
                "dataset": args.dataset, "condition": "output", "level": level,
                "n": args.n, "seed": args.seed, "doc_char_cap": DOC_CHAR_CAP,
                "system_prompt": sp, "max_new_tokens": max_new,
                "temperature": args.temperature,
                "written_at": datetime.now(timezone.utc).isoformat(),
            }, indent=2), encoding="utf-8")

        n_new = 0
        t0 = time.time()
        import contextlib, io
        _sink = io.StringIO() if args.dry_run else open(out_path, "a", encoding="utf-8")
        with contextlib.closing(_sink) as fout:
            for idx in sample:
                if idx in done:
                    continue
                item = ds[idx]
                doc = item[DOC_KEY][:DOC_CHAR_CAP]
                if args.dry_run:
                    result = {"output": f"[dry {level}/{idx}]",
                              "input_tokens": 0, "output_tokens": 0,
                              "latency_s": 0.0}
                else:
                    result = run_inference(
                        tokenizer=tokenizer, model=model,
                        system_prompt=sp, user_message=doc,
                        max_new_tokens=max_new, temperature=args.temperature,
                        enable_thinking=(False if args.disable_thinking else None))
                rec = {
                    "idx": idx, "level": level, "condition": "output",
                    "model": tag, "dataset": args.dataset,
                    "gold_summary": item[SUM_KEY],
                    "doc_chars": len(item[DOC_KEY]),
                    "doc_truncated": len(item[DOC_KEY]) > DOC_CHAR_CAP,
                    "output": result["output"],
                    "input_tokens": result["input_tokens"],
                    "output_tokens": result["output_tokens"],
                    "latency_s": result["latency_s"],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                n_new += 1
                if n_new % 50 == 0:
                    el = time.time() - t0
                    print(f"[{tag} xsum {level}] {n_new} done "
                          f"({el/max(n_new,1):.1f}s/item)", flush=True)
        print(f"[cell {tag}/xsum/{level}] complete, new={n_new}", flush=True)
    print("[xsum pilot] ALL DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
