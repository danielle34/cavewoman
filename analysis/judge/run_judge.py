"""LLM-as-judge runner for CAVEWOMAN.

Reads existing per-item JSONL records from `results/<model>_<cond>/<dataset>/`,
constructs judge prompts via analysis/judge/prompts.py, runs an
instruction-following LLM judge on a sampled subset, and emits one
judgement JSONL per (target_model, dataset, condition) cell.

Two modes:

  --mode pair      : judge (L0_trace, Lx_trace) pairs across L1, L2, L3, L4
                     (one judgement per pair per item).
  --mode recovery  : judge L0 traces where the original predicted_answer was
                     None, to recover signal from extraction failures.

Designed to run inside a SLURM job; loads the judge model ONCE and processes
all items for the cell. Resume-safe: skips items already in the output JSONL.

Outputs land at:
  <output_dir>/<target_model>_<dataset>_<condition>_<mode>.jsonl

CLI example:
  python analysis/judge/run_judge.py \\
      --mode pair --target_model qwen-2.5 \\
      --dataset gsm8k --condition output \\
      --n_samples 100 \\
      --judge_model_path <path-or-huggingface-id> \\
      --judge_model_tag <short-tag> \\
      --output_dir ./results/judge_runs
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# src/ on path so we can import model_loader at runtime, and the local
# directory so `from prompts import ...` resolves to the sibling file.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prompts import (  # noqa: E402
    PAIR_SYSTEM, PAIR_USER_TEMPLATE, PAIR_LABELS,
    RECOVERY_SYSTEM, RECOVERY_USER_TEMPLATE, RECOVERY_LABELS,
    extract_verdict,
)


# CLI

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", required=True, choices=["pair", "recovery"])
    p.add_argument("--target_model", required=True,
                   help="Tag of the model whose outputs we are judging (e.g. qwen-2.5, gpt-4o, sonnet-4.6).")
    p.add_argument("--dataset", required=True,
                   choices=["gsm8k", "boolq", "arc_easy", "commonsenseqa", "mmlu_stem"])
    p.add_argument("--condition", required=True, choices=["output", "input"])
    p.add_argument("--judge_model_path", required=True,
                   help="Local path to the judge model snapshot, or a HuggingFace ID.")
    p.add_argument("--judge_model_tag", required=True,
                   help="Short tag for the judge model, used in output filenames.")
    p.add_argument("--results_root", default="./results")
    p.add_argument("--output_dir", default=None,
                   help="Output directory. Defaults to ./results/judge_<judge_model_tag>/.")
    p.add_argument("--n_samples", type=int, default=100,
                   help="Per-cell sample size. Use mode determines what is sampled.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_new_tokens", type=int, default=800,
                   help="Judge generation budget. With a verdict-first prompt "
                        "the verdict is captured even on truncation, but 800 "
                        "tokens give room for full optional reasoning.")
    p.add_argument("--levels", default="L1,L2,L3,L4",
                   help="Comma-separated levels to judge in pair mode (against L0). "
                        "Ignored in recovery mode (only L0 is relevant).")
    p.add_argument("--dry_run", action="store_true",
                   help="Skip model load + judge calls; write placeholder verdicts. "
                        "Useful for testing the pipeline without GPU access.")
    args = p.parse_args()
    if args.output_dir is None:
        args.output_dir = f"./results/judge_{args.judge_model_tag}"
    return args


# Record discovery

def jsonl_path(results_root: str, target_model: str, condition: str,
               dataset: str, level: str) -> Path:
    """Path of the per-item JSONL for one (model, condition, dataset, level) cell.

    Mirrors the production naming convention. We prefer the `_with_entailment`
    variant if it exists (richest sidecar) and fall back to plain and embedding
    variants in that order.
    """
    base = Path(results_root) / f"{target_model}_{condition}" / dataset
    name = f"caveman_{target_model}_{dataset}_{condition}_{level}"
    for suffix in ("_with_entailment", "_with_embeddings", ""):
        p = base / f"{name}{suffix}.jsonl"
        if p.exists():
            return p
    return base / f"{name}.jsonl"  # may not exist; caller checks


def load_jsonl(path: Path) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def index_by_idx(records: list[dict]) -> dict[int, dict]:
    return {r["idx"]: r for r in records if "idx" in r}


# Sampling

def sample_indices(all_idx: list[int], n: int, seed: int) -> list[int]:
    if n >= len(all_idx):
        return sorted(all_idx)
    rng = random.Random(seed)
    return sorted(rng.sample(all_idx, n))


# Prompt building

def format_pair_user(question: str, trace_a: str, trace_b: str) -> str:
    return PAIR_USER_TEMPLATE.format(question=question, trace_a=trace_a, trace_b=trace_b)


def format_recovery_user(question: str, choices: str, gold_answer: str, trace: str) -> str:
    return RECOVERY_USER_TEMPLATE.format(
        question=question, choices=choices, gold_answer=gold_answer, trace=trace
    )


def parse_choices_for_recovery(record: dict) -> str:
    """Pull a 'A) ... B) ... C) ...' string from the record. Falls back to
    the `question_formatted` field if explicit choices aren't separable.
    """
    q = record.get("question_formatted") or record.get("question") or ""
    # The question_formatted field already includes the choices for MC
    # datasets, so we just return it verbatim.
    return q


# Resume helpers

def completed_keys(out_path: Path, key_fn) -> set:
    if not out_path.exists():
        return set()
    out = set()
    with open(out_path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            k = key_fn(rec)
            if k is not None:
                out.add(k)
    return out


# Main per-mode dispatch

def run_pair_mode(args, judge_runner) -> int:
    """Compare each (L0, Lx) pair for sampled items across requested levels.

    One judgement per (item, level) pair lands in the output JSONL with key
    (idx, level).
    """
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.target_model}_{args.dataset}_{args.condition}_pair.jsonl"

    # Load L0
    l0_path = jsonl_path(args.results_root, args.target_model, args.condition,
                          args.dataset, "L0")
    if not l0_path.exists():
        print(f"[fatal] L0 records missing: {l0_path}", file=sys.stderr)
        return 2
    l0 = index_by_idx(load_jsonl(l0_path))

    levels = [lv.strip() for lv in args.levels.split(",") if lv.strip()]
    print(f"[pair] L0 records: {len(l0)}  levels: {levels}", flush=True)

    # Sample items from those that exist in L0
    sample_idx = sample_indices(sorted(l0.keys()), args.n_samples, args.seed)
    print(f"[pair] sampled {len(sample_idx)} items", flush=True)

    # Completed (idx, level) keys for resume
    done_keys = completed_keys(out_path, key_fn=lambda r: (r.get("idx"), r.get("level")))
    print(f"[pair] resuming with {len(done_keys)} pairs already judged", flush=True)

    # Process each level
    total_processed = 0
    t_start = time.time()
    with open(out_path, "a") as fout:
        for level in levels:
            lx_path = jsonl_path(args.results_root, args.target_model, args.condition,
                                  args.dataset, level)
            if not lx_path.exists():
                print(f"[pair] {level}: file missing, skipping ({lx_path})", flush=True)
                continue
            lx = index_by_idx(load_jsonl(lx_path))

            for idx in sample_idx:
                if (idx, level) in done_keys:
                    continue
                if idx not in lx:
                    continue
                r0 = l0[idx]; rx = lx[idx]
                trace_a = (r0.get("output") or "").strip()
                trace_b = (rx.get("output") or "").strip()
                question = r0.get("question") or r0.get("question_formatted") or ""

                user_msg = format_pair_user(question, trace_a, trace_b)
                judgement = judge_runner(PAIR_SYSTEM, user_msg, args.max_new_tokens, args.dry_run)
                verdict = extract_verdict(judgement["output"], PAIR_LABELS)

                rec = {
                    "judge_mode": "pair",
                    "target_model": args.target_model,
                    "dataset": args.dataset,
                    "condition": args.condition,
                    "level": level,
                    "idx": int(idx),
                    "gt_answer": r0.get("gt_answer"),
                    "l0_predicted": r0.get("predicted_answer"),
                    "lx_predicted": rx.get("predicted_answer"),
                    "l0_correct": r0.get("correct"),
                    "lx_correct": rx.get("correct"),
                    "judge_verdict": verdict,
                    "judge_raw_output": judgement["output"],
                    "judge_input_tokens": judgement["input_tokens"],
                    "judge_output_tokens": judgement["output_tokens"],
                    "judge_latency_s": judgement["latency_s"],
                    "judge_model_tag": args.judge_model_tag,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                fout.write(json.dumps(rec) + "\n"); fout.flush()
                total_processed += 1
                if total_processed % 25 == 0:
                    elapsed = time.time(), t_start
                    print(f"[pair] {total_processed} judged  elapsed={elapsed:.0f}s", flush=True)
    print(f"[pair] DONE  {total_processed} new judgements -> {out_path}", flush=True)
    return 0


def run_recovery_mode(args, judge_runner) -> int:
    """Judge L0 traces where the original `predicted_answer` was None.

    Only L0 is judged (level constraint), only items where the regex failed.
    """
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.target_model}_{args.dataset}_{args.condition}_recovery.jsonl"

    l0_path = jsonl_path(args.results_root, args.target_model, args.condition,
                          args.dataset, "L0")
    if not l0_path.exists():
        print(f"[fatal] L0 records missing: {l0_path}", file=sys.stderr)
        return 2
    records = load_jsonl(l0_path)

    # Pick records where extraction failed
    none_records = [r for r in records if r.get("predicted_answer") is None]
    print(f"[recovery] L0 total: {len(records)}  predicted=None: {len(none_records)}", flush=True)
    if not none_records:
        print(f"[recovery] no None records, nothing to recover. Exit cleanly.", flush=True)
        return 0

    # Sample up to n_samples of them
    rng = random.Random(args.seed)
    if len(none_records) > args.n_samples:
        sampled = rng.sample(none_records, args.n_samples)
    else:
        sampled = none_records
    print(f"[recovery] sampled {len(sampled)} records to judge", flush=True)

    done_keys = completed_keys(out_path, key_fn=lambda r: r.get("idx"))
    print(f"[recovery] resuming with {len(done_keys)} items already judged", flush=True)

    total_processed = 0
    t_start = time.time()
    with open(out_path, "a") as fout:
        for r in sampled:
            idx = r["idx"]
            if idx in done_keys:
                continue
            trace = (r.get("output") or "").strip()
            question = r.get("question") or ""
            choices = parse_choices_for_recovery(r)
            gold = r.get("gt_answer") or ""

            user_msg = format_recovery_user(question, choices, str(gold), trace)
            judgement = judge_runner(RECOVERY_SYSTEM, user_msg, args.max_new_tokens, args.dry_run)
            verdict = extract_verdict(judgement["output"], RECOVERY_LABELS)

            rec = {
                "judge_mode": "recovery",
                "target_model": args.target_model,
                "dataset": args.dataset,
                "condition": args.condition,
                "level": "L0",
                "idx": int(idx),
                "gt_answer": gold,
                "l0_predicted_extracted": r.get("predicted_answer"),  # always None here
                "l0_correct_extracted": r.get("correct"),
                "judge_verdict": verdict,
                "judge_raw_output": judgement["output"],
                "judge_input_tokens": judgement["input_tokens"],
                "judge_output_tokens": judgement["output_tokens"],
                "judge_latency_s": judgement["latency_s"],
                "judge_model_tag": args.judge_model_tag,
                "trace_chars": len(trace),
                "trace_output_tokens": r.get("output_tokens"),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            fout.write(json.dumps(rec) + "\n"); fout.flush()
            total_processed += 1
            if total_processed % 25 == 0:
                elapsed = time.time(), t_start
                print(f"[recovery] {total_processed} judged  elapsed={elapsed:.0f}s", flush=True)

    print(f"[recovery] DONE  {total_processed} new judgements -> {out_path}", flush=True)
    return 0


# Judge model loader (uses the shared model_loader)

def make_judge_runner(args):
    if args.dry_run:
        # Placeholder that writes a fixed verdict for pipeline testing
        def _dry(system_prompt, user_message, max_new_tokens, _dry_run):
            return {
                "output": "Brief reasoning placeholder.\nVerdict: UNCLEAR",
                "input_tokens": (len(system_prompt) + len(user_message)) // 4,
                "output_tokens": 10,
                "latency_s": 0.0,
            }
        return _dry

    # Uses the same model_loader as the experiments runner.
    # Pass model_type="auto" so the loader infers the family from the path.
    from model_loader import load_model, run_inference
    print(f"[load] loading judge model from {args.judge_model_path}", flush=True)
    t0 = time.time()
    tokenizer, model = load_model(args.judge_model_path, model_type="auto")
    print(f"[load] judge loaded in {time.time(), t0:.1f}s  "
          f"family={getattr(model, '_caveman_family', None)!r}", flush=True)

    def _run(system_prompt, user_message, max_new_tokens, _dry_run):
        result = run_inference(
            tokenizer=tokenizer,
            model=model,
            system_prompt=system_prompt,
            user_message=user_message,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
        )
        return {
            "output": result["output"],
            "input_tokens": result["input_tokens"],
            "output_tokens": result["output_tokens"],
            "latency_s": result["latency_s"],
        }
    return _run


# Main

def main() -> int:
    args = parse_args()
    print(f"[start] mode={args.mode} target={args.target_model} dataset={args.dataset} "
          f"condition={args.condition} n={args.n_samples} dry_run={args.dry_run}", flush=True)

    runner = make_judge_runner(args)
    if args.mode == "pair":
        return run_pair_mode(args, runner)
    elif args.mode == "recovery":
        return run_recovery_mode(args, runner)
    else:
        print(f"[fatal] unknown mode: {args.mode}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
