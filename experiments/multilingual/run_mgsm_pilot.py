"""MGSM multilingual runner: local models, Condition B.

For each (language, level): load the MGSM test split, generate under the
level's register constraint, extract the numeric answer, score against
answer_number.

Outputs (resume-safe, appends per cell):
  results/multilingual/<tag>_output/<lang>/caveman_<tag>_mgsm-<lang>_output_<Lx>.jsonl
  results/multilingual/<tag>_output/<lang>/run_config_<Lx>.json

Usage:
  python experiments/multilingual/run_mgsm_pilot.py \
      --model_name qwen2.5 --model_path ~/models/Qwen2.5-VL-7B \
      --langs de,zh,bn,sw --levels L0,L1,L2,L3,L4 --n 250

Add --disable_thinking for Qwen3-family models.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
REPO = _REPO_ROOT
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prompts_mgsm import system_prompt, MAX_NEW_TOKENS  # noqa: E402

# Bengali digits -> ASCII (models occasionally answer in Bengali numerals)
BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")

NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def extract_number(text: str):
    """Prefer the last number on an 'Answer:' line; fall back to the last
    number anywhere. Normalizes Bengali digits and strips separators."""
    t = text.translate(BN_DIGITS)
    t = t.replace(",", "").replace(" ", " ")
    answer_lines = [ln for ln in t.splitlines() if "answer" in ln.lower()]
    for source in (answer_lines[::-1], [t]):
        for chunk in source:
            nums = NUM_RE.findall(chunk)
            if nums:
                return nums[-1]
    return None


def check_numeric(pred, gold) -> bool:
    if pred is None or gold is None:
        return False
    try:
        return abs(float(pred) - float(gold)) <= 0.01
    except ValueError:
        return False


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
    ap.add_argument("--langs", default="de,zh,bn,sw")
    ap.add_argument("--levels", default="L0,L1,L2")
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--output_root",
                    default=str(REPO / "results" / "multilingual"))
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="Decoding temperature (0.7 for paired-L0 noise-floor draws).")
    ap.add_argument("--suffix", default="",
                    help="Filename suffix, e.g. _paired for the T>0 second draw.")
    ap.add_argument("--disable_thinking", action="store_true",
                    help="Suppress the chat template's reasoning block. Use this "
                         "on Qwen3-family models.")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    levels = [x.strip() for x in args.levels.split(",") if x.strip()]

    from datasets import load_dataset
    data = {lang: load_dataset("juletxara/mgsm", lang, split="test")
            for lang in langs}

    tokenizer = model = None
    if not args.dry_run:
        from model_loader import load_model, run_inference
        print(f"[load] {args.model_path}", flush=True)
        tokenizer, model = load_model(args.model_path, model_type="auto")

    tag = args.model_name
    for lang in langs:
        ds = data[lang]
        for level in levels:
            out_dir = Path(args.output_root) / f"{tag}_output" / lang
            if not args.dry_run:
                out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"caveman_{tag}_mgsm-{lang}_output_{level}{args.suffix}.jsonl"
            done = completed_idx(out_path)
            sp = system_prompt(level, lang)
            max_new = MAX_NEW_TOKENS[level]
            if not args.dry_run:
                (out_dir / f"run_config_{level}.json").write_text(json.dumps({
                    "model_name": tag, "model_path": args.model_path,
                    "dataset": f"mgsm-{lang}", "condition": "output",
                    "level": level, "n": min(args.n, len(ds)),
                    "system_prompt": sp, "max_new_tokens": max_new,
                    "temperature": args.temperature,
                    "written_at": datetime.now(timezone.utc).isoformat(),
                }, indent=2), encoding="utf-8")

            n_new = 0
            t0 = time.time()
            import contextlib, io
            _sink = io.StringIO() if args.dry_run else open(out_path, "a", encoding="utf-8")
            with contextlib.closing(_sink) as fout:
                for idx in range(min(args.n, len(ds))):
                    if idx in done:
                        continue
                    item = ds[idx]
                    q = item["question"]
                    gold = item["answer_number"]
                    if args.dry_run:
                        result = {"output": f"[dry {lang}/{level}/{idx}] Answer: {gold}",
                                  "input_tokens": 0, "output_tokens": 0,
                                  "latency_s": 0.0}
                    else:
                        result = run_inference(
                            tokenizer=tokenizer, model=model,
                            system_prompt=sp, user_message=q,
                            max_new_tokens=max_new, temperature=args.temperature,
                            enable_thinking=(False if args.disable_thinking else None))
                    pred = extract_number(result["output"] or "")
                    rec = {
                        "idx": idx, "lang": lang, "level": level,
                        "condition": "output", "model": tag,
                        "question": q, "gt_answer": gold,
                        "output": result["output"],
                        "predicted_answer": pred,
                        "correct": check_numeric(pred, gold),
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
                        print(f"[{tag} {lang} {level}] {n_new} done "
                              f"({el/max(n_new,1):.1f}s/item)", flush=True)
            print(f"[cell {tag}/{lang}/{level}] complete, new={n_new}", flush=True)
    print("[mgsm pilot] ALL DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
