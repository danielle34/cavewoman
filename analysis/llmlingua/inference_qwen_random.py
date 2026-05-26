"""Step 3b (Random-deletion variant), Qwen2.5-VL-7B inference on randomly-compressed prompts.

Mirrors inference_qwen.py exactly, but reads from `results/random/compressed/`
and writes to `results/random/inference/qwen-2.5/` with `Random_t0.5` / `Random_t0.8`
level names instead of `LLMLingua_...`.

Use:
    --tau_tag t0.5     reads  <ds>_random_compressed_t0.5.jsonl
                       writes <ds>_Random_t0.5.jsonl   (level=Random_t0.5)
    --tau_tag t0.8     same with t0.8

Resume-safe by idx. Shardable via --shard / --num_shards.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from dataset_loader_multi import extract_answer, check_correct  # noqa: E402

COMPRESSED_BASE = REPO / "results" / "random" / "compressed"
OUT_BASE = REPO / "results" / "random" / "inference" / "qwen-2.5"

NEUTRAL_INPUT_SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Answer the following question accurately and completely."
)
MAX_NEW_TOKENS = 400

_STOP = {"flag": False}


def _stop_handler(signum, frame):
    print(f"[qwen-rand] caught signal {signum}; finishing current item then exiting cleanly",
          flush=True)
    _STOP["flag"] = True


signal.signal(signal.SIGTERM, _stop_handler)
signal.signal(signal.SIGUSR1, _stop_handler)


def _resolve(tau_tag):
    """Random files always have a tau_tag (no default, we run 0.5 and 0.8)."""
    if not tau_tag:
        raise SystemExit("Random inference requires --tau_tag (e.g. 't0.5' or 't0.8')")
    level = f"Random_{tau_tag}"
    in_filename = "{ds}_random_compressed_" + tau_tag + ".jsonl"
    out_filename = "{ds}_" + level + ".jsonl"
    return level, in_filename, out_filename


def read_completed(path: Path) -> set:
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


def load_compressed(dataset: str, in_filename_tmpl: str) -> list:
    p = COMPRESSED_BASE / dataset / in_filename_tmpl.format(ds=dataset)
    rows = []
    if not p.exists():
        return rows
    with p.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
                rows.append(rec)
            except Exception:
                continue
    rows.sort(key=lambda r: r["idx"])
    return rows


def run_one(dataset, tokenizer, model, run_inference,
            shard, num_shards, level_name, in_tmpl, out_tmpl):
    rows = load_compressed(dataset, in_tmpl)
    rows = [r for r in rows if (r["idx"] % num_shards) == shard]
    out_dir = OUT_BASE / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / out_tmpl.format(ds=dataset)
    done = read_completed(out_path)
    todo = [r for r in rows if r["idx"] not in done]
    print(f"[{dataset} {level_name} shard {shard}/{num_shards}] "
          f"items={len(rows)} done={len(done)} todo={len(todo)}", flush=True)

    t0 = time.time()
    correct = 0
    n = 0
    with out_path.open("a") as f:
        for rec in todo:
            if _STOP["flag"]:
                print(f"[{dataset}] stopping early due to signal", flush=True)
                break
            prompt = rec["compressed_prompt"]
            try:
                res = run_inference(
                    tokenizer=tokenizer,
                    model=model,
                    system_prompt=NEUTRAL_INPUT_SYSTEM_PROMPT,
                    user_message=prompt,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=0.0,
                )
                out_text = res["output"]
                in_tok = res["input_tokens"]
                out_tok = res["output_tokens"]
                latency = res["latency_s"]
                err = None
            except Exception as e:
                out_text, in_tok, out_tok, latency = "", 0, 0, 0.0
                err = f"{type(e).__name__}: {e}"
                print(f"  [error] {dataset} idx={rec['idx']}: {err}", flush=True)

            predicted = extract_answer(out_text, rec["answer_type"]) if out_text else None
            is_correct = check_correct(predicted, rec["gold_answer"], rec["answer_type"]) if predicted else False
            correct += int(bool(is_correct))
            row = {
                "idx": rec["idx"],
                "dataset": dataset,
                "level": level_name,
                "model_tag": "qwen-2.5",
                "system_prompt_kind": "neutral_input",
                "compressed_prompt": prompt,
                "original_prompt": rec["original_prompt"],
                "target_tau": rec["target_tau"],
                "actual_tau": rec["actual_tau"],
                "output": out_text,
                "predicted_answer": predicted,
                "gold_answer": rec["gold_answer"],
                "answer_type": rec["answer_type"],
                "correct": bool(is_correct),
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "latency_s": round(latency, 3),
                "error": err,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            n += 1
            if n % 25 == 0 or n == 1:
                elapsed = time.time(), t0
                rate = n / elapsed if elapsed > 0 else 0
                eta = (len(todo), n) / rate if rate > 0 else 0
                acc = correct / n if n else 0
                print(f"  [{dataset}] n={n}/{len(todo)}  acc={acc:.3f}  "
                      f"rate={rate:.2f}/s  eta={eta/60:.1f}min", flush=True)
    print(f"[{dataset} {level_name} shard {shard}/{num_shards}] EXIT  n={n}  correct={correct}",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--datasets", nargs="+",
                    default=["gsm8k", "boolq", "arc_easy", "commonsenseqa", "mmlu_stem"])
    ap.add_argument("--model_path", required=True,
                    help="Local path to a Qwen2.5-VL-7B snapshot or its HuggingFace ID.")
    ap.add_argument("--tau_tag", required=True, help="e.g. 't0.5' or 't0.8'")
    args = ap.parse_args()

    level, in_tmpl, out_tmpl = _resolve(args.tau_tag)
    print(f"[qwen-rand] level={level}  reading={in_tmpl}  writing={out_tmpl}", flush=True)

    from model_loader import load_model, run_inference
    print(f"[qwen-rand] loading {args.model_path} ...", flush=True)
    tokenizer, model = load_model(args.model_path, model_type="qwen")
    print("[qwen-rand] model ready", flush=True)

    for ds in args.datasets:
        if _STOP["flag"]:
            break
        run_one(ds, tokenizer, model, run_inference,
                args.shard, args.num_shards, level, in_tmpl, out_tmpl)


if __name__ == "__main__":
    main()
