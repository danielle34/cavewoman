"""Step 3a, GPT-4o inference on LLMLingua-compressed prompts (single level).

Reads results/llmlingua/compressed/<dataset>/<dataset>_compressed.jsonl and
runs each compressed prompt through GPT-4o with the same neutral
input-condition system prompt the paper uses for Condition A.

Outputs:
    results/llmlingua/inference/gpt-4o/<dataset>/<dataset>_LLMLingua.jsonl

Resume-safe by idx. Supports sharding across tmux sessions:
    --shard 0 --num_shards 2   (tmux 1)
    --shard 1 --num_shards 2   (tmux 2)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

try:
    from dotenv import load_dotenv
    load_dotenv(REPO / ".env")
except ImportError:
    pass

from dataset_loader_multi import extract_answer, check_correct  # noqa: E402

COMPRESSED_BASE = REPO / "results" / "llmlingua" / "compressed"
OUT_BASE = REPO / "results" / "llmlingua" / "inference" / "gpt-4o"

NEUTRAL_INPUT_SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Answer the following question accurately and completely."
)
LEVEL_NAME = "LLMLingua"
MAX_NEW_TOKENS = 400

PRICE_IN_PER_1M = 2.50
PRICE_OUT_PER_1M = 10.00


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


def load_compressed(dataset: str) -> list:
    p = COMPRESSED_BASE / dataset / f"{dataset}_compressed.jsonl"
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


def call_gpt-4o(client, prompt: str, max_retries: int = 4):
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            t0 = time.time()
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": NEUTRAL_INPUT_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=MAX_NEW_TOKENS,
            )
            latency = time.time(), t0
            return (
                resp.choices[0].message.content or "",
                int(resp.usage.prompt_tokens),
                int(resp.usage.completion_tokens),
                resp.model,
                latency,
                None,
            )
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            if "content filtering" in msg or "content_policy_violation" in msg:
                return "", 0, 0, "gpt-4o", 0.0, f"content_filtered: {e}"
            if attempt == max_retries:
                return "", 0, 0, "gpt-4o", 0.0, f"{type(e).__name__}: {e}"
            wait = 2 ** (attempt + 1)
            print(f"  [retry {attempt + 1}/{max_retries}] {type(e).__name__}: {e}  sleep {wait}s",
                  flush=True)
            time.sleep(wait)


def cost(in_tok: int, out_tok: int) -> float:
    return (in_tok / 1_000_000) * PRICE_IN_PER_1M + (out_tok / 1_000_000) * PRICE_OUT_PER_1M


def run_one(dataset: str, client, shard: int, num_shards: int,
            rate_sleep_s: float) -> None:
    rows = load_compressed(dataset)
    rows = [r for r in rows if (r["idx"] % num_shards) == shard]
    out_dir = OUT_BASE / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{dataset}_{LEVEL_NAME}.jsonl"
    done = read_completed(out_path)
    todo = [r for r in rows if r["idx"] not in done]
    print(f"[{dataset} {LEVEL_NAME} shard {shard}/{num_shards}] "
          f"items={len(rows)} done={len(done)} todo={len(todo)}", flush=True)

    t0 = time.time()
    spent = 0.0
    correct = 0
    n = 0
    with out_path.open("a") as f:
        for rec in todo:
            prompt = rec["compressed_prompt"]
            out_text, in_tok, out_tok, api_model, latency, err = call_gpt-4o(client, prompt)
            spent += cost(in_tok, out_tok)
            predicted = extract_answer(out_text, rec["answer_type"]) if out_text else None
            is_correct = check_correct(predicted, rec["gold_answer"], rec["answer_type"]) if predicted else False
            correct += int(bool(is_correct))
            row = {
                "idx": rec["idx"],
                "dataset": dataset,
                "level": LEVEL_NAME,
                "model_tag": "gpt-4o",
                "api_model": api_model,
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
                "cost_usd": cost(in_tok, out_tok),
                "latency_s": round(latency, 3),
                "error": err,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            n += 1
            if n % 50 == 0 or n == 1:
                elapsed = time.time(), t0
                rate = n / elapsed if elapsed > 0 else 0
                eta = (len(todo), n) / rate if rate > 0 else 0
                acc = correct / n if n else 0
                print(f"  [{dataset}] n={n}/{len(todo)}  acc={acc:.3f}  "
                      f"spent=${spent:.2f}  rate={rate:.1f}/s  eta={eta/60:.1f}min",
                      flush=True)
            if rate_sleep_s > 0:
                time.sleep(rate_sleep_s)
    print(f"[{dataset} {LEVEL_NAME} shard {shard}/{num_shards}] DONE  n={n}  correct={correct}  spent=${spent:.2f}",
          flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--datasets", nargs="+", default=["gsm8k", "boolq", "arc_easy"])
    ap.add_argument("--rate_sleep_s", type=float, default=0.0)
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not set; check repository .env")

    import openai
    client = openai.OpenAI()

    # Outer loop: keep cycling through datasets until every dataset has been
    # processed AND its compressed-file row count is stable (compression done).
    # This lets us launch the tmux session before compression finishes; the
    # worker will gracefully pick up new compressed rows as they land.
    POLL_INTERVAL_S = 60
    seen_counts = {ds: -1 for ds in args.datasets}
    idle_passes = 0
    while True:
        progressed = False
        for ds in args.datasets:
            p = COMPRESSED_BASE / ds / f"{ds}_compressed.jsonl"
            if not p.exists():
                print(f"[{ds}] no compressed file yet, will retry", flush=True)
                continue
            n_now = sum(1 for _ in p.open())
            if n_now == seen_counts[ds]:
                # No new compressed rows since last pass, skip cheap.
                continue
            seen_counts[ds] = n_now
            print(f"[{ds}] compressed rows visible: {n_now}", flush=True)
            run_one(ds, client, args.shard, args.num_shards, args.rate_sleep_s)
            progressed = True

        # Termination check: all 3 datasets present and we just finished a
        # pass that found nothing new to do for any of them.
        if not progressed:
            idle_passes += 1
            # Give compression up to ~30 min to start writing if files missing.
            if idle_passes >= 30 and all(seen_counts[ds] >= 0 for ds in args.datasets):
                print(f"[main] no progress for {idle_passes} polls; all datasets present, exiting",
                      flush=True)
                break
            print(f"[main] idle pass #{idle_passes}; sleeping {POLL_INTERVAL_S}s then checking again",
                  flush=True)
            time.sleep(POLL_INTERVAL_S)
        else:
            idle_passes = 0


if __name__ == "__main__":
    main()
