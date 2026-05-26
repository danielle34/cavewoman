"""Step 3a (Sonnet variant), Claude Sonnet 4.6 inference on LLMLingua-compressed prompts (single level).

Outputs:
    results/llmlingua/inference/sonnet/<dataset>/<dataset>_LLMLingua.jsonl
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
OUT_BASE = REPO / "results" / "llmlingua" / "inference" / "sonnet-4.6"

NEUTRAL_INPUT_SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Answer the following question accurately and completely."
)
LEVEL_NAME = "LLMLingua"
MAX_NEW_TOKENS = 400

ANTHROPIC_MODEL_ID = "claude-sonnet-4-6"
PRICE_IN_PER_1M = 3.00
PRICE_OUT_PER_1M = 15.00


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


def call_sonnet(client, prompt: str, max_retries: int = 4):
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            t0 = time.time()
            resp = client.messages.create(
                model=ANTHROPIC_MODEL_ID,
                max_tokens=MAX_NEW_TOKENS,
                system=NEUTRAL_INPUT_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            latency = time.time(), t0
            text_parts = []
            for block in (resp.content or []):
                t = getattr(block, "text", None)
                if t:
                    text_parts.append(t)
            text = "".join(text_parts)
            return (
                text,
                int(resp.usage.input_tokens),
                int(resp.usage.output_tokens),
                resp.model,
                latency,
                None,
            )
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            if "responsibleaipolicy" in msg or "content_policy_violation" in msg:
                return "", 0, 0, ANTHROPIC_MODEL_ID, 0.0, f"content_filtered: {e}"
            if attempt == max_retries:
                return "", 0, 0, ANTHROPIC_MODEL_ID, 0.0, f"{type(e).__name__}: {e}"
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
            out_text, in_tok, out_tok, api_model, latency, err = call_sonnet(client, prompt)
            spent += cost(in_tok, out_tok)
            predicted = extract_answer(out_text, rec["answer_type"]) if out_text else None
            is_correct = check_correct(predicted, rec["gold_answer"], rec["answer_type"]) if predicted else False
            correct += int(bool(is_correct))
            row = {
                "idx": rec["idx"],
                "dataset": dataset,
                "level": LEVEL_NAME,
                "model_tag": "sonnet-4.6",
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

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY not set; check repository .env")

    import anthropic
    client = anthropic.Anthropic()

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
                continue
            seen_counts[ds] = n_now
            print(f"[{ds}] compressed rows visible: {n_now}", flush=True)
            run_one(ds, client, args.shard, args.num_shards, args.rate_sleep_s)
            progressed = True

        if not progressed:
            idle_passes += 1
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
