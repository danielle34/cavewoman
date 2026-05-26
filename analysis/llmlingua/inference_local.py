"""Generalized local-GPU inference on LLMLingua-compressed prompts.

Replaces the per-model variants (inference_qwen.py). Takes --model_tag
and dispatches to the right loader via src/model_loader.py.

Supported tags (with their HuggingFace IDs as the recommended --model_path):
    qwen-2.5  -> Qwen/Qwen2.5-VL-7B-Instruct           (model_type='qwen')
    qwen-3.5  -> Qwen/Qwen3.5-9B-Instruct              (model_type='qwen-3.5')
    deepseek -> deepseek-ai/DeepSeek-R1-Distill-Qwen-7B (model_type='deepseek')

`--model_path` is required; pass a local snapshot path or the HuggingFace ID.

Output:
    results/llmlingua/inference/<model_tag>/<dataset>/<dataset>_LLMLingua.jsonl
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from dataset_loader_multi import extract_answer, check_correct  # noqa: E402

COMPRESSED_BASE = REPO / "results" / "llmlingua" / "compressed"
INFERENCE_BASE = REPO / "results" / "llmlingua" / "inference"

NEUTRAL_INPUT_SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Answer the following question accurately and completely."
)
LEVEL_NAME = "LLMLingua"
MAX_NEW_TOKENS = 400

MODEL_REGISTRY = {
    # Keys are the HuggingFace-release tags used in output paths.
    # `model_type` is the internal family hint consumed by src/model_loader.py.
    "qwen-2.5":     {"model_type": "qwen"},
    "qwen-3.5":     {"model_type": "qwen3.5"},
    "deepseek-r1":  {"model_type": "deepseek"},
    "gemma-4":      {"model_type": "gemma"},
}

_STOP = {"flag": False}


def _stop_handler(signum, frame):
    print(f"[local-inf] caught signal {signum}; finishing current item then exiting cleanly",
          flush=True)
    _STOP["flag"] = True


signal.signal(signal.SIGTERM, _stop_handler)
signal.signal(signal.SIGUSR1, _stop_handler)


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
    with p.open() as f:
        for line in f:
            rec = json.loads(line)
            rows.append(rec)
    rows.sort(key=lambda r: r["idx"])
    return rows


def run_one(dataset: str, model_tag: str, tokenizer, model, run_inference,
            shard: int, num_shards: int) -> None:
    rows = load_compressed(dataset)
    rows = [r for r in rows if (r["idx"] % num_shards) == shard]
    out_dir = INFERENCE_BASE / model_tag / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{dataset}_{LEVEL_NAME}.jsonl"
    done = read_completed(out_path)
    todo = [r for r in rows if r["idx"] not in done]
    print(f"[{model_tag} {dataset} {LEVEL_NAME} shard {shard}/{num_shards}] "
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
                out_text = ""
                in_tok = 0
                out_tok = 0
                latency = 0.0
                err = f"{type(e).__name__}: {e}"
                print(f"  [error] {dataset} idx={rec['idx']}: {err}", flush=True)

            predicted = extract_answer(out_text, rec["answer_type"]) if out_text else None
            is_correct = check_correct(predicted, rec["gold_answer"], rec["answer_type"]) if predicted else False
            correct += int(bool(is_correct))
            row = {
                "idx": rec["idx"],
                "dataset": dataset,
                "level": LEVEL_NAME,
                "model_tag": model_tag,
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
    print(f"[{model_tag} {dataset} {LEVEL_NAME} shard {shard}/{num_shards}] "
          f"EXIT  n={n}  correct={correct}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_tag", required=True, choices=sorted(MODEL_REGISTRY),
                    help="Which model to run; selects output dir and loader config.")
    ap.add_argument("--model_path", required=True,
                    help="Local path to the model snapshot or its HuggingFace ID.")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--datasets", nargs="+", default=["gsm8k", "boolq", "arc_easy"])
    args = ap.parse_args()

    cfg = MODEL_REGISTRY[args.model_tag]
    model_path = args.model_path
    model_type = cfg["model_type"]

    from model_loader import load_model, run_inference  # noqa: E402

    print(f"[local-inf] loading model_tag={args.model_tag}  path={model_path}  type={model_type}",
          flush=True)
    tokenizer, model = load_model(model_path, model_type=model_type)
    print("[local-inf] model ready", flush=True)

    for ds in args.datasets:
        if _STOP["flag"]:
            break
        run_one(ds, args.model_tag, tokenizer, model, run_inference,
                args.shard, args.num_shards)


if __name__ == "__main__":
    main()
