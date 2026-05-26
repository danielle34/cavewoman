"""CAVEWOMAN runner for API models, OpenAI (gpt-4o) AND Anthropic (claude-haiku-4-5).

Mirrors run_experiment.py's experimental contract exactly: same constraint
prompts, same dataset loader, same input-compression function, same
answer-extraction / correctness logic, same per-record schema PLUS two new
fields (`cost_usd`, `api_model`).

Loads API keys from a `.env` file at the repository root via python-dotenv:
, OPENAI_API_KEY    (needed for any `gpt-*` model)
, ANTHROPIC_API_KEY (needed for any `claude-*` model)

CLI:
    python experiments/run_experiment_api.py \\
        --model {gpt-4o | claude-haiku-4-5} \\
        --dataset gsm8k --condition output --level L0 \\
        --output_dir ./results/<tag>_output/gsm8k \\
        [--dry_run] [--start_idx N] [--end_idx N]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Load .env from the repo root BEFORE anything else (so API keys are set when clients init).
_REPO_ROOT = Path(__file__).resolve().parent.parent
try:
    from dotenv import load_dotenv
    load_dotenv(_REPO_ROOT / ".env")
except ImportError:
    print("[warn] python-dotenv not installed; will rely on OS env for API keys.", file=sys.stderr)

import argparse
import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

sys.path.insert(0, str(_REPO_ROOT / "src"))

from constraint_prompts import (  # noqa: E402
    CONSTRAINT_PROMPTS,
    LEVEL_ORDER,
    get_max_tokens,
)
from dataset_loader_multi import (  # noqa: E402
    load_dataset_caveman,
    extract_answer,
    check_correct,
    compress_input,
)
from metrics_utils import (  # noqa: E402
    count_semantic_units,
    compute_info_density,
)


# Mirror run_experiment.py's neutral Condition-B prompt, single source of truth here
# (we don't import it from run_experiment.py because that module pulls torch/transformers).
NEUTRAL_INPUT_SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Answer the following question accurately and completely."
)


# Model registry, provider, filename tag, real API model id, pricing
#
# Pricing is in USD per million tokens (as of 2025).
#
# To add a new model: drop a row here. Code branches on `provider`.

MODEL_REGISTRY: Dict[str, Dict] = {
    "gpt-4o": {
        "provider": "openai",
        "tag": "gpt-4o",
        "api_model": "gpt-4o",
        "price_in_per_1m": 2.50,
        "price_out_per_1m": 10.00,
    },
    "gpt-5.4-2026-03-05": {
        "provider": "openai",
        "tag": "gpt-5.4",
        "api_model": "gpt-5.4-2026-03-05",
        # Placeholder pricing until OpenAI publishes official rates; update
        # when the model's pricing page is live.
        "price_in_per_1m": 5.00,
        "price_out_per_1m": 20.00,
    },
    "claude-haiku-4-5": {
        "provider": "anthropic",
        "tag": "haiku-4.5",
        "api_model": "claude-haiku-4-5-20251001",
        "price_in_per_1m": 1.00,
        "price_out_per_1m": 5.00,
    },
    "claude-sonnet-4-6": {
        "provider": "anthropic",
        "tag": "sonnet-4.6",
        # Alias; Anthropic resolves to the current dated version and returns
        # it as resp.model, that exact string lands in the `api_model` field
        # of every JSONL record for reproducibility.
        "api_model": "claude-sonnet-4-6",
        "price_in_per_1m": 3.00,
        "price_out_per_1m": 15.00,
    },
    "kimi-k2.6": {
        # Kimi-K2.6 emits hidden reasoning tokens that count against max_tokens
        # but are not returned in message.content. The L0 budget is bumped to
        # 1200 for Kimi only; L1-L4 keep the standard compression budgets.
        "provider": "azure_openai",
        "tag": "kimi-k2.6",
        # For Azure deployments `api_model` is the deployment name configured
        # in your Azure portal; the SDK uses that string as the `model` kwarg.
        "api_model": "Kimi-K2.6",
        # Azure endpoint is read from the AZURE_KIMI_ENDPOINT env var at
        # client-init time; set it in your .env alongside AZURE_KIMI_API_KEY.
        "azure_endpoint": None,
        "azure_api_version": "2024-10-21",
        # Moonshot-published Kimi-K2.6 pricing (cache-miss input rate).
        "price_in_per_1m": 0.95,
        "price_out_per_1m": 4.00,
    },
}


def get_model_config(model: str) -> Dict:
    """Look up the model in the registry; fall back to OpenAI defaults if unknown."""
    if model in MODEL_REGISTRY:
        return MODEL_REGISTRY[model]
    # Backwards-compat fallback: treat unknown models as OpenAI Chat Completion.
    return {
        "provider": "openai",
        "tag": model.replace("-", "").replace(".", ""),
        "api_model": model,
        "price_in_per_1m": 2.50,
        "price_out_per_1m": 10.00,
    }


def compute_cost(input_tokens: int, output_tokens: int, price_in_per_1m: float, price_out_per_1m: float) -> float:
    return (input_tokens / 1e6) * price_in_per_1m + (output_tokens / 1e6) * price_out_per_1m


_DATASET_CANONICAL_SPLIT = {
    "gsm8k": "test",
    "boolq": "validation",
    "arc_easy": "test",
    "commonsenseqa": "validation",
    "mmlu_stem": "test",
}


# CLI

def parse_args():
    p = argparse.ArgumentParser(
        description="CAVEWOMAN runner for API models (OpenAI + Anthropic)."
    )
    p.add_argument(
        "--model",
        default="gpt-4o",
        help=("Model id. Registered: " + ", ".join(MODEL_REGISTRY.keys()) +
              ". Unregistered values fall back to OpenAI Chat Completion."),
    )
    p.add_argument("--dataset", required=True,
                   choices=["gsm8k", "boolq", "arc_easy", "commonsenseqa", "mmlu_stem"])
    p.add_argument("--condition", required=True, choices=["output", "input"])
    p.add_argument("--level", required=True,
                   choices=["all"] + LEVEL_ORDER,
                   help="L0..L4 or 'all'.")
    p.add_argument("--output_dir", required=True,
                   help="Where to write the per-level JSONL files.")
    p.add_argument("--dry_run", action="store_true",
                   help="Don't call the API; write placeholder records.")
    p.add_argument("--start_idx", type=int, default=None,
                   help="Lower bound on dataset idx (inclusive).")
    p.add_argument("--end_idx", type=int, default=None,
                   help="Upper bound on dataset idx (inclusive).")
    p.add_argument("--seed", type=int, default=42,
                   help="Sampling seed (only relevant if subsetting; default 42).")
    p.add_argument("--rate_sleep_s", type=float, default=0.5,
                   help="Sleep between API calls (default 0.5s).")
    p.add_argument("--max_retries", type=int, default=3,
                   help="API retry budget (default 3).")
    return p.parse_args()


# JSONL helpers

def _read_completed_idx(path: Path) -> Set[int]:
    if not path.exists():
        return set()
    done: Set[int] = set()
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            idx = rec.get("idx")
            if isinstance(idx, int):
                done.add(idx)
    return done


# Provider-dispatched API call (with retry)

def call_with_retry(
    *,
    provider: str,
    client,
    api_model: str,
    system_prompt: str,
    user_message: str,
    max_tokens: int,
    max_retries: int,
    dry_run: bool,
):
    """Returns (output_text, input_tokens, output_tokens, api_model_returned, latency_s).

    Branches on `provider`:
    , "openai"    -> client.chat.completions.create(...)
    , "anthropic" -> client.messages.create(...)
    """
    if dry_run:
        placeholder = f"[DRY RUN, provider={provider} model={api_model}]\nAnswer: <placeholder>"
        in_est = (len(system_prompt) + len(user_message)) // 4
        out_est = len(placeholder) // 4
        return placeholder, in_est, out_est, api_model, 0.0

    last_err: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            t0 = time.time()
            # OpenAI and Azure OpenAI share the same Chat Completions surface;
            # the only difference is how the client was constructed.
            if provider in ("openai", "azure_openai"):
                # Reasoning-mode models (OpenAI o-series, GPT-5.x, and the Azure
                # Kimi-K2.6 deployment) reject `max_tokens` and require
                # `max_completion_tokens` instead; they also reject non-default
                # temperature. With `max_completion_tokens`, only visible output
                # tokens count against the budget (reasoning is "free" up to
                # model context). Detect by api_model / deployment prefix.
                is_reasoning_model = (
                    api_model.startswith(("gpt-5", "o1", "o3", "o4"))
                    or api_model.lower().startswith("kimi-k2.6")
                )
                is_new_openai = is_reasoning_model  # back-compat alias
                kwargs: Dict[str, Any] = {
                    "model": api_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_message},
                    ],
                }
                if is_new_openai:
                    kwargs["max_completion_tokens"] = max_tokens
                    # temperature must be 1 (default) on these models, omit
                else:
                    kwargs["max_tokens"] = max_tokens
                    kwargs["temperature"] = 0.0
                resp = client.chat.completions.create(**kwargs)
                latency_s = time.time(), t0
                text = resp.choices[0].message.content or ""
                return (
                    text,
                    int(resp.usage.prompt_tokens),
                    int(resp.usage.completion_tokens),
                    resp.model,
                    latency_s,
                )
            elif provider == "anthropic":
                resp = client.messages.create(
                    model=api_model,
                    max_tokens=max_tokens,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_message}],
                    temperature=0.0,
                )
                latency_s = time.time(), t0
                # `resp.content` is a list of content blocks. For text-only,
                # the first block's .text is what we want.
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
                    latency_s,
                )
            else:
                raise RuntimeError(f"Unknown provider: {provider!r}")
        except Exception as e:
            last_err = e
            # Content-policy rejections are not transient, retrying won't help.
            # Surface them with a distinct marker so the caller can record a
            # placeholder and move on instead of crashing the whole sweep.
            msg = str(e)
            is_content_filter = (
                "content filtering policy" in msg.lower()
                or "content_policy_violation" in msg.lower()
                or "responsibleaipolicyviolation" in msg.lower()
            )
            if is_content_filter:
                raise ContentFilteredError(msg) from e
            if attempt == max_retries:
                raise
            wait = 2 ** (attempt + 1)
            print(f"  [retry {attempt + 1}/{max_retries}] {type(e).__name__}: {e}  sleeping {wait}s", flush=True)
            time.sleep(wait)
    assert last_err is not None
    raise last_err


class ContentFilteredError(RuntimeError):
    """Raised when the provider rejects an item due to content policy.
    The caller should record a placeholder JSONL entry and continue."""
    pass


# Per-level sweep

def run_one_level(args, client, level: str, items: List[Dict], model_config: Dict) -> Dict:
    model_tag = model_config["tag"]
    api_model_id = model_config["api_model"]
    provider = model_config["provider"]
    price_in = model_config["price_in_per_1m"]
    price_out = model_config["price_out_per_1m"]

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"caveman_{model_tag}_{args.dataset}_{args.condition}_{level}.jsonl"

    completed = _read_completed_idx(out_path)
    todo = [it for it in items if it["idx"] not in completed]
    if completed:
        print(f"[{level}] Resuming: {len(completed)} done, {len(todo)} to go (file: {out_path.name})", flush=True)
    else:
        print(f"[{level}] Starting fresh: {len(todo)} items (file: {out_path.name})", flush=True)

    if args.condition == "input":
        system_prompt = NEUTRAL_INPUT_SYSTEM_PROMPT
    else:
        system_prompt = CONSTRAINT_PROMPTS[level]
    max_new = get_max_tokens(level)
    # Kimi-K2.6 emits hidden reasoning tokens that count against max_tokens
    # (see KIMI_STATUS.md). At L0 the 400-token default is exhausted by
    # thinking before the visible "Answer: X" line is emitted; budget probe
    # (2026-05-25) showed 1200 gives 5/5 pass rate at avg ~590 actual tokens
    # used. L1-L4 keep the original CAVEWOMAN compression budgets, Kimi will
    # often fail extraction at those levels, which is a finding (reasoning
    # models cannot satisfy strict output-token compression), not a bug.
    if model_tag == "kimi-k2.6" and level == "L0":
        max_new = max(max_new, 1200)

    stats = {
        "level": level,
        "n_processed": 0,
        "n_correct": 0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_cost_usd": 0.0,
        "n_resumed": len(completed),
    }
    t_start = time.time()

    with open(out_path, "a") as fout:
        for i, item in enumerate(todo, start=1):
            user_message = (
                compress_input(item["question_formatted"], level)
                if args.condition == "input"
                else item["question_formatted"]
            )

            content_filtered = False
            try:
                output_text, in_tok, out_tok, api_model_returned, latency_s = call_with_retry(
                    provider=provider,
                    client=client,
                    api_model=api_model_id,
                    system_prompt=system_prompt,
                    user_message=user_message,
                    max_tokens=max_new,
                    max_retries=args.max_retries,
                    dry_run=args.dry_run,
                )
            except ContentFilteredError as cfe:
                # Provider refused this item, write a placeholder so the idx
                # doesn't block the rest of the sweep. The downstream entailment
                # / embedding / accuracy scripts already handle missing outputs.
                print(f"  [content_filter] idx={item['idx']} skipping: {cfe}",
                      flush=True)
                output_text = ""
                in_tok = 0
                out_tok = 0
                api_model_returned = api_model_id
                latency_s = 0.0
                content_filtered = True
            cost = compute_cost(in_tok, out_tok, price_in, price_out)

            answer_type = item.get("answer_type", "numeric")
            gt = item["answer_gt"]
            predicted = extract_answer(output_text, answer_type) if output_text else None
            correct = check_correct(predicted, gt, answer_type) if predicted is not None else False
            sem = count_semantic_units(output_text) if output_text else 0
            density = compute_info_density(output_text, out_tok) if output_text else 0.0

            record = {
                "idx": item["idx"],
                "level": level,
                "model": api_model_returned,
                "question": item.get("question_raw", item.get("question_formatted", "")),
                "gt_answer": gt,
                "output": output_text,
                "predicted_answer": predicted,
                "correct": correct,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "latency_s": latency_s,
                "semantic_units": sem,
                "info_density": density,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "dataset": args.dataset,
                "condition": args.condition,
                "answer_type": answer_type,
                "question_formatted": item.get("question_formatted", ""),
                "user_message": user_message,
                "system_prompt_kind": "neutral" if args.condition == "input" else f"caveman_{level}",
                "start_idx_filter": args.start_idx,
                "end_idx_filter": args.end_idx,
                "cost_usd": round(cost, 8),
                "api_model": api_model_returned,
                "content_filtered": content_filtered,
            }
            fout.write(json.dumps(record) + "\n")
            fout.flush()

            stats["n_processed"] += 1
            stats["total_input_tokens"] += in_tok
            stats["total_output_tokens"] += out_tok
            stats["total_cost_usd"] += cost
            if correct:
                stats["n_correct"] += 1

            if i % 100 == 0 and stats["n_processed"] > 0:
                done = stats["n_processed"]
                remain = len(todo), done
                cost_so_far = stats["total_cost_usd"]
                est_total = cost_so_far / done * len(todo) if done else 0.0
                est_remain = max(est_total, cost_so_far, 0.0)
                elapsed = time.time(), t_start
                rate = done / elapsed if elapsed > 0 else 0.0
                print(
                    f"[{level}] {done}/{len(todo)}  remain={remain}  "
                    f"cost=${cost_so_far:.4f}  est_total=${est_total:.4f}  "
                    f"est_remain=${est_remain:.4f}  rate={rate:.2f}/s",
                    flush=True,
                )

            if not args.dry_run and args.rate_sleep_s > 0:
                time.sleep(args.rate_sleep_s)

    print(
        f"[{level}] END  calls={stats['n_processed']}  "
        f"in_tok={stats['total_input_tokens']}  out_tok={stats['total_output_tokens']}  "
        f"cost=${stats['total_cost_usd']:.4f}",
        flush=True,
    )
    return stats


# Main

def main() -> None:
    args = parse_args()
    config = get_model_config(args.model)
    provider = config["provider"]
    model_tag = config["tag"]

    # Init the right client unless dry-run
    if args.dry_run:
        client = None
        print(f"[dry_run] provider={provider} model={args.model} -> tag={model_tag}; NO API calls will be made.", flush=True)
    else:
        if provider == "openai":
            key = os.environ.get("OPENAI_API_KEY", "").strip()
            if not key or "your_" in key:
                print("[error] OPENAI_API_KEY missing or still placeholder in repository .env.", file=sys.stderr)
                sys.exit(2)
            from openai import OpenAI  # noqa: PLC0415
            client = OpenAI(api_key=key)
        elif provider == "anthropic":
            key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
            if not key or "your_" in key:
                print("[error] ANTHROPIC_API_KEY missing or still placeholder in repository .env.", file=sys.stderr)
                sys.exit(2)
            from anthropic import Anthropic  # noqa: PLC0415
            client = Anthropic(api_key=key)
        elif provider == "azure_openai":
            # Per-model Azure key and endpoint: env var names are derived from
            # the model tag, e.g. tag="kimi-k2.6" → AZURE_KIMI_API_KEY +
            # AZURE_KIMI_ENDPOINT. Keeps multiple Azure deployments isolated.
            tag_upper = config["tag"].upper()
            key_var = f"AZURE_{tag_upper}_API_KEY"
            ep_var = f"AZURE_{tag_upper}_ENDPOINT"
            key = os.environ.get(key_var, "").strip()
            if not key or "your_" in key:
                print(f"[error] {key_var} missing or still placeholder in repository .env.", file=sys.stderr)
                sys.exit(2)
            endpoint = config.get("azure_endpoint") or os.environ.get(ep_var, "").strip()
            if not endpoint:
                print(f"[error] {ep_var} not set in repository .env (or hardcoded azure_endpoint in MODEL_REGISTRY).", file=sys.stderr)
                sys.exit(2)
            from openai import AzureOpenAI  # noqa: PLC0415
            client = AzureOpenAI(
                azure_endpoint=endpoint,
                api_key=key,
                api_version=config["azure_api_version"],
            )
        else:
            print(f"[error] Unknown provider {provider!r} for model {args.model!r}.", file=sys.stderr)
            sys.exit(2)

    # Dataset
    split = _DATASET_CANONICAL_SPLIT[args.dataset]
    print(f"[load] dataset={args.dataset}  split={split}  seed={args.seed}", flush=True)
    items = load_dataset_caveman(name=args.dataset, split=split, n=None, seed=args.seed)
    print(f"[load] {len(items)} items in full split", flush=True)

    # Chunk filter
    if args.start_idx is not None or args.end_idx is not None:
        start = args.start_idx if args.start_idx is not None else 0
        end = args.end_idx if args.end_idx is not None else max((it["idx"] for it in items), default=0)
        items = [it for it in items if start <= it["idx"] <= end]
        print(f"[chunk] idx filter [{start}, {end}]  ->  {len(items)} items", flush=True)

    levels = LEVEL_ORDER if args.level == "all" else [args.level]

    # Write run_config.json BEFORE running so a crash still leaves a record of intent.
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = out_dir / f"run_config_{model_tag}_{args.dataset}_{args.condition}.json"
    config_dump = {
        "model": args.model,
        "model_tag": model_tag,
        "api_model": config["api_model"],
        "provider": provider,
        "dataset": args.dataset,
        "split": split,
        "condition": args.condition,
        "levels": levels,
        "output_dir": str(out_dir),
        "dry_run": args.dry_run,
        "rate_sleep_s": args.rate_sleep_s,
        "max_retries": args.max_retries,
        "start_idx": args.start_idx,
        "end_idx": args.end_idx,
        "seed": args.seed,
        "pricing": {
            "input_per_1M_usd": config["price_in_per_1m"],
            "output_per_1M_usd": config["price_out_per_1m"],
        },
        "constraint_prompts_used": (
            {L: CONSTRAINT_PROMPTS[L] for L in levels}
            if args.condition == "output" else None
        ),
        "neutral_input_system_prompt": (
            NEUTRAL_INPUT_SYSTEM_PROMPT if args.condition == "input" else None
        ),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(config_path, "w") as f:
        json.dump(config_dump, f, indent=2)
    print(f"[config] wrote {config_path}", flush=True)

    # Run each level
    overall_start = time.time()
    cost_per_level: Dict[str, float] = {}
    all_stats: Dict[str, Dict] = {}
    for level in levels:
        s = run_one_level(args, client, level, items, config)
        all_stats[level] = s
        cost_per_level[level] = round(s["total_cost_usd"], 8)

    # Update config with final cost
    config_dump["cost_per_level_usd"] = cost_per_level
    config_dump["total_cost_usd"] = round(sum(cost_per_level.values()), 8)
    config_dump["finished_at"] = datetime.now(timezone.utc).isoformat()
    config_dump["wall_time_s"] = round(time.time(), overall_start, 2)
    with open(config_path, "w") as f:
        json.dump(config_dump, f, indent=2)

    # Final summary
    bar = "=" * 80
    print("\n" + bar)
    print(f"FINAL SUMMARY  model={args.model}  provider={provider}  dataset={args.dataset}  condition={args.condition}")
    print(bar)
    print(f"{'Level':<6} {'N':>6} {'Acc':>7} {'InTok':>9} {'OutTok':>9} {'Cost':>10}")
    print("-" * 80)
    grand = 0.0
    for lvl in levels:
        s = all_stats[lvl]
        acc = s["n_correct"] / max(s["n_processed"], 1)
        grand += s["total_cost_usd"]
        print(
            f"{lvl:<6} {s['n_processed']:>6d} {acc:>7.3f} "
            f"{s['total_input_tokens']:>9d} {s['total_output_tokens']:>9d} "
            f"${s['total_cost_usd']:>8.4f}"
        )
    print("-" * 80)
    print(f"TOTAL COST: ${grand:.4f}")
    print(f"pricing:    in=${config['price_in_per_1m']:.2f}/M  out=${config['price_out_per_1m']:.2f}/M")
    print(f"config:     {config_path}")
    print(bar)


if __name__ == "__main__":
    main()
