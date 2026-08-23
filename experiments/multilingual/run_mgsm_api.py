"""MGSM runner: API models (GPT-4o, Claude Haiku 4.5), Condition B.

Same design as the local runner (run_mgsm_pilot.py): per-language register
prompts from prompts_mgsm.py, numeric extraction with Bengali-digit
normalization, resume-safe per-cell JSONL append. Client construction and
retry come from run_experiment_api.py.

Records carry usage-based token counts and realized cost at the May-2026
snapshot prices in MODEL_REGISTRY.

Outputs:
  results/multilingual/<tag>_output/<lang>/caveman_<tag>_mgsm-<lang>_output_<Lx>.jsonl

Usage (no GPU needed):
  python experiments/multilingual/run_mgsm_api.py --model gpt-4o
  python experiments/multilingual/run_mgsm_api.py --model claude-haiku-4-5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
REPO = _REPO_ROOT
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prompts_mgsm import system_prompt, MAX_NEW_TOKENS  # noqa: E402
from run_mgsm_pilot import extract_number, check_numeric, completed_idx  # noqa: E402
from run_experiment_api import get_model_config, call_with_retry  # noqa: E402


def load_env(path: Path):
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def make_client(provider: str, config: dict, dry_run: bool):
    if dry_run:
        return None
    if provider == "openai":
        from openai import OpenAI
        key = os.environ.get("OPENAI_API_KEY", "").strip()
        assert key and "your_" not in key, "OPENAI_API_KEY missing"
        return OpenAI(api_key=key)
    if provider == "anthropic":
        from anthropic import Anthropic
        key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        assert key and "your_" not in key, "ANTHROPIC_API_KEY missing"
        return Anthropic(api_key=key)
    raise ValueError(provider)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True,
                    help="Registry name: gpt-4o | claude-haiku-4-5")
    ap.add_argument("--langs", default="de,zh,bn,sw")
    ap.add_argument("--levels", default="L0,L1,L2")
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--max_retries", type=int, default=5)
    ap.add_argument("--output_root", default=str(REPO / "results" / "multilingual"))
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    load_env(REPO / ".env")
    config = get_model_config(args.model)
    provider, tag = config["provider"], config["tag"]
    api_model = config["api_model"]
    p_in = config["price_in_per_1m"] / 1e6
    p_out = config["price_out_per_1m"] / 1e6
    client = make_client(provider, config, args.dry_run)

    from datasets import load_dataset
    langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    levels = [x.strip() for x in args.levels.split(",") if x.strip()]
    data = {lang: load_dataset("juletxara/mgsm", lang, split="test") for lang in langs}

    total_cost = 0.0
    for lang in langs:
        ds = data[lang]
        for level in levels:
            out_dir = Path(args.output_root) / f"{tag}_output" / lang
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"caveman_{tag}_mgsm-{lang}_output_{level}.jsonl"
            done = completed_idx(out_path)
            sp = system_prompt(level, lang)
            max_new = MAX_NEW_TOKENS[level]
            (out_dir / f"run_config_{level}.json").write_text(json.dumps({
                "model_name": tag, "api_model": api_model, "provider": provider,
                "dataset": f"mgsm-{lang}", "condition": "output", "level": level,
                "n": min(args.n, len(ds)), "system_prompt": sp,
                "max_tokens": max_new, "temperature": 0.0,
                "price_in_per_1m": config["price_in_per_1m"],
                "price_out_per_1m": config["price_out_per_1m"],
                "written_at": datetime.now(timezone.utc).isoformat(),
            }, indent=2), encoding="utf-8")

            n_new = 0
            t0 = time.time()
            with open(out_path, "a", encoding="utf-8") as fout:
                for idx in range(min(args.n, len(ds))):
                    if idx in done:
                        continue
                    item = ds[idx]
                    out_text, tok_in, tok_out, api_model_ret, lat = call_with_retry(
                        provider=provider, client=client, api_model=api_model,
                        system_prompt=sp, user_message=item["question"],
                        max_tokens=max_new, max_retries=args.max_retries,
                        dry_run=args.dry_run)
                    pred = extract_number(out_text or "")
                    cost = tok_in * p_in + tok_out * p_out
                    total_cost += cost
                    rec = {
                        "idx": idx, "lang": lang, "level": level,
                        "condition": "output", "model": tag,
                        "api_model": api_model_ret,
                        "question": item["question"],
                        "gt_answer": item["answer_number"],
                        "output": out_text,
                        "predicted_answer": pred,
                        "correct": check_numeric(pred, item["answer_number"]),
                        "input_tokens": tok_in, "output_tokens": tok_out,
                        "cost_usd": round(cost, 8),
                        "latency_s": round(lat, 3),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fout.flush()
                    n_new += 1
                    if n_new % 50 == 0:
                        el = time.time() - t0
                        print(f"[{tag} {lang} {level}] {n_new} done "
                              f"({el/max(n_new,1):.1f}s/item, cum cost ${total_cost:.2f})",
                              flush=True)
            print(f"[cell {tag}/{lang}/{level}] complete, new={n_new}, "
                  f"cum cost ${total_cost:.2f}", flush=True)
    print(f"[mgsm api] ALL DONE  total new-call cost ${total_cost:.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
