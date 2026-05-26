"""Step 1, compute matched LLMLingua compression ratios from existing data.

For each (model, dataset) in {gpt-4o, qwen-2.5} × {gsm8k, boolq, arc_easy},
read the Condition-A (condition='input') L0, L1, L2 JSONL outputs and compute
the mean input_tokens at each level. Then derive the L1 and L2 token-retention
ratio τ = mean(input_tokens_Lx) / mean(input_tokens_L0). These τ values are
what we pass to LLMLingua-2's compress_prompt(rate=τ) so that LLMLingua's
token reduction matches caveman's POS-filter token reduction.

Outputs:
  results/llmlingua/target_compression_ratios.csv  (per cell)
  results/llmlingua/compression_targets.json       (cross-cell meanused by compress.py)

This script is fast (<1 min). Run locally:
    python analysis/llmlingua/ratios.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO / "results"
OUT_DIR = REPO / "results" / "llmlingua"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["gpt-4o", "qwen-2.5"]
DATASETS = ["gsm8k", "boolq", "arc_easy"]
LEVELS = ["L0", "L1", "L2"]
CONDITION = "input"  # paper's Condition A


def read_input_tokens(model: str, dataset: str, level: str) -> list[int]:
    """Read input_tokens column from one (model, dataset, level) JSONL."""
    path = (
        RESULTS_DIR
        / f"{model}_{CONDITION}"
        / dataset
        / f"caveman_{model}_{dataset}_{CONDITION}_{level}.jsonl"
    )
    if not path.exists():
        raise FileNotFoundError(f"Missing baseline file: {path}")
    tokens = []
    with path.open() as f:
        for line in f:
            rec = json.loads(line)
            tokens.append(int(rec["input_tokens"]))
    return tokens


def main() -> None:
    rows = []
    for model in MODELS:
        for ds in DATASETS:
            level_means = {}
            for level in LEVELS:
                toks = read_input_tokens(model, ds, level)
                level_means[level] = sum(toks) / len(toks)
            l0 = level_means["L0"]
            for level in LEVELS:
                rows.append(
                    {
                        "model": model,
                        "dataset": ds,
                        "level": level,
                        "mean_input_tokens": round(level_means[level], 2),
                        "tau_fraction_kept": round(level_means[level] / l0, 4),
                        "compression_ratio_1_over_tau": round(l0 / level_means[level], 3),
                    }
                )

    df = pd.DataFrame(rows)
    out_csv = OUT_DIR / "target_compression_ratios.csv"
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")
    print(df.to_string(index=False))

    # Cross-cell means at L1 and L2, these are what we pass to LLMLingua.
    l1_tau = df.loc[df["level"] == "L1", "tau_fraction_kept"].mean()
    l2_tau = df.loc[df["level"] == "L2", "tau_fraction_kept"].mean()
    targets = {
        "l1_tau_fraction_kept": round(float(l1_tau), 4),
        "l2_tau_fraction_kept": round(float(l2_tau), 4),
        "l1_compression_ratio": round(1.0 / float(l1_tau), 3),
        "l2_compression_ratio": round(1.0 / float(l2_tau), 3),
        "note": (
            "tau = fraction of tokens KEPT; pass as 'rate' to "
            "PromptCompressor.compress_prompt (per LLMLingua-2 paper Sec. 4.2)."
        ),
    }
    out_json = OUT_DIR / "compression_targets.json"
    out_json.write_text(json.dumps(targets, indent=2))
    print()
    print(f"Wrote {out_json}")
    print(json.dumps(targets, indent=2))


if __name__ == "__main__":
    main()
