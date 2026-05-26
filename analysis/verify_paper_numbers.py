"""Verify paper numbers.

Spot-check every numeric claim in the paper against the released CSVs.

Run:  python analysis/verify_paper_numbers.py

Pure pandas/json over committed CSVs and per-cell analysis_numbers.json files.
No GPU, no network. Prints {paper value, current value, status} for each claim
so paper edits in Lane C can be driven directly from this output.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MASTER = ROOT / "results/tables/master_cell_summary.csv"  # See README; provide your own per-cell summary CSV.
RESULTS = ROOT / "results"


def fmt_row(label: str, paper, current, tol: float = 0.01) -> str:
    if isinstance(paper, (int, float)) and isinstance(current, (int, float)):
        ok = abs(paper, current) <= tol
        tag = "OK  " if ok else "DIFF"
        return f"[{tag}] {label:55s}  paper={paper:>8}  current={current:>8.4g}"
    return f"[INFO] {label:55s}  paper={paper!s:>8}  current={current!s:>8}"


def main() -> None:
    df = pd.read_csv(MASTER)
    print("=" * 90)
    print("B5, Spot-check paper numbers vs current master_cell_summary.csv")
    print("=" * 90)

    # ----- Finding 2: per-model C2 rates -----
    # Reconciliation: paper Figure 5 caption says "Cond B, L1-L3, all benchmarks".
    # The matching aggregation is item-weighted (micro), not unweighted (macro).
    # Paper line 606's prose "at L1-B" is slightly misleading, the numbers are L1-L3
    # under Cond B (item-weighted), not L1 alone.
    print("\n--- Finding 2: per-model C2 rate (L1-L3 Cond B, item-weighted micro mean) ---")
    paper_finding2 = {
        "haiku-4.5": 36.1, "qwen-2.5": 48.4, "gpt-4o": 50.3, "gpt-5.4": 50.9, "sonnet-4.6": 57.2,
    }
    sub = df[df.level.isin(["L1", "L2", "L3"]) & (df.condition == "B_output_constraint") & df.c2_correct_nonentail.notna()]
    for model, paper_pct in paper_finding2.items():
        cell = sub[sub.model == model]
        if cell.empty:
            cur = float("nan")
        else:
            cur = ((cell.c2_correct_nonentail * cell.n_items).sum() / cell.n_items.sum()) * 100
        print(fmt_row(f"C2 rate {model} (L1-L3 Cond B micro)", paper_pct, cur, tol=0.15))

    # ----- Finding 2: Sonnet C2, C3 at L1-B -----
    print("\n--- Finding 2: Sonnet C2, C3 at L1-B ---")
    sonnet_l1b = df[(df.model == "sonnet-4.6") & (df.level == "L1") & (df.condition == "B_output_constraint")]
    c2 = sonnet_l1b.c2_correct_nonentail.mean()
    c3 = sonnet_l1b.c3_incorrect_entail.mean()
    print(fmt_row("Sonnet C2-C3 at L1-B (paper +0.555)", 0.555, (c2, c3) if pd.notna(c2, c3) else float("nan"), tol=0.02))
    print(f"        Sonnet C2 at L1-B = {c2:.4f}, C3 = {c3:.4f}")

    # ----- Finding 1 / Discussion P1: L1 cost mean and max across 4 API models -----
    # NOTE: the CSV column percent_cost_reduction_vs_L0 is positive when cost decreased
    # (e.g. 33 means 33 % cheaper). The paper writes it with an explicit minus sign
    # ("mean -33 %") to mean "cost dropped by 33 %". So 33.x % current = "-33 %" paper.
    print("\n--- Finding 1: L1 cost reductions across the four API models, Cond B (sign convention: +X% in CSV = paper '-X%') ---")
    api_models = ["gpt-4o", "gpt-5.4", "haiku-4.5", "sonnet-4.6"]
    l1b_api = df[(df.level == "L1") & (df.condition == "B_output_constraint") & df.model.isin(api_models)]
    cost_mean = l1b_api.percent_cost_reduction_vs_L0.mean()
    print(fmt_row("Mean L1 cost reduction Cond B (paper 33 %)", 33.0, cost_mean, tol=2.0))

    # Paper "up to 3.1x" cost reduction. 3.1x means cost/3.1 -> 1, 1/3.1 = 67.7 % reduction.
    # Restrict to L1 only (the headline ramp) Cond B (output channel).
    print("\n  Max L1 Cond B cost reduction across all models (paper 'up to 3.1x' = ~67.7 % cut):")
    l1b_all = df[(df.level == "L1") & (df.condition == "B_output_constraint")]
    cost_max_l1b = l1b_all.percent_cost_reduction_vs_L0.max()
    print(fmt_row("Max L1-B cost reduction any model (paper 67.7 %)", 67.7, cost_max_l1b, tol=4.0))
    top5 = l1b_all.nlargest(5, "percent_cost_reduction_vs_L0")[
        ["model", "dataset", "condition", "level", "percent_cost_reduction_vs_L0", "delta_accuracy_vs_L0"]
    ]
    for _, r in top5.iterrows():
        print(f"           {r.model:8s} {r.dataset:13s} {r.condition[0]} {r.level}  cost%={r.percent_cost_reduction_vs_L0:6.2f}  Δacc={r.delta_accuracy_vs_L0:+.3f}")

    # ----- Discussion P1 / Conclusion: GPT-4o GSM8K L1-B (+4.1 pp acc, -51.8 % cost) -----
    print("\n--- GPT-4o GSM8K L1-B headline cell ---")
    cell = df[(df.model == "gpt-4o") & (df.dataset == "gsm8k") & (df.condition == "B_output_constraint") & (df.level == "L1")]
    if not cell.empty:
        r = cell.iloc[0]
        print(fmt_row("GPT-4o GSM8K L1-B Δacc (paper +0.041)", 0.041, r.delta_accuracy_vs_L0, tol=0.005))
        print(fmt_row("GPT-4o GSM8K L1-B cost% (paper 51.8 reduction)", 51.8, r.percent_cost_reduction_vs_L0, tol=1.0))

    # ----- Finding 1 / Discussion P1: MMLU-STEM parse rates 0.788 / 0.955 -----
    print("\n--- MMLU-STEM parse rates (GPT-4o; paper 0.788 / 0.955) ---")
    p = RESULTS / "gpt-4o_output/mmlu_stem/analysis_numbers.json"
    if p.exists():
        d = json.loads(p.read_text())
        by_lvl = {x["level"]: x for x in d.get("per_level", [])}
        for lvl, paper_val in [("L0", 0.788), ("L1", 0.955)]:
            cur = by_lvl.get(lvl, {}).get("ans_extract_rate")
            print(fmt_row(f"MMLU GPT-4o output {lvl} ans_extract_rate", paper_val, cur, tol=0.01))
    else:
        # paper specifies output condition for the 0.788 vs 0.955 contrast; try input as well
        print(f"        ERR: {p} not found")
    # Cross-check with Cond B (input compression)
    p_input = RESULTS / "gpt-4o_input/mmlu_stem/analysis_numbers.json"
    if p_input.exists():
        d = json.loads(p_input.read_text())
        by_lvl = {x["level"]: x for x in d.get("per_level", [])}
        for lvl in ("L0", "L1"):
            cur = by_lvl.get(lvl, {}).get("ans_extract_rate")
            print(f"        [aux] MMLU GPT-4o INPUT {lvl} ans_extract_rate = {cur}")

    # ----- Finding 4 / Discussion P3: GPT-5.4 L1-A mean accuracy = 0.583 -----
    print("\n--- GPT-5.4 L1-A mean accuracy across datasets (paper 0.583) ---")
    cell = df[(df.model == "gpt-5.4") & (df.condition == "A_input_compression") & (df.level == "L1")]
    cur = cell.accuracy.mean()
    print(fmt_row("gpt-5.4 L1-A mean accuracy", 0.583, cur, tol=0.005))

    # ----- DeepSeek inversion paragraphs -----
    # Paper line 606 (Finding 2 prose):  C2=11.9%, C3=27.2%  (probably L1-L3 Cond B agg)
    # Paper line ~644 (Finding 4 DeepSeek paragraph): 0.175 / 0.227 / cos 0.892 / NLI 0.434 (L1-B)
    print("\n--- DeepSeek L1-B (Finding 4 paragraph: C2=0.175 C3=0.227 cos=0.892 NLI=0.434) ---")
    cell = df[(df.model == "deepseek-r1") & (df.condition == "B_output_constraint") & (df.level == "L1")]
    print(fmt_row("DeepSeek L1-B C2", 0.175, cell.c2_correct_nonentail.mean(), tol=0.01))
    print(fmt_row("DeepSeek L1-B C3", 0.227, cell.c3_incorrect_entail.mean(), tol=0.01))
    print(fmt_row("DeepSeek L1-B cosine", 0.892, cell.embedding_similarity_mean.mean(), tol=0.01))
    print(fmt_row("DeepSeek L1-B NLI bidir", 0.434, cell.bidirectional_entailment_mean.mean(), tol=0.01))

    print("\n--- DeepSeek L1-L3 Cond B (Finding 2 prose 0.119 / 0.272; item-weighted) ---")
    sub = df[(df.model == "deepseek-r1") & (df.condition == "B_output_constraint") & df.level.isin(["L1", "L2", "L3"])]
    c2_micro = (sub.c2_correct_nonentail * sub.n_items).sum() / sub.n_items.sum()
    c3_micro = (sub.c3_incorrect_entail * sub.n_items).sum() / sub.n_items.sum()
    print(fmt_row("DeepSeek L1-L3 Cond B C2 (paper 0.119)", 0.119, c2_micro, tol=0.005))
    print(fmt_row("DeepSeek L1-L3 Cond B C3 (paper 0.272)", 0.272, c3_micro, tol=0.005))

    # ----- Appendix: ceiling-hit rates 0.7 / 5.7 / 0.7 (paper line ~687) -----
    print("\n--- Ceiling-hit rates at L1 across 5 datasets (paper 0.7% GPT-4o / 5.7% Haiku / 0.7% Qwen2.5) ---")
    ceil = ROOT / "analysis/cavewoman_strengthening/outputs/section_4/4c_ceiling_hits.csv"
    if ceil.exists():
        cdf = pd.read_csv(ceil)
        # Paper aggregates: mean ceiling-hit rate at L1 across 5 datasets per model.
        l1 = cdf[cdf.level == "L1"]
        for model, paper_val, name in [("gpt-4o", 0.7, "GPT-4o"), ("haiku-4.5", 5.7, "Haiku"), ("qwen-2.5", 0.7, "Qwen2.5")]:
            cell = l1[l1.model == model]
            cur = cell.ceiling_hit_rate.mean() if not cell.empty else float("nan")
            print(fmt_row(f"{name} L1 ceiling-hit % mean over datasets", paper_val, cur, tol=0.1))
    else:
        print(f"        ERR: {ceil} not found")

    print("\n" + "=" * 90)
    print("Done. Rows tagged [OK] need no paper edit; [DIFF] rows need to be reconciled.")
    print("=" * 90)


if __name__ == "__main__":
    main()
