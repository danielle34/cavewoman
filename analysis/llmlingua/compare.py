"""Step 5, dissociation tables + LLMLingua-vs-caveman comparison (single level).

Treats "LLMLingua" as one method/level peer to caveman's L0/L1/L2/L3/L4.

Reads:
    LLMLingua scored outputs (one level per (model, dataset)):
        results/llmlingua/inference/<model>/<dataset>/<dataset>_LLMLingua_scored.jsonl
    Cavewoman Condition-A entailment-augmented outputs (L1, L2, same input
    compression condition, but using POS-filter compression at different
    aggressiveness levels):
        results/<model>_input/<dataset>/caveman_<model>_<dataset>_input_L{1,2}_with_entailment.jsonl

Builds:
    results/llmlingua/dissociation_table.csv          (LLMLingua only, one row per model x dataset)
    results/llmlingua/caveman_dissociation_table.csv  (caveman L1/L2)
    results/llmlingua/method_comparison_table.csv     (LLMLingua + caveman side-by-side)
    results/llmlingua/summary_for_paper.csv           (cross-cell mean per method/level)
    figures/llmlingua/c2_comparison_llmlingua_vs_caveman.{pdf,png}
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[2]

INF_BASE = REPO / "results" / "llmlingua" / "inference"
CAV_BASE = REPO / "results"
OUT_DIR = REPO / "results" / "llmlingua"
FIG_DIR = REPO / "figures" / "llmlingua"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["gpt-4o", "sonnet-4.6", "qwen-2.5"]
DATASETS = ["gsm8k", "boolq", "arc_easy"]
LLM_LEVELS = ["LLMLingua", "LLMLingua_t0.8"]  # tau=0.5 (all models) + tau=0.8 (qwen only)
CAV_LEVELS = ["L1", "L2"]


def read_jsonl(p: Path) -> list:
    if not p.exists():
        return []
    return [json.loads(line) for line in p.open()]


def load_llmlingua_scored() -> pd.DataFrame:
    rows = []
    for m in MODELS:
        for d in DATASETS:
            for lv in LLM_LEVELS:
                p = INF_BASE / m / d / f"{d}_{lv}_scored.jsonl"
                if not p.exists():
                    continue
                # Each level surfaces as its own "method" so the comparison
                # figure can show LLMLingua-τ=0.5 and LLMLingua-τ=0.8 side by
                # side as peer methods to caveman L1/L2.
                method_label = "LLMLingua (τ=0.5)" if lv == "LLMLingua" else f"LLMLingua (τ=0.8)"
                for r in read_jsonl(p):
                    rows.append({
                        "method": method_label,
                        "model": m,
                        "dataset": d,
                        "level": lv,
                        "idx": r["idx"],
                        "accuracy": int(bool(r.get("correct", False))),
                        "bidirectional_entailment": bool(r.get("bidirectional_entailment", False)),
                        "cosine_similarity": float(r.get("cosine_similarity_l0", 0.0)),
                    })
    return pd.DataFrame(rows)


def load_caveman() -> pd.DataFrame:
    rows = []
    for m in MODELS:
        for d in DATASETS:
            for lv in CAV_LEVELS:
                p = (CAV_BASE / f"{m}_input" / d /
                     f"caveman_{m}_{d}_input_{lv}_with_entailment.jsonl")
                for r in read_jsonl(p):
                    bi = r.get("bidirectional_entailment", r.get("nli_bidirectional_entailment"))
                    cos = r.get("cosine_similarity") or r.get("cosine_sim") or 0.0
                    rows.append({
                        "method": "Cavewoman (POS filter)",
                        "model": m,
                        "dataset": d,
                        "level": lv,
                        "idx": r["idx"],
                        "accuracy": int(bool(r.get("correct", False))),
                        "bidirectional_entailment": bool(bi) if bi is not None else False,
                        "cosine_similarity": float(cos),
                    })
    return pd.DataFrame(rows)


def add_dissociation_cells(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    a = df["accuracy"].astype(bool)
    b = df["bidirectional_entailment"].astype(bool)
    df = df.copy()
    df["C1"] = (a & b).astype(int)
    df["C2"] = (a & ~b).astype(int)
    df["C3"] = (~a & b).astype(int)
    df["C4"] = (~a & ~b).astype(int)
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df.groupby(["method", "model", "dataset", "level"]).agg(
        n=("idx", "count"),
        accuracy=("accuracy", "mean"),
        nli_rate=("bidirectional_entailment", "mean"),
        cosine_mean=("cosine_similarity", "mean"),
        C1=("C1", "mean"),
        C2=("C2", "mean"),
        C3=("C3", "mean"),
        C4=("C4", "mean"),
    ).reset_index()


def make_c2_figure(summary: pd.DataFrame) -> None:
    """C2 by (model, dataset), bars per (method, level): one LLMLingua bar +
    two caveman bars (L1, L2)."""
    rows = MODELS
    cols = DATASETS
    fig, axes = plt.subplots(len(rows), len(cols),
                             figsize=(4 * len(cols), 3.2 * len(rows)),
                             sharey=True)
    colors = {
        ("Cavewoman (POS filter)", "L1"): "#85c1e9",
        ("Cavewoman (POS filter)", "L2"): "#3498db",
        ("LLMLingua (τ=0.5)", "LLMLingua"): "#e74c3c",
        ("LLMLingua (τ=0.8)", "LLMLingua_t0.8"): "#f39c12",
    }
    bar_order = [
        ("Cavewoman (POS filter)", "L1", "Cav L1"),
        ("Cavewoman (POS filter)", "L2", "Cav L2"),
        ("LLMLingua (τ=0.5)", "LLMLingua", "LLM τ=.5"),
        ("LLMLingua (τ=0.8)", "LLMLingua_t0.8", "LLM τ=.8"),
    ]
    for i, m in enumerate(rows):
        for j, d in enumerate(cols):
            ax = axes[i][j] if len(rows) > 1 else axes[j]
            sub = summary[(summary["model"] == m) & (summary["dataset"] == d)]
            xs, ys, cs, labs = [], [], [], []
            for k, (method, level, label) in enumerate(bar_order):
                s = sub[(sub["method"] == method) & (sub["level"] == level)]
                v = float(s["C2"].values[0]) if len(s) else 0.0
                xs.append(k); ys.append(v); cs.append(colors[(method, level)]); labs.append(label)
            ax.bar(xs, ys, color=cs)
            ax.set_xticks(xs)
            ax.set_xticklabels(labs, fontsize=8)
            ax.set_title(f"{m}, {d}", fontsize=10)
            ax.set_ylim(0, 1)
            ax.axhline(0.1, color="gray", linestyle=":", alpha=0.5)
            if j == 0:
                ax.set_ylabel("C2 rate\n(correct + non-entailing)", fontsize=9)
    fig.suptitle("Dissociation (C2) under LLMLingua vs Cavewoman compression\n"
                 "Condition A (input compression)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"c2_comparison_llmlingua_vs_caveman.{ext}",
                    dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df_ll = add_dissociation_cells(load_llmlingua_scored())
    df_cv = add_dissociation_cells(load_caveman())
    if df_ll.empty:
        print("[compare] No LLMLingua scored outputs found. Run score.py first.")
        return

    s_ll = summarize(df_ll)
    s_cv = summarize(df_cv)
    s_ll.to_csv(OUT_DIR / "dissociation_table.csv", index=False)
    s_cv.to_csv(OUT_DIR / "caveman_dissociation_table.csv", index=False)

    cmp = pd.concat([s_ll, s_cv], ignore_index=True)
    cmp = cmp.sort_values(["model", "dataset", "method", "level"])
    cmp.to_csv(OUT_DIR / "method_comparison_table.csv", index=False)
    print("=== METHOD COMPARISON ===")
    print(cmp.to_string(index=False))

    cross = cmp.groupby(["method", "level"]).agg(
        mean_accuracy=("accuracy", "mean"),
        mean_nli=("nli_rate", "mean"),
        mean_C2=("C2", "mean"),
        mean_cosine=("cosine_mean", "mean"),
        cells=("n", "sum"),
    ).reset_index()
    cross.to_csv(OUT_DIR / "summary_for_paper.csv", index=False)
    print("\n=== SUMMARY (cross-cell mean per method/level) ===")
    print(cross.to_string(index=False))

    make_c2_figure(cmp)
    print(f"\nFigure written: {FIG_DIR}/c2_comparison_llmlingua_vs_caveman.{{pdf,png}}")


if __name__ == "__main__":
    main()
