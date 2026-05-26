"""Aggregate descriptive statistics by (model, dataset, condition, level).

Computes mean, std, median, and bootstrap CI for:
  accuracy, semantic_collapse_rate, bidirectional_entailment_rate,
  embedding_similarity, input_tokens, output_tokens, total_tokens, cost_usd.

Writes per-cell tables to tables/ and LaTeX-ready main table to latex/.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from _lib import (
    Config, add_common_args, apply_common_overrides, load_config,
    setup_logging, setup_output_dirs, load_all, add_derived_columns,
    bootstrap_ci, write_csv, write_latex,
)


METRICS = [
    # (key, lambda(df, cfg) -> series, label)
    ("accuracy",                lambda df, cfg: df[cfg.col("correct")].astype(float),
                                "Accuracy"),
    ("semantic_collapse_rate",  lambda df, cfg: df.get("semantic_collapse",
                                                       pd.Series(dtype=float)).astype(float),
                                "Sem. collapse"),
    ("bidir_entailment_rate",   lambda df, cfg: df[cfg.col("bidirectional_entailment")]
                                                  .astype(float)
                                if cfg.col("bidirectional_entailment") in df.columns
                                else pd.Series(dtype=float),
                                "Bi-entail."),
    ("embedding_similarity",    lambda df, cfg: df[cfg.col("embedding_similarity")]
                                                  .astype(float)
                                if cfg.col("embedding_similarity") in df.columns
                                else pd.Series(dtype=float),
                                "Cos. sim."),
    ("input_tokens",            lambda df, cfg: df[cfg.col("input_tokens")].astype(float),
                                "Input tok."),
    ("output_tokens",           lambda df, cfg: df[cfg.col("output_tokens")].astype(float),
                                "Output tok."),
    ("total_tokens",            lambda df, cfg: df.get("total_tokens",
                                                       pd.Series(dtype=float)).astype(float),
                                "Total tok."),
    ("cost_usd",                lambda df, cfg: df.get("cost_usd_safe",
                                                       pd.Series(dtype=float)).astype(float),
                                "Cost ($)"),
]


def cell_summary(values: Sequence[float], *, n_boot: int, ci: float,
                 rng: np.random.Generator) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return {"n": 0, "mean": np.nan, "std": np.nan, "median": np.nan,
                "ci_lo": np.nan, "ci_hi": np.nan}
    point, lo, hi = bootstrap_ci(arr, n=n_boot, ci=ci, rng=rng)
    return {
        "n": int(arr.size), "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "median": float(np.median(arr)),
        "ci_lo": lo, "ci_hi": hi,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    return add_common_args(ap).parse_args()


def main() -> int:
    args = parse_args()
    cfg = apply_common_overrides(load_config(args.config), args)
    logger = setup_logging(cfg, "descriptive_stats", verbose=args.verbose)
    out = setup_output_dirs(cfg)

    df = load_all(cfg, logger=logger)
    if df.empty:
        logger.error("no data loaded, abort")
        return 1
    df = add_derived_columns(df, cfg)

    rng = np.random.default_rng(cfg.stats["rng_seed"])
    n_boot = int(cfg.stats["bootstrap_n"])
    ci = float(cfg.stats["bootstrap_ci"])

    rows: List[Dict[str, Any]] = []
    level_col = cfg.col("level")
    for (m, ds, c, L), grp in df.groupby(["__model", "__dataset", "__condition", level_col]):
        base = {"model": m, "dataset": ds, "condition": c, "level": L,
                "n_records": len(grp)}
        for key, extractor, _ in METRICS:
            series = extractor(grp, cfg)
            if series is None or len(series) == 0:
                continue
            s = cell_summary(series, n_boot=n_boot, ci=ci, rng=rng)
            for stat, val in s.items():
                base[f"{key}__{stat}"] = val
        rows.append(base)

    out_df = pd.DataFrame(rows)
    write_csv(out_df, out["tables"] / "descriptive_per_cell.csv", logger=logger)

    # ----------------- compact LaTeX-friendly main table ---------------------
    # Restrict to accuracy + semantic_collapse + bidir entail + emb sim, pivoting level
    main_cols = ["accuracy", "semantic_collapse_rate", "bidir_entailment_rate",
                 "embedding_similarity"]
    pivot_frames = []
    for k in main_cols:
        sub = out_df[["model", "dataset", "condition", "level", f"{k}__mean"]].copy()
        sub.rename(columns={f"{k}__mean": k}, inplace=True)
        sub["metric"] = k
        sub.rename(columns={k: "value"}, inplace=True)
        pivot_frames.append(sub)
    if pivot_frames:
        long = pd.concat(pivot_frames, ignore_index=True)
        wide = long.pivot_table(
            index=["model", "dataset", "condition", "metric"],
            columns="level", values="value", aggfunc="first",
        ).reset_index()
        wide = wide[["model", "dataset", "condition", "metric"]
                    + [L for L in cfg.levels if L in wide.columns]]
        write_csv(wide, out["tables"] / "descriptive_main_wide.csv", logger=logger)
        write_latex(wide, out["latex"] / "tab_descriptive_main.tex",
                    caption="Descriptive statistics by (model, dataset, condition) "
                            "across constraint levels.",
                    label="tab:descriptive-main", float_format="%.3f", logger=logger)

    # ----------------- per-(model, condition) summaries ----------------------
    agg = (out_df.groupby(["model", "condition", "level"])
                 .agg(accuracy_mean=("accuracy__mean", "mean"),
                      sem_collapse_mean=("semantic_collapse_rate__mean", "mean"),
                      entail_mean=("bidir_entailment_rate__mean", "mean"),
                      cos_mean=("embedding_similarity__mean", "mean"),
                      out_tok_mean=("output_tokens__mean", "mean"),
                      cost_sum=("cost_usd__mean",
                                lambda s: float(np.nansum(s.astype(float))))
                 ).reset_index())
    write_csv(agg, out["tables"] / "descriptive_per_model_condition.csv", logger=logger)
    logger.info(f"computed descriptives for {len(out_df)} cells "
                f"across {out_df['model'].nunique()} model(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
