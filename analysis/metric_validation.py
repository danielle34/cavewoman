"""Metric agreement validation: bidirectional NLI entailment vs cosine similarity.

Per (model, dataset, condition, level), computes:
, Pearson r and Spearman rho between NLI rate and cosine similarity
, Kappa-style agreement using the cosine-collapse threshold to binarize cosine
, Disagreement examples (top-K saved):
      high_cos_no_entail  (cosine high, NLI says NOT entailment)
      low_cos_entail      (cosine low, NLI says entailment)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import scipy.stats as stats

from _lib import (
    Config, add_common_args, apply_common_overrides, load_config,
    setup_logging, setup_output_dirs, load_all, add_derived_columns,
    write_csv, write_json,
)


def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's kappa for two binary 0/1 sequences."""
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask].astype(int), b[mask].astype(int)
    if a.size == 0:
        return float("nan")
    po = float((a == b).mean())
    pa = a.mean(); pb = b.mean()
    pe = pa * pb + (1, pa) * (1, pb)
    return float((po, pe) / (1, pe)) if pe < 1 else float("nan")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--top-k", type=int, default=20,
                    help="Number of disagreement examples to save per direction")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    cfg = apply_common_overrides(load_config(args.config), args)
    logger = setup_logging(cfg, "metric_validation", verbose=args.verbose)
    out = setup_output_dirs(cfg)

    df = load_all(cfg, logger=logger)
    if df.empty:
        logger.error("no data loaded"); return 1
    df = add_derived_columns(df, cfg)

    entail = cfg.col("bidirectional_entailment")
    cos = cfg.col("embedding_similarity")
    cos_thr = float(cfg.thresholds["semantic_collapse_cosine"])

    if entail not in df.columns or cos not in df.columns:
        logger.error("entailment or cosine column missing, cannot validate")
        return 1

    sub = df.dropna(subset=[entail, cos]).copy()
    sub["nli_bin"] = sub[entail].astype(int)
    sub["cos_bin"] = (sub[cos].astype(float) >= cos_thr).astype(int)

    # Per-cell correlations
    rows = []
    level_col = cfg.col("level")
    for (m, ds, c, L), grp in sub.groupby(["__model", "__dataset", "__condition", level_col]):
        nli = grp["nli_bin"].astype(float).to_numpy()
        cs = grp[cos].astype(float).to_numpy()
        if nli.size < 5 or np.unique(nli).size < 2 or np.unique(cs).size < 2:
            continue
        try:
            pearson = stats.pearsonr(nli, cs)
            spearman = stats.spearmanr(nli, cs)
        except Exception:
            continue
        kappa = cohen_kappa(grp["nli_bin"].to_numpy(),
                            grp["cos_bin"].to_numpy())
        rows.append({
            "model": m, "dataset": ds, "condition": c, "level": L,
            "n": int(nli.size),
            "pearson_r": float(pearson[0]), "pearson_p": float(pearson[1]),
            "spearman_rho": float(spearman[0]), "spearman_p": float(spearman[1]),
            "cohen_kappa": kappa,
            "cos_thr_used": cos_thr,
            "nli_rate": float(grp["nli_bin"].mean()),
            "cos_pass_rate": float(grp["cos_bin"].mean()),
        })
    corr_df = pd.DataFrame(rows)
    write_csv(corr_df, out["tables"] / "nli_cosine_agreement.csv", logger=logger)

    # Disagreement examples
    disagree_hi_lo: List[Dict[str, Any]] = []
    disagree_lo_hi: List[Dict[str, Any]] = []
    keep_cols = [cfg.col("idx"), "__model", "__dataset", "__condition", level_col,
                 cfg.col("output")[:0] or cfg.col("output"),
                 entail, cos]
    keep_cols = [c for c in keep_cols if c and c in sub.columns]

    # high cosine, no entailment
    high_no_ent = sub[(sub[cos] >= 0.7) & (sub["nli_bin"] == 0)].copy()
    if not high_no_ent.empty:
        high_no_ent = high_no_ent.sort_values(cos, ascending=False).head(args.top_k)
        disagree_hi_lo = high_no_ent[keep_cols].to_dict("records")
    # low cosine, entailment yes
    low_ent = sub[(sub[cos] < cos_thr) & (sub["nli_bin"] == 1)].copy()
    if not low_ent.empty:
        low_ent = low_ent.sort_values(cos, ascending=True).head(args.top_k)
        disagree_lo_hi = low_ent[keep_cols].to_dict("records")

    write_json({"high_cosine_no_entail": disagree_hi_lo,
                "low_cosine_yes_entail": disagree_lo_hi,
                "cos_thr": cos_thr},
               out["stats"] / "disagreement_examples.json", logger=logger)

    # Overall agreement summary
    summary = {
        "n_cells": int(len(corr_df)),
        "median_pearson_r": float(corr_df["pearson_r"].median()) if not corr_df.empty else None,
        "median_spearman_rho": float(corr_df["spearman_rho"].median()) if not corr_df.empty else None,
        "median_kappa": float(corr_df["cohen_kappa"].median()) if not corr_df.empty else None,
        "n_disagree_high_cos_no_ent": len(disagree_hi_lo),
        "n_disagree_low_cos_yes_ent": len(disagree_lo_hi),
    }
    write_json(summary, out["stats"] / "metric_validation_summary.json", logger=logger)
    logger.info(f"median Pearson r = {summary['median_pearson_r']}, "
                f"median Spearman rho = {summary['median_spearman_rho']}, "
                f"median kappa = {summary['median_kappa']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
