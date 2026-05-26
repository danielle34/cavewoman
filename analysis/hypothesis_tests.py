"""Hypothesis tests for CAVEWOMAN:

  H1  Semantic collapse occurs before accuracy collapse (paired bootstrap).
  H2  Input vs output condition asymmetry at matched (model, dataset, level).
  H3  Dissociation 2×2: C2 (correct & semantically collapsed) > 0 (binomial)
       and C2 dominates off-diagonal failures.

Writes per-test JSON + a combined CSV with corrected p-values.
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
    bootstrap_ci, paired_bootstrap_diff, cohens_d,
    adjust_pvalues, write_csv, write_json,
)


# ---------------------------- H1 -------------------------------------------
def h1_semantic_before_accuracy(thresholds_df: pd.DataFrame) -> Dict[str, Any]:
    """Across lanes, test whether L_c_sem < L_c_acc (semantic earlier)."""
    valid = thresholds_df.dropna(subset=["L_c_acc", "L_c_sem"])
    if valid.empty:
        return {"n_lanes": 0, "warning": "no lanes with both thresholds"}
    diff = (valid["L_c_acc"], valid["L_c_sem"]).astype(float).to_numpy()
    # Wilcoxon vs 0 (paired)
    try:
        w_stat, w_p = stats.wilcoxon(diff, alternative="greater",
                                     zero_method="wilcox", correction=False)
    except Exception:
        w_stat, w_p = float("nan"), float("nan")
    # Paired bootstrap on mean diff
    point, lo, hi = bootstrap_ci(diff, n=2000)
    # Permutation: shuffle sign of paired differences
    rng = np.random.default_rng(123)
    obs = float(diff.mean())
    n_perm = 10_000
    perm = rng.choice([-1, 1], size=(n_perm, diff.size)) * diff
    null_means = perm.mean(axis=1)
    perm_p = float((null_means >= obs).mean())
    return {
        "n_lanes": int(valid.size),
        "mean_gap": float(diff.mean()),
        "median_gap": float(np.median(diff)),
        "ci_lo": lo, "ci_hi": hi,
        "wilcoxon_stat": float(w_stat), "wilcoxon_p_one_sided_gt0": float(w_p),
        "permutation_p_one_sided_gt0": perm_p,
        "lanes_with_positive_gap": int((diff > 0).sum()),
        "lanes_with_zero_gap": int((diff == 0).sum()),
        "lanes_with_negative_gap": int((diff < 0).sum()),
    }


# ---------------------------- H2 -------------------------------------------
def h2_input_vs_output(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Per (model, dataset, level), compare condition output vs input."""
    rows: List[Dict[str, Any]] = []
    metrics = [
        ("accuracy", cfg.col("correct")),
        ("output_tokens", cfg.col("output_tokens")),
        ("input_tokens", cfg.col("input_tokens")),
        ("cost_usd", "cost_usd_safe"),
        ("embedding_similarity", cfg.col("embedding_similarity")),
        ("bidir_entailment", cfg.col("bidirectional_entailment")),
    ]
    idx_col = cfg.col("idx")
    level_col = cfg.col("level")
    for (m, ds, L), grp in df.groupby(["__model", "__dataset", level_col]):
        out_grp = grp[grp["__condition"] == "output"]
        in_grp = grp[grp["__condition"] == "input"]
        if out_grp.empty or in_grp.empty:
            continue
        # Try to pair on idx
        joined = out_grp.merge(in_grp, on=idx_col, suffixes=("_out", "_in"))
        if joined.empty:
            continue
        for label, col in metrics:
            col_o = f"{col}_out"; col_i = f"{col}_in"
            if col_o not in joined.columns or col_i not in joined.columns:
                continue
            ao = pd.to_numeric(joined[col_o], errors="coerce").to_numpy()
            ai = pd.to_numeric(joined[col_i], errors="coerce").to_numpy()
            mask = ~(np.isnan(ao) | np.isnan(ai))
            ao, ai = ao[mask], ai[mask]
            if ao.size < 2:
                continue
            try:
                w_stat, w_p = stats.wilcoxon(ao, ai, zero_method="wilcox")
            except Exception:
                w_stat, w_p = float("nan"), float("nan")
            mean_diff, lo, hi = paired_bootstrap_diff(ao, ai, n=1000)
            d = cohens_d(ao, ai)
            rows.append({
                "model": m, "dataset": ds, "level": L, "metric": label,
                "n_pairs": int(ao.size),
                "mean_output": float(ao.mean()),
                "mean_input": float(ai.mean()),
                "mean_diff_out_minus_in": mean_diff,
                "ci_lo": lo, "ci_hi": hi,
                "cohens_d": d,
                "wilcoxon_stat": float(w_stat) if w_stat == w_stat else None,
                "wilcoxon_p": float(w_p) if w_p == w_p else None,
            })
    out_df = pd.DataFrame(rows)
    if not out_df.empty:
        adj = adjust_pvalues(out_df["wilcoxon_p"].fillna(1.0).to_numpy(),
                             method="holm-bonferroni")
        out_df["p_adj_holm"] = adj
        adj2 = adjust_pvalues(out_df["wilcoxon_p"].fillna(1.0).to_numpy(),
                              method="benjamini-hochberg")
        out_df["p_adj_bh"] = adj2
    return out_df


# ---------------------------- H3 -------------------------------------------
def h3_dissociation_table(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Build C1/C2/C3/C4 cells per (model, dataset, condition, level)."""
    correct = cfg.col("correct")
    entail = cfg.col("bidirectional_entailment")
    if entail not in df.columns:
        return pd.DataFrame()
    sub = df.dropna(subset=[correct, entail]).copy()
    sub["__correct"] = sub[correct].astype(bool)
    sub["__entails"] = sub[entail].astype(bool)
    rows: List[Dict[str, Any]] = []
    level_col = cfg.col("level")
    for (m, ds, c, L), grp in sub.groupby(["__model", "__dataset", "__condition",
                                           level_col]):
        n = len(grp)
        c1 = int(((grp["__correct"]) & (grp["__entails"])).sum())   # correct & entails
        c2 = int(((grp["__correct"]) & (~grp["__entails"])).sum())  # correct & collapsed
        c3 = int(((~grp["__correct"]) & (grp["__entails"])).sum())  # wrong & entails
        c4 = int(((~grp["__correct"]) & (~grp["__entails"])).sum()) # wrong & collapsed
        # Binomial test: is C2 > 0? Use 1-cell vs n.
        # p-value: probability of C2 or more under H0: p=0 → trivially 0 if C2>0.
        # More informative: binomial test that C2 > expected under marginals.
        p_marg = (c1 + c2) * (c2 + c4) / max(n * n, 1)  # under independence
        if c2 > 0 and p_marg > 0:
            try:
                # one-sided: is observed C2 greater than independence prediction?
                bres = stats.binomtest(c2, n, p_marg, alternative="greater")
                c2_p = float(bres.pvalue)
            except Exception:
                c2_p = float("nan")
        else:
            c2_p = float("nan")
        # McNemar test on (correct vs entails) discordance
        try:
            # 2x2 contingency: rows=correct, cols=entails
            tab = np.array([[c1, c2], [c3, c4]])
            # McNemar uses off-diagonals b = C2 (correct & ¬entail) and c = C3
            b, cc = c2, c3
            if b + cc > 0:
                mc_stat = (abs(b, cc), 1) ** 2 / (b + cc)
                mc_p = float(1, stats.chi2.cdf(mc_stat, df=1))
            else:
                mc_stat, mc_p = float("nan"), float("nan")
        except Exception:
            mc_stat, mc_p = float("nan"), float("nan")
        rows.append({
            "model": m, "dataset": ds, "condition": c, "level": L,
            "n": n,
            "C1_correct_entails":   c1, "C1_pct":  c1 / max(n, 1),
            "C2_correct_collapsed": c2, "C2_pct":  c2 / max(n, 1),
            "C3_wrong_entails":     c3, "C3_pct":  c3 / max(n, 1),
            "C4_wrong_collapsed":   c4, "C4_pct":  c4 / max(n, 1),
            "C2_vs_independence_p": c2_p,
            "mcnemar_stat":         mc_stat, "mcnemar_p": mc_p,
        })
    return pd.DataFrame(rows)


# ---------------------------- driver ---------------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    return add_common_args(ap).parse_args()


def main() -> int:
    args = parse_args()
    cfg = apply_common_overrides(load_config(args.config), args)
    logger = setup_logging(cfg, "hypothesis_tests", verbose=args.verbose)
    out = setup_output_dirs(cfg)

    df = load_all(cfg, logger=logger)
    if df.empty:
        logger.error("no data loaded")
        return 1
    df = add_derived_columns(df, cfg)

    # H1, re-derive thresholds inline (cheap)
    from importlib import import_module
    sys.path.insert(0, str(Path(__file__).parent))
    cthr = import_module("collapse_thresholds")
    thr_rows = []
    for (m, ds, c), lane in df.groupby(["__model", "__dataset", "__condition"]):
        acc = cthr.acc_threshold(lane, cfg)
        sem = cthr.sem_threshold(lane, cfg)
        thr_rows.append({"model": m, "dataset": ds, "condition": c,
                         "L_c_acc": acc.get("L_c_acc"),
                         "L_c_sem": sem.get("L_c_sem")})
    thresholds_df = pd.DataFrame(thr_rows)
    h1 = h1_semantic_before_accuracy(thresholds_df)
    write_json(h1, out["stats"] / "h1_semantic_before_accuracy.json", logger=logger)

    # H2
    h2_df = h2_input_vs_output(df, cfg)
    write_csv(h2_df, out["tables"] / "h2_input_vs_output.csv", logger=logger)

    # H3
    h3_df = h3_dissociation_table(df, cfg)
    write_csv(h3_df, out["tables"] / "h3_dissociation_2x2.csv", logger=logger)

    # Combined p-value index
    p_idx: List[Dict[str, Any]] = []
    if h1.get("wilcoxon_p_one_sided_gt0") is not None:
        p_idx.append({"test": "H1_wilcoxon", "p_raw": h1["wilcoxon_p_one_sided_gt0"]})
    if not h2_df.empty:
        for _, r in h2_df.iterrows():
            if pd.notna(r.get("wilcoxon_p")):
                p_idx.append({"test": f"H2_{r['model']}/{r['dataset']}/L{r['level']}/{r['metric']}",
                              "p_raw": float(r["wilcoxon_p"])})
    if p_idx:
        idx_df = pd.DataFrame(p_idx)
        idx_df["p_adj_holm"] = adjust_pvalues(idx_df["p_raw"].to_numpy(), "holm-bonferroni")
        idx_df["p_adj_bh"] = adjust_pvalues(idx_df["p_raw"].to_numpy(), "benjamini-hochberg")
        write_csv(idx_df, out["tables"] / "hypothesis_pvalue_index.csv", logger=logger)

    logger.info("H1 / H2 / H3 hypothesis tests complete")
    logger.info(f"H1 mean gap = {h1.get('mean_gap')}; permutation p = {h1.get('permutation_p_one_sided_gt0')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
