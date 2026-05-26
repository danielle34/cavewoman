"""Compute accuracy + semantic collapse thresholds and the dissociation gap.

For each (model, dataset, condition) lane, compute:
  L_c_acc  = first level where mean accuracy < (L0 accuracy × accuracy_collapse_rel)
  L_c_sem  = first level where mean embedding similarity < semantic_collapse_cosine
  gap      = L_c_acc, L_c_sem    (positive = semantic collapses earlier)

Bootstrap CIs on each threshold by resampling within-level records 2000×.

Writes:
  tables/collapse_thresholds.csv
  latex/tab_collapse_thresholds.tex
  stats/collapse_thresholds_bootstrap.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from _lib import (
    Config, add_common_args, apply_common_overrides, load_config,
    setup_logging, setup_output_dirs, load_all, add_derived_columns,
    write_csv, write_json, write_latex,
)


def first_level_below(series_by_level: Dict[str, float], threshold: float,
                      level_order: List[str]) -> Optional[int]:
    """Return numeric index 0..4 of first level where value < threshold; else None."""
    for i, L in enumerate(level_order):
        v = series_by_level.get(L)
        if v is None or np.isnan(v):
            continue
        if v < threshold:
            return i
    return None


def acc_threshold(lane: pd.DataFrame, cfg: Config) -> Dict:
    """Find L_c_acc and its bootstrap CI."""
    levels = cfg.levels
    correct = cfg.col("correct")
    level_col = cfg.col("level")
    by_level = {L: lane[lane[level_col] == L][correct].astype(float).to_numpy()
                for L in levels}
    if "L0" not in by_level or by_level["L0"].size == 0:
        return {"L_c_acc": None, "L0_acc": None, "ci_lo": None, "ci_hi": None}
    l0_mean = by_level["L0"].mean()
    rel = float(cfg.thresholds["accuracy_collapse_rel"])
    threshold = l0_mean * rel
    means = {L: (arr.mean() if arr.size else np.nan) for L, arr in by_level.items()}
    L_c = first_level_below(means, threshold, levels)

    # Bootstrap CI on the threshold index
    rng = np.random.default_rng(int(cfg.stats["rng_seed"]))
    boots: List[int] = []
    n_boot = int(cfg.stats["bootstrap_n"])
    for _ in range(n_boot):
        bm = {}
        for L, arr in by_level.items():
            if arr.size == 0:
                bm[L] = np.nan; continue
            bm[L] = arr[rng.integers(0, arr.size, size=arr.size)].mean()
        idx = first_level_below(bm, threshold, levels)
        if idx is not None:
            boots.append(idx)
    alpha = (1.0, float(cfg.stats["bootstrap_ci"])) / 2.0
    if boots:
        lo = float(np.quantile(boots, alpha))
        hi = float(np.quantile(boots, 1, alpha))
    else:
        lo = hi = None
    return {
        "L_c_acc": L_c, "L0_acc": float(l0_mean),
        "threshold": float(threshold), "ci_lo": lo, "ci_hi": hi,
        "boot_n": len(boots),
    }


def sem_threshold(lane: pd.DataFrame, cfg: Config) -> Dict:
    """Find L_c_sem using mean cosine similarity < semantic_collapse_cosine."""
    levels = cfg.levels
    emb = cfg.col("embedding_similarity")
    level_col = cfg.col("level")
    if emb not in lane.columns:
        return {"L_c_sem": None, "ci_lo": None, "ci_hi": None}
    by_level = {L: lane[lane[level_col] == L][emb].astype(float).dropna().to_numpy()
                for L in levels}
    means = {L: (arr.mean() if arr.size else np.nan) for L, arr in by_level.items()}
    threshold = float(cfg.thresholds["semantic_collapse_cosine"])
    L_c = first_level_below(means, threshold, levels)

    rng = np.random.default_rng(int(cfg.stats["rng_seed"]) + 1)
    boots: List[int] = []
    n_boot = int(cfg.stats["bootstrap_n"])
    for _ in range(n_boot):
        bm = {}
        for L, arr in by_level.items():
            if arr.size == 0:
                bm[L] = np.nan; continue
            bm[L] = arr[rng.integers(0, arr.size, size=arr.size)].mean()
        idx = first_level_below(bm, threshold, levels)
        if idx is not None:
            boots.append(idx)
    alpha = (1.0, float(cfg.stats["bootstrap_ci"])) / 2.0
    if boots:
        lo = float(np.quantile(boots, alpha))
        hi = float(np.quantile(boots, 1, alpha))
    else:
        lo = hi = None
    return {"L_c_sem": L_c, "threshold": threshold,
            "ci_lo": lo, "ci_hi": hi, "boot_n": len(boots)}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    return add_common_args(ap).parse_args()


def main() -> int:
    args = parse_args()
    cfg = apply_common_overrides(load_config(args.config), args)
    logger = setup_logging(cfg, "collapse_thresholds", verbose=args.verbose)
    out = setup_output_dirs(cfg)

    df = load_all(cfg, logger=logger)
    if df.empty:
        logger.error("no data loaded")
        return 1
    df = add_derived_columns(df, cfg)

    rows = []
    boot_records = {}
    for (m, ds, c), lane in df.groupby(["__model", "__dataset", "__condition"]):
        acc = acc_threshold(lane, cfg)
        sem = sem_threshold(lane, cfg)
        gap = None
        if acc["L_c_acc"] is not None and sem["L_c_sem"] is not None:
            gap = acc["L_c_acc"], sem["L_c_sem"]
        rows.append({
            "model": m, "dataset": ds, "condition": c,
            "L0_accuracy": acc.get("L0_acc"),
            "L_c_acc": acc.get("L_c_acc"),
            "L_c_acc_ci_lo": acc.get("ci_lo"), "L_c_acc_ci_hi": acc.get("ci_hi"),
            "L_c_sem": sem.get("L_c_sem"),
            "L_c_sem_ci_lo": sem.get("ci_lo"), "L_c_sem_ci_hi": sem.get("ci_hi"),
            "gap_acc_minus_sem": gap,
        })
        boot_records[f"{m}/{ds}/{c}"] = {"acc": acc, "sem": sem}

    table = pd.DataFrame(rows)
    write_csv(table, out["tables"] / "collapse_thresholds.csv", logger=logger)
    write_json(boot_records, out["stats"] / "collapse_thresholds_bootstrap.json",
               logger=logger)
    write_latex(table, out["latex"] / "tab_collapse_thresholds.tex",
                caption="Accuracy vs semantic collapse thresholds, with dissociation gap "
                        "(positive = semantic collapses earlier).",
                label="tab:collapse-thresholds", float_format="%.2f", logger=logger)

    logger.info(f"computed thresholds for {len(table)} lanes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
