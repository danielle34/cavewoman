"""Data quality + schema validation for CAVEWOMAN result files.

Walks every (model, dataset, condition, level) combo declared in config.yaml,
checks file existence + record counts + required columns + missing-value
profile, and writes a single data-quality report (CSV + JSON).

Usage:
    python validate_results.py                       # default config
    python validate_results.py --config my.yaml
    python validate_results.py --models gpt-4o haiku
    python validate_results.py --datasets gsm8k
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from _lib import (
    Config, add_common_args, apply_common_overrides, load_config,
    setup_logging, setup_output_dirs, lane_path, read_jsonl,
    write_csv, write_json,
)


REQUIRED_COLS = [
    "idx", "level", "model", "dataset", "condition",
    "correct", "predicted_answer", "gt_answer",
    "input_tokens", "output_tokens", "output",
]


def file_quality(cfg: Config, model: str, dataset: str, condition: str,
                 level: str, logger) -> Dict[str, Any]:
    p = lane_path(cfg, model, dataset, condition, level)
    expected = cfg.n_expected(dataset)
    record: Dict[str, Any] = {
        "model": model, "dataset": dataset, "condition": condition,
        "level": level, "path": str(p), "exists": p.exists(),
        "expected": expected,
    }
    if not p.exists():
        record.update({"n_records": 0, "missing_files": True})
        return record

    df = read_jsonl(p)
    record["n_records"] = int(len(df))
    record["count_ok"] = bool(len(df) == expected)
    record["count_pct"] = round(len(df) / max(expected, 1) * 100.0, 2)

    # Required columns
    actual = set(df.columns)
    missing_cols = [c for c in REQUIRED_COLS if cfg.col(c) not in actual]
    record["missing_columns"] = missing_cols
    record["all_required_present"] = len(missing_cols) == 0

    # Duplicate idx
    idx_col = cfg.col("idx")
    if idx_col in df.columns:
        n_dup = int(df[idx_col].duplicated().sum())
        record["duplicate_idx"] = n_dup
    else:
        record["duplicate_idx"] = None

    # Missing-value profile for core numeric columns
    nulls: Dict[str, int] = {}
    for col_key in ("correct", "input_tokens", "output_tokens", "output", "predicted_answer"):
        c = cfg.col(col_key)
        if c in df.columns:
            nulls[col_key] = int(df[c].isna().sum())
        else:
            nulls[col_key] = -1   # column absent
    record["nulls"] = nulls

    # Augmentation files
    ent_p = lane_path(cfg, model, dataset, condition, level, augmented="entailment")
    emb_p = lane_path(cfg, model, dataset, condition, level, augmented="embeddings")
    record["has_entailment_file"] = bool(ent_p.exists())
    record["has_embeddings_file"] = bool(emb_p.exists())

    if record["has_entailment_file"]:
        ent = read_jsonl(ent_p)
        record["entailment_records"] = int(len(ent))
        record["entailment_field_present"] = bool(
            cfg.col("bidirectional_entailment") in ent.columns
        )
    if record["has_embeddings_file"]:
        emb = read_jsonl(emb_p)
        record["embedding_records"] = int(len(emb))
        record["embedding_field_present"] = bool(
            cfg.col("embedding_similarity") in emb.columns
        )

    return record


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap = add_common_args(ap)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    cfg = apply_common_overrides(load_config(args.config), args)
    logger = setup_logging(cfg, "validate_results", verbose=args.verbose)
    out = setup_output_dirs(cfg)

    logger.info(f"results_root = {cfg.paths['results_root']}")
    logger.info(f"models       = {cfg.models}")
    logger.info(f"datasets     = {cfg.datasets}")
    logger.info(f"conditions   = {cfg.conditions}")

    rows: List[Dict[str, Any]] = []
    for m in cfg.models:
        for ds in cfg.datasets:
            for c in cfg.conditions:
                for L in cfg.levels:
                    rec = file_quality(cfg, m, ds, c, L, logger)
                    rows.append(rec)

    df = pd.DataFrame(rows)

    # Summary aggregate per (model, condition, dataset)
    if not df.empty:
        agg = (df.groupby(["model", "condition", "dataset"])
                 .agg(levels_present=("exists", "sum"),
                      records_total=("n_records", "sum"),
                      count_ok=("count_ok", "sum"),
                      all_required_present=("all_required_present", "all"),
                      entailment_complete=("has_entailment_file",
                                           lambda s: int(s.sum())),
                      embeddings_complete=("has_embeddings_file",
                                           lambda s: int(s.sum())),
                 ).reset_index())
    else:
        agg = pd.DataFrame()

    write_csv(df, out["tables"] / "data_quality_per_level.csv", logger=logger)
    write_csv(agg, out["tables"] / "data_quality_per_lane.csv", logger=logger)

    # Headline summary in JSON
    summary: Dict[str, Any] = {
        "total_combos": int(len(df)),
        "files_present": int(df["exists"].sum()) if not df.empty else 0,
        "files_missing": int((~df["exists"]).sum()) if not df.empty else 0,
        "lanes_with_full_5_levels": int((agg["levels_present"] == 5).sum())
                                   if not agg.empty else 0,
        "lanes_with_full_records": int((agg["count_ok"] == 5).sum())
                                   if not agg.empty else 0,
        "lanes_with_required_columns": int(agg["all_required_present"].sum())
                                       if not agg.empty else 0,
        "lanes_with_entailment_4lvls": int((agg["entailment_complete"] >= 4).sum())
                                       if not agg.empty else 0,
        "lanes_with_embeddings_5lvls": int((agg["embeddings_complete"] >= 5).sum())
                                       if not agg.empty else 0,
        "models_seen": sorted(df["model"].unique().tolist()) if not df.empty else [],
        "datasets_seen": sorted(df["dataset"].unique().tolist()) if not df.empty else [],
    }
    write_json(summary, out["stats"] / "data_quality_summary.json", logger=logger)

    # Console summary
    logger.info("")
    logger.info("====== DATA QUALITY SUMMARY ======")
    for k, v in summary.items():
        logger.info(f"  {k:<32} = {v}")
    logger.info("==================================")

    return 0


if __name__ == "__main__":
    sys.exit(main())
