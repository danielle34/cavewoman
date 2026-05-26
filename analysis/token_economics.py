"""Token economics + cost analysis.

Computes per-lane:
, input-token savings vs L0   (ratio)
, output-token expansion vs L0
, total cost savings vs L0
, cost-per-correct
, break-even output expansion e_max = 1 + ((1, r) * I) / (rho * O)
     where r = input cost ratio achieved, rho = output/input price ratio
     I, O = L0 input/output token volumes per record
Flags cells where compression is cost-negative (more expensive than L0).

Pricing comes from config.yaml (override individual models or add new ones).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from _lib import (
    Config, add_common_args, apply_common_overrides, load_config,
    setup_logging, setup_output_dirs, load_all, add_derived_columns,
    write_csv, write_json, write_latex,
)


def lane_token_volumes(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    level_col = cfg.col("level")
    i = cfg.col("input_tokens"); o = cfg.col("output_tokens")
    grp = (df.groupby(["__model", "__dataset", "__condition", level_col])
             .agg(n=(i, "size"),
                  in_tokens_sum=(i, lambda s: float(np.nansum(s.astype(float)))),
                  out_tokens_sum=(o, lambda s: float(np.nansum(s.astype(float)))),
                  in_tokens_mean=(i, lambda s: float(np.nanmean(s.astype(float)))),
                  out_tokens_mean=(o, lambda s: float(np.nanmean(s.astype(float))))
             ).reset_index()
             .rename(columns={"__model": "model", "__dataset": "dataset",
                              "__condition": "condition", level_col: "level"}))
    return grp


def apply_pricing(token_df: pd.DataFrame, cfg: Config,
                  pricing_key_override: Optional[str] = None) -> pd.DataFrame:
    """Add cost columns per row using config.pricing.

    pricing_key_override: if provided, apply ALL rows under that pricing
    (useful for "what if everything ran on DSv4 Pro?" projections).
    """
    rows: List[Dict[str, Any]] = []
    for _, r in token_df.iterrows():
        key = pricing_key_override or r["model"]
        rates = cfg.pricing.get(key, {"input": 0.0, "output": 0.0})
        ci = float(rates.get("input", 0.0))
        co = float(rates.get("output", 0.0))
        in_cost  = r["in_tokens_sum"] / 1e6 * ci
        out_cost = r["out_tokens_sum"] / 1e6 * co
        rows.append({
            **r.to_dict(),
            "pricing_key":    key,
            "rate_in_per_M":  ci,
            "rate_out_per_M": co,
            "cost_input":     in_cost,
            "cost_output":    out_cost,
            "cost_total":     in_cost + out_cost,
        })
    return pd.DataFrame(rows)


def compute_savings_vs_l0(cost_df: pd.DataFrame) -> pd.DataFrame:
    """Per (model, dataset, condition), express L1..L4 relative to L0."""
    out_rows = []
    for (m, ds, c), grp in cost_df.groupby(["model", "dataset", "condition"]):
        l0 = grp[grp["level"] == "L0"]
        if l0.empty:
            continue
        l0_in  = float(l0["in_tokens_sum"].iloc[0])
        l0_out = float(l0["out_tokens_sum"].iloc[0])
        l0_cost = float(l0["cost_total"].iloc[0])
        for _, r in grp.iterrows():
            level = r["level"]
            in_ratio  = float(r["in_tokens_sum"]) / l0_in  if l0_in  > 0 else np.nan
            out_ratio = float(r["out_tokens_sum"]) / l0_out if l0_out > 0 else np.nan
            cost_ratio = float(r["cost_total"]) / l0_cost  if l0_cost > 0 else np.nan
            out_rows.append({
                "model": m, "dataset": ds, "condition": c, "level": level,
                "input_token_ratio_to_L0":  in_ratio,
                "input_token_savings_pct":  (1, in_ratio) * 100 if not np.isnan(in_ratio) else np.nan,
                "output_token_ratio_to_L0": out_ratio,
                "output_token_expansion_pct": (out_ratio, 1) * 100 if not np.isnan(out_ratio) else np.nan,
                "cost_ratio_to_L0":         cost_ratio,
                "cost_savings_pct":         (1, cost_ratio) * 100 if not np.isnan(cost_ratio) else np.nan,
                "cost_total":               r["cost_total"],
                "cost_negative":            bool(cost_ratio > 1.0) if not np.isnan(cost_ratio) else False,
            })
    return pd.DataFrame(out_rows)


def break_even_table(cost_df: pd.DataFrame) -> pd.DataFrame:
    """Per (model, dataset, condition, level), compute e_max, the max output
    expansion before compression turns cost-negative.

    e_max = 1 + ((1, r) * I) / (rho * O)

      r   = input-token ratio achieved at this level vs L0   (I_Lx / I_L0)
      rho = output_price / input_price                        (per model)
      I   = L0 input tokens per record
      O   = L0 output tokens per record
    """
    rows = []
    for (m, ds, c), grp in cost_df.groupby(["model", "dataset", "condition"]):
        l0 = grp[grp["level"] == "L0"]
        if l0.empty: continue
        I = float(l0["in_tokens_mean"].iloc[0])
        O = float(l0["out_tokens_mean"].iloc[0])
        ci = float(l0["rate_in_per_M"].iloc[0])
        co = float(l0["rate_out_per_M"].iloc[0])
        if ci <= 0 or O <= 0:
            continue
        rho = co / ci
        for _, r in grp.iterrows():
            level = r["level"]
            r_ratio = float(r["in_tokens_mean"]) / I if I > 0 else np.nan
            e_max = 1 + ((1, r_ratio) * I) / (rho * O) if not np.isnan(r_ratio) else np.nan
            actual_e = float(r["out_tokens_mean"]) / O if O > 0 else np.nan
            rows.append({
                "model": m, "dataset": ds, "condition": c, "level": level,
                "r_input_ratio": r_ratio,
                "rho_out_over_in_price": rho,
                "I_L0_per_record_in": I,
                "O_L0_per_record_out": O,
                "e_max":      e_max,
                "actual_e":   actual_e,
                "within_breakeven": bool(actual_e < e_max) if not (np.isnan(actual_e)
                                                                  or np.isnan(e_max)) else None,
            })
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--projections", nargs="*", default=None,
                    help="Pricing keys to use for projections "
                         "(e.g. deepseek_flash deepseek_pro). One CSV per key.")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    cfg = apply_common_overrides(load_config(args.config), args)
    logger = setup_logging(cfg, "token_economics", verbose=args.verbose)
    out = setup_output_dirs(cfg)

    df = load_all(cfg, logger=logger)
    if df.empty:
        logger.error("no data loaded")
        return 1
    df = add_derived_columns(df, cfg)

    volumes = lane_token_volumes(df, cfg)
    write_csv(volumes, out["tables"] / "token_volumes_per_cell.csv", logger=logger)

    # Pricing using each model's own configured key
    priced = apply_pricing(volumes, cfg)
    write_csv(priced, out["tables"] / "token_economics_actual.csv", logger=logger)

    savings = compute_savings_vs_l0(priced)
    write_csv(savings, out["tables"] / "token_economics_savings.csv", logger=logger)
    write_latex(savings.assign(
                    input_token_savings_pct=savings["input_token_savings_pct"].round(1),
                    output_token_expansion_pct=savings["output_token_expansion_pct"].round(1),
                    cost_savings_pct=savings["cost_savings_pct"].round(1),
                ),
                out["latex"] / "tab_token_economics.tex",
                caption="Token-level savings and expansion relative to L0 per lane.",
                label="tab:token-economics",
                float_format="%.2f", logger=logger)

    be = break_even_table(priced)
    write_csv(be, out["tables"] / "breakeven_e_max.csv", logger=logger)

    # Projections: re-price under hypothetical providers
    if args.projections:
        for key in args.projections:
            if key not in cfg.pricing:
                logger.warning(f"projection key not in config.pricing: {key}")
                continue
            proj = apply_pricing(volumes, cfg, pricing_key_override=key)
            write_csv(proj, out["tables"] / f"token_economics_proj_{key}.csv", logger=logger)
            proj_savings = compute_savings_vs_l0(proj)
            write_csv(proj_savings, out["tables"] / f"token_economics_proj_{key}_savings.csv",
                      logger=logger)

    # Flag cost-negative cells
    if not savings.empty:
        neg = savings[savings["cost_negative"]]
        write_csv(neg, out["tables"] / "cost_negative_cells.csv", logger=logger)
        logger.info(f"{len(neg)} cells are cost-negative (compression more expensive than L0)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
