"""CAVEWOMAN evaluation, read JSONL outputs from run_experiment.py and compute
per-level + cross-level metrics.

Pure stdlib + numpy + scipy. No torch, no transformers, no GPU required.

Usage:
    python evaluate_results.py --results_dir results/run_20260511_154322
    python evaluate_results.py --results_dir results/run_xxx --output reports/run_xxx

Reads any subset of caveman_gsm8k_L{0..4}.jsonl that's present and skips the
rest. Cross-level analyses require >= 2 levels with data.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy import stats

# Reuse the canonical per-level aggregator.
_SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

from constraint_prompts import LEVEL_ORDER  # noqa: E402
from metrics_utils import summarize_level_results  # noqa: E402


# ---------- CLI ----------

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate CAVEWOMAN per-level JSONL results."
    )
    p.add_argument(
        "--results_dir",
        required=True,
        help="Directory containing caveman_gsm8k_L*.jsonl files.",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Where to write the report (default: same as --results_dir).",
    )
    return p.parse_args()


# ---------- io ----------

def _read_jsonl(path: Path) -> List[Dict]:
    out: List[Dict] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                # Skip malformed lines (e.g. a torn write from a killed job).
                continue
    return out


def _level_num(level: str) -> int:
    """'L0' -> 0, 'L4' -> 4."""
    return int(level[1])


# ---------- analysis ----------

def evaluate(results_dir: Path) -> Dict:
    per_level: Dict[str, Dict] = {}
    levels_analyzed: List[str] = []

    for lvl in LEVEL_ORDER:
        f = results_dir / f"caveman_gsm8k_{lvl}.jsonl"
        if not f.exists():
            print(f"[skip] {lvl}: no file at {f.name}")
            continue
        records = _read_jsonl(f)
        if not records:
            print(f"[skip] {lvl}: file present but empty")
            continue
        per_level[lvl] = summarize_level_results(records)
        levels_analyzed.append(lvl)
        print(f"[load] {lvl}: n={per_level[lvl]['n']}  acc={per_level[lvl]['accuracy']:.3f}")

    if len(levels_analyzed) < 2:
        print("[warn] fewer than 2 levels with data; skipping cross-level analyses.")
        return {
            "per_level": per_level,
            "cross_level": None,
            "levels_analyzed": levels_analyzed,
        }

    # Build aligned vectors over the levels that actually have data.
    x = np.array([_level_num(lvl) for lvl in levels_analyzed], dtype=float)
    acc = np.array([per_level[lvl]["accuracy"] for lvl in levels_analyzed])
    out_toks = np.array([per_level[lvl]["mean_output_tokens"] for lvl in levels_analyzed])
    densities = np.array([per_level[lvl]["mean_info_density"] for lvl in levels_analyzed])

    # 1) Linear regression: accuracy ~ level_num.
    lr = stats.linregress(x, acc)
    linear_regression = {
        "slope": float(lr.slope),
        "intercept": float(lr.intercept),
        "r_value": float(lr.rvalue),
        "r_squared": float(lr.rvalue ** 2),
        "p_value": float(lr.pvalue),
        "std_err": float(lr.stderr),
        "n_levels": int(len(levels_analyzed)),
    }

    # 2) Threshold scan: drop[i] = acc(L_{i-1}), acc(L_i). Positive = decrease.
    drops: List[Dict] = []
    for i in range(1, len(levels_analyzed)):
        prev_lvl, cur_lvl = levels_analyzed[i, 1], levels_analyzed[i]
        drop = float(per_level[prev_lvl]["accuracy"], per_level[cur_lvl]["accuracy"])
        drops.append({"from_level": prev_lvl, "to_level": cur_lvl, "drop": drop})
    largest_drop = max(drops, key=lambda d: d["drop"])

    # 3-4) Spearman correlations. Tuple-unpacking is version-independent.
    rho_tokens, p_tokens = stats.spearmanr(x, out_toks)
    rho_density, p_density = stats.spearmanr(x, densities)

    return {
        "per_level": per_level,
        "cross_level": {
            "linear_regression": linear_regression,
            "drops": drops,
            "largest_drop": largest_drop,
            "spearman_level_vs_output_tokens": {
                "correlation": float(rho_tokens),
                "p_value": float(p_tokens),
            },
            "spearman_level_vs_info_density": {
                "correlation": float(rho_density),
                "p_value": float(p_density),
            },
        },
        "levels_analyzed": levels_analyzed,
    }


# ---------- presentation ----------

def print_report(result: Dict) -> None:
    per_level = result["per_level"]
    cross = result.get("cross_level")

    bar = "=" * 80
    print("\n" + bar)
    print("CAVEWOMAN evaluation report".center(80))
    print(bar)

    print("\nPer-level metrics:")
    print("-" * 80)
    print(
        f"{'Level':<6} {'N':>6} {'Acc':>7} {'OutTok':>9} {'OTMed':>8} "
        f"{'InfoDen':>9} {'Extract':>9} {'L4Viol':>9}"
    )
    print("-" * 80)
    for lvl in LEVEL_ORDER:
        s = per_level.get(lvl)
        if not s:
            continue
        print(
            f"{lvl:<6} {s['n']:>6d} {s['accuracy']:>7.3f} "
            f"{s['mean_output_tokens']:>9.1f} {s['median_output_tokens']:>8.1f} "
            f"{s['mean_info_density']:>9.3f} "
            f"{s['answer_extraction_rate']*100:>8.1f}% "
            f"{s['l4_budget_violations']*100:>8.1f}%"
        )
    print("-" * 80)
    print("(L4Viol is the proportion of outputs exceeding 15 tokens; only L4 is bound by this.)")

    if cross is None:
        print("\n(insufficient levels for cross-level analyses)")
        print(bar)
        return

    lr = cross["linear_regression"]
    print("\nLinear regression (accuracy ~ level_num):")
    print(f"  slope     = {lr['slope']:+.4f}   (accuracy change per +1 level)")
    print(f"  intercept = {lr['intercept']:.4f}")
    print(f"  R²        = {lr['r_squared']:.4f}")
    print(f"  p-value   = {lr['p_value']:.4f}   n_levels = {lr['n_levels']}")

    print("\nSingle-step accuracy drops  (positive = decrease):")
    largest = cross["largest_drop"]
    for d in cross["drops"]:
        marker = "   <-- largest" if d is largest else ""
        print(f"  {d['from_level']} -> {d['to_level']}:  {d['drop']:+.4f}{marker}")

    spt = cross["spearman_level_vs_output_tokens"]
    spd = cross["spearman_level_vs_info_density"]
    print("\nSpearman correlations vs level number:")
    print(f"  mean_output_tokens : rho = {spt['correlation']:+.4f}   p = {spt['p_value']:.4f}")
    print(f"  mean_info_density  : rho = {spd['correlation']:+.4f}   p = {spd['p_value']:.4f}")

    print(bar)


def write_csv(per_level: Dict[str, Dict], path: Path) -> None:
    cols = [
        "level", "n", "accuracy", "answer_extraction_rate",
        "mean_output_tokens", "median_output_tokens",
        "mean_info_density", "l4_budget_violations",
    ]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for lvl in LEVEL_ORDER:
            s = per_level.get(lvl)
            if not s:
                continue
            w.writerow([
                lvl, s["n"], s["accuracy"], s["answer_extraction_rate"],
                s["mean_output_tokens"], s["median_output_tokens"],
                s["mean_info_density"], s["l4_budget_violations"],
            ])


# ---------- main ----------

def main() -> None:
    args = parse_args()

    results_dir = Path(args.results_dir).expanduser().resolve()
    if not results_dir.is_dir():
        raise FileNotFoundError(
            f"--results_dir does not exist or is not a directory: {results_dir}"
        )

    out_dir = (
        Path(args.output).expanduser().resolve() if args.output else results_dir
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[eval] reading from : {results_dir}")
    print(f"[eval] writing to   : {out_dir}")

    result = evaluate(results_dir)
    result["meta"] = {
        "results_dir": str(results_dir),
        "output_dir": str(out_dir),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    print_report(result)

    json_path = out_dir / "caveman_evaluation_report.json"
    csv_path = out_dir / "caveman_metrics_table.csv"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    write_csv(result["per_level"], csv_path)

    print(f"\n[eval] JSON report -> {json_path}")
    print(f"[eval] CSV table   -> {csv_path}")


if __name__ == "__main__":
    main()
