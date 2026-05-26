"""Add bidirectional NLI entailment scores to CAVEWOMAN per-level JSONL files.

For every (L1..L4) item, compare its `output` against the L0 `output` for the
matching `idx` using `cross-encoder/nli-deberta-v3-base` in BOTH directions.
A pair is bidirectionally entailing iff both directions return `entailment`
as the highest-probability label.

CLI:
    python add_entailment_scores.py --results_dir <dir> --model_name qwen --dataset gsm8k

Outputs:
    caveman_<model_name>_<dataset>_L{1..4}_with_entailment.jsonl

The L0 file is NOT modified; an L0-vs-L0 sanity-check rate is printed instead.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

_SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

from entailment_scorer import load_nli_model, batch_score_entailment  # noqa: E402

LEVELS = ["L0", "L1", "L2", "L3", "L4"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Add bidirectional NLI entailment-vs-L0 scores to CAVEWOMAN JSONL files."
    )
    p.add_argument("--results_dir", required=True, help="Directory holding caveman_*_L*.jsonl files")
    p.add_argument("--model_name", required=True, help="Model tag in file names, e.g. 'qwen-2.5'")
    p.add_argument("--dataset", required=True, help="Dataset tag in file names, e.g. 'gsm8k'")
    p.add_argument("--condition", choices=["output", "input"], default=None,
                   help="Condition tag in file names ('output' or 'input'). "
                        "If omitted, auto-detect from filenames; falls back to legacy "
                        "naming (no condition segment) if neither variant is present.")
    p.add_argument("--batch_size", type=int, default=32, help="NLI batch size (default 32)")
    return p.parse_args()


def _build_lane_paths(results_dir: Path, model_name: str, dataset: str,
                      condition: str, levels) -> dict:
    """Build paths with or without the `_<condition>_` segment."""
    if condition:
        return {lvl: results_dir / f"caveman_{model_name}_{dataset}_{condition}_{lvl}.jsonl"
                for lvl in levels}
    return {lvl: results_dir / f"caveman_{model_name}_{dataset}_{lvl}.jsonl"
            for lvl in levels}


def _resolve_condition(results_dir: Path, model_name: str, dataset: str,
                       requested: str = None) -> str:
    """Return the condition tag to use, auto-detecting if `requested` is None.

    Returns empty string for legacy (no-condition) filenames.
    """
    if requested:
        return requested
    if (results_dir / f"caveman_{model_name}_{dataset}_output_L0.jsonl").exists():
        return "output"
    if (results_dir / f"caveman_{model_name}_{dataset}_input_L0.jsonl").exists():
        return "input"
    return ""


def read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def score_pairs_in_batches(nli, pairs, batch_size: int, desc: str):
    """Wrap `batch_score_entailment` with a tqdm progress bar over chunks."""
    out = []
    for i in tqdm(range(0, len(pairs), batch_size), desc=desc, unit="batch"):
        chunk = pairs[i : i + batch_size]
        out.extend(batch_score_entailment(nli, chunk, batch_size=batch_size))
    return out


def process_level(
    nli,
    lvl: str,
    lx_path: Path,
    l0_by_idx: Dict[int, str],
    out_path: Path,
    batch_size: int,
) -> Dict:
    records = read_jsonl(lx_path)
    paired_records, unpaired = [], 0
    for r in records:
        if r.get("idx") in l0_by_idx:
            paired_records.append(r)
        else:
            unpaired += 1
    if unpaired:
        print(f"  [{lvl}] WARNING: {unpaired} records have no L0 idx match (kept in output with NaN scores)")

    lx_to_l0_pairs = [(r.get("output", "") or "", l0_by_idx[r["idx"]]) for r in paired_records]
    l0_to_lx_pairs = [(l0_by_idx[r["idx"]], r.get("output", "") or "") for r in paired_records]

    print(f"  [{lvl}] scoring {len(paired_records)} pairs in both directions"
          f" ({2 * len(paired_records)} NLI calls total)")
    res_lx = score_pairs_in_batches(nli, lx_to_l0_pairs, batch_size, f"{lvl} Lx->L0")
    res_l0 = score_pairs_in_batches(nli, l0_to_lx_pairs, batch_size, f"{lvl} L0->Lx")

    res_by_idx = {r["idx"]: (a, b) for r, a, b in zip(paired_records, res_lx, res_l0)}

    n_bi = n_a = n_b = 0
    sum_a_prob = 0.0
    with open(out_path, "w") as fout:
        for r in records:
            idx = r.get("idx")
            new_rec = dict(r)
            if idx in res_by_idx:
                a, b = res_by_idx[idx]
                a_ent = a["label"] == "entailment"
                b_ent = b["label"] == "entailment"
                bidir = bool(a_ent and b_ent)
                new_rec["lx_entails_l0"] = bool(a_ent)
                new_rec["l0_entails_lx"] = bool(b_ent)
                new_rec["bidirectional_entailment"] = bidir
                new_rec["lx_entails_l0_prob"] = round(float(a["entailment_prob"]), 6)
                new_rec["l0_entails_lx_prob"] = round(float(b["entailment_prob"]), 6)
                new_rec["entailment_label"] = a["label"]
                n_a += int(a_ent)
                n_b += int(b_ent)
                n_bi += int(bidir)
                sum_a_prob += float(a["entailment_prob"])
            else:
                new_rec["lx_entails_l0"] = None
                new_rec["l0_entails_lx"] = None
                new_rec["bidirectional_entailment"] = None
                new_rec["lx_entails_l0_prob"] = None
                new_rec["l0_entails_lx_prob"] = None
                new_rec["entailment_label"] = None
            fout.write(json.dumps(new_rec) + "\n")

    n = max(len(paired_records), 1)
    stats = {
        "level": lvl,
        "n_total": len(records),
        "n_paired": len(paired_records),
        "bidirectional_rate": n_bi / n,
        "lx_entails_l0_rate": n_a / n,
        "l0_entails_lx_rate": n_b / n,
        "mean_lx_entails_l0_prob": sum_a_prob / n,
        "out_path": str(out_path),
    }
    print(f"  [{lvl}] wrote {out_path.name}: bidir={n_bi}/{n} ({stats['bidirectional_rate']:.1%})")
    return stats


def sanity_check_l0_self(nli, l0_records, batch_size: int,
                          sample_size: int = 500) -> Dict:
    """Score L0 outputs against themselves and verify bidirectional entailment ≈ 1.0.

    For large datasets (> sample_size) we sub-sample deterministically with seed
    so the sanity check still finishes in minutes rather than hours on CPU. A
    sample of 500 is statistically sufficient to detect rates below 0.99
    (binomial 95% CI half-width ≈ 1pp at p=0.99).
    """
    import random
    n_total = len(l0_records)
    if n_total > sample_size:
        rng = random.Random(42)
        sampled = rng.sample(l0_records, sample_size)
        print(f"\n[sanity] sub-sampling {sample_size} of {n_total} L0 records for self-check "
              f"(deterministic seed=42; full check at this scale would take ~{(n_total/500)*5:.0f} "
              f"min on CPU)")
    else:
        sampled = l0_records
    pairs = [(r.get("output", "") or "", r.get("output", "") or "") for r in sampled]
    print(f"[sanity] scoring {len(pairs)} L0-vs-L0 self-pairs (1 direction; symmetric)...")
    res = score_pairs_in_batches(nli, pairs, batch_size, "L0 self")
    n_ent = sum(1 for x in res if x["label"] == "entailment")
    rate = n_ent / max(len(res), 1)
    mean_prob = sum(x["entailment_prob"] for x in res) / max(len(res), 1)
    print(f"[sanity] L0-vs-L0 entailment rate = {rate:.4f} ({n_ent}/{len(res)});"
          f" mean entailment prob = {mean_prob:.4f}")
    if rate < 0.99:
        print(f"[sanity] WARNING: rate < 0.99, the NLI model fails to call identical strings entailing"
              f" {len(res), n_ent} times. Inspect those records before trusting downstream stats.")
    return {"n": len(res), "n_total": n_total, "sampled": n_total > sample_size,
            "entailment_rate": rate, "mean_entailment_prob": mean_prob}


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir).expanduser().resolve()
    if not results_dir.is_dir():
        raise FileNotFoundError(f"--results_dir not found: {results_dir}")

    condition = _resolve_condition(results_dir, args.model_name, args.dataset, args.condition)
    paths = _build_lane_paths(results_dir, args.model_name, args.dataset, condition, LEVELS)
    missing = [lvl for lvl, p in paths.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing JSONL files for levels: {missing} under {results_dir}")
    if "L0" not in paths:
        raise FileNotFoundError("L0 file required as baseline")

    print(f"[scan] results_dir={results_dir}  condition={condition or '(legacy)'}")
    for lvl in LEVELS:
        print(f"  {lvl}: {paths[lvl].name}")

    l0_records = read_jsonl(paths["L0"])
    l0_by_idx = {r["idx"]: (r.get("output", "") or "") for r in l0_records if "idx" in r}
    print(f"\n[L0] {len(l0_by_idx)} reference outputs indexed by idx")

    # Plan which levels still need work so a timed-out re-run resumes instead
    # of redoing everything. We treat a level as "done" iff its output file
    # exists AND has the expected record count.
    plan = []
    for lvl in ["L1", "L2", "L3", "L4"]:
        out_path = paths[lvl].with_name(paths[lvl].stem + "_with_entailment.jsonl")
        if out_path.exists():
            try:
                with open(out_path) as f:
                    n = sum(1 for _ in f)
            except Exception:
                n = 0
            expected = sum(1 for _ in open(paths[lvl]))
            if n == expected:
                print(f"[resume] {lvl}: already complete ({n}/{expected}) → skip")
                continue
            else:
                print(f"[resume] {lvl}: partial ({n}/{expected}) → redo")
        plan.append((lvl, out_path))

    if not plan:
        print("[done] all 4 levels already have entailment files; nothing to do.")
        return

    print(f"\n[nli] loading cross-encoder/nli-deberta-v3-base ...")
    nli = load_nli_model()

    # Sanity check still runs once per session (cheap with sub-sampling)
    sanity = sanity_check_l0_self(nli, l0_records, batch_size=args.batch_size)

    all_stats = []
    for lvl, out_path in plan:
        s = process_level(nli, lvl, paths[lvl], l0_by_idx, out_path, batch_size=args.batch_size)
        all_stats.append(s)

    print("\n" + "=" * 96)
    print("Entailment summary".center(96))
    print("=" * 96)
    header = f"{'level':<6} {'n_total':>8} {'n_paired':>9} {'bidir':>10} {'Lx->L0':>10} {'L0->Lx':>10} {'mean P(Lx->L0)':>16}"
    print(header)
    print("-" * 96)
    for s in all_stats:
        print(
            f"{s['level']:<6} {s['n_total']:>8d} {s['n_paired']:>9d} "
            f"{s['bidirectional_rate']*100:>9.2f}% "
            f"{s['lx_entails_l0_rate']*100:>9.2f}% "
            f"{s['l0_entails_lx_rate']*100:>9.2f}% "
            f"{s['mean_lx_entails_l0_prob']:>16.4f}"
        )
    print("=" * 96)
    print(f"L0 self-sanity: entailment_rate={sanity['entailment_rate']:.4f},"
          f" mean_prob={sanity['mean_entailment_prob']:.4f}, n={sanity['n']}")
    print("[done] entailment-augmented JSONL files written.")


if __name__ == "__main__":
    main()
