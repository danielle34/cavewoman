"""Sampling floor: bidirectional NLI between the greedy L0 generation and a
T=0.7 paired L0 draw, per (model, condition, dataset).

This is how far a model's output moves when nothing is compressed, and it is
the reference the compression numbers should be read against.

Scans results/<model>_<cond>/<ds>/ for caveman_*_L0.jsonl alongside
*_L0_paired.jsonl, scores every idx present in both, and writes:

  results/revision/paired_l0/<model>__<ds>__<cond>__pairs.jsonl
  results/revision/paired_l0/<model>__<ds>__<cond>__summary.json
  results/revision/paired_l0/_grand_summary.json

Cells with an existing summary are skipped; --force rescans. GPU recommended.

Usage:
  python analysis/floors/score_sampling_floor.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
import sys as _sys
_sys.path.insert(0, str(_REPO_ROOT / "src"))
from release_paths import results_root  # noqa: E402
_RESULTS = results_root(_REPO_ROOT / "results")

REPO = _REPO_ROOT
RESULTS = _RESULTS
OUT = RESULTS / "revision" / "paired_l0"
sys.path.insert(0, str(_REPO_ROOT / "src"))

MODELS = ["qwen2.5", "qwen3.5", "deepseek", "gemma4"]
DATASETS = ["gsm8k", "boolq", "arc_easy", "commonsenseqa", "mmlu_stem"]
CONDITIONS = ["output", "input"]
# Match production add_entailment_scores.py: a direction entails iff the
# argmax NLI label is "entailment"; bidirectional = both directions.


def load_by_idx(path: Path) -> dict:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "idx" in r:
                out[r["idx"]] = r
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-root", default=None,
                       help="Path to the results tree "
                            "(default: <repo>/results, or $CAVEWOMAN_RESULTS).")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    from entailment_scorer import load_nli_model, batch_score_entailment
    nli = None
    OUT.mkdir(parents=True, exist_ok=True)

    for m in MODELS:
        for c in CONDITIONS:
            for d in DATASETS:
                base = RESULTS / f"{m}_{c}" / d
                p0 = base / f"caveman_{m}_{d}_{c}_L0.jsonl"
                pp = base / f"caveman_{m}_{d}_{c}_L0_paired.jsonl"
                if not (p0.exists() and pp.exists()):
                    continue
                summ_path = OUT / f"{m}__{d}__{c}__summary.json"
                if summ_path.exists() and not args.force:
                    continue
                l0 = load_by_idx(p0)
                lp = load_by_idx(pp)
                common = sorted(set(l0) & set(lp))
                if not common:
                    continue
                # incomplete paired draws: score what exists; summary records n
                if nli is None:
                    print("[load] NLI judge", flush=True)
                    nli = load_nli_model()
                t0 = time.time()
                fwd_pairs = [((l0[i].get("output") or ""), (lp[i].get("output") or "")) for i in common]
                bwd_pairs = [(b, a) for (a, b) in fwd_pairs]
                fwd = batch_score_entailment(nli, fwd_pairs, batch_size=args.batch_size)
                bwd = batch_score_entailment(nli, bwd_pairs, batch_size=args.batch_size)
                n_bi = 0
                with open(OUT / f"{m}__{d}__{c}__pairs.jsonl", "w", encoding="utf-8") as f:
                    for i, idx in enumerate(common):
                        fe = fwd[i]["label"] == "entailment"
                        be = bwd[i]["label"] == "entailment"
                        bi = fe and be
                        n_bi += bi
                        f.write(json.dumps({
                            "idx": idx, "model": m, "dataset": d, "condition": c,
                            "fwd_entails": bool(fe),
                            "bwd_entails": bool(be),
                            "fwd_entail_prob": round(fwd[i]["entailment_prob"], 4),
                            "bwd_entail_prob": round(bwd[i]["entailment_prob"], 4),
                            "bidirectional": bool(bi),
                        }) + "\n")
                summ = {
                    "model": m, "dataset": d, "condition": c,
                    "n_pairs": len(common),
                    "n_l0": len(l0), "n_paired": len(lp),
                    "bidir_entailment_rate": round(n_bi / len(common), 4),
                    "noise_floor_C2_equivalent": round(1 - n_bi / len(common), 4),
                    "elapsed_s": round(time.time() - t0, 1),
                }
                summ_path.write_text(json.dumps(summ, indent=2), encoding="utf-8")
                print(f"[cell] {m}/{d}/{c}: {len(common)} pairs, "
                      f"bidir={summ['bidir_entailment_rate']:.3f}", flush=True)

    # Pilot draws (MGSM per-language + XSum): same scoring on
    # results/{multilingual|longform}/<tag>_output/**/*_L0_paired.jsonl
    import glob as _glob
    for pp_str in sorted(
            _glob.glob(str(RESULTS / "multilingual" / "*_output" / "*" / "*_L0_paired.jsonl")) +
            _glob.glob(str(RESULTS / "longform" / "*_output" / "*" / "*_L0_paired.jsonl"))):
        pp = Path(pp_str)
        p0 = pp.with_name(pp.name.replace("_L0_paired", "_L0"))
        if not p0.exists():
            continue
        # caveman_<tag>_<dataset>_output_L0_paired.jsonl
        stem_bits = pp.name.replace(".jsonl", "").split("_")
        tag = stem_bits[1]
        dataset = "_".join(stem_bits[2:-3])  # between tag and 'output_L0_paired'
        summ_path = OUT / f"{tag}__{dataset}__output__summary.json"
        if summ_path.exists() and not args.force:
            continue
        l0 = load_by_idx(p0)
        lp = load_by_idx(pp)
        common = sorted(set(l0) & set(lp))
        if not common:
            continue
        if nli is None:
            print("[load] NLI judge", flush=True)
            nli = load_nli_model()
        t0 = time.time()
        fwd_pairs = [((l0[i].get("output") or ""), (lp[i].get("output") or "")) for i in common]
        bwd_pairs = [(b, a) for (a, b) in fwd_pairs]
        fwd = batch_score_entailment(nli, fwd_pairs, batch_size=args.batch_size)
        bwd = batch_score_entailment(nli, bwd_pairs, batch_size=args.batch_size)
        n_bi = 0
        with open(OUT / f"{tag}__{dataset}__output__pairs.jsonl", "w", encoding="utf-8") as f:
            for i, idx in enumerate(common):
                fe = fwd[i]["label"] == "entailment"
                be = bwd[i]["label"] == "entailment"
                bi = fe and be
                n_bi += bi
                f.write(json.dumps({
                    "idx": idx, "model": tag, "dataset": dataset, "condition": "output",
                    "fwd_entails": bool(fe), "bwd_entails": bool(be),
                    "bidirectional": bool(bi),
                }) + "\n")
        summ = {
            "model": tag, "dataset": dataset, "condition": "output",
            "n_pairs": len(common), "n_l0": len(l0), "n_paired": len(lp),
            "bidir_entailment_rate": round(n_bi / len(common), 4),
            "noise_floor_C2_equivalent": round(1 - n_bi / len(common), 4),
            "elapsed_s": round(time.time() - t0, 1),
        }
        summ_path.write_text(json.dumps(summ, indent=2), encoding="utf-8")
        print(f"[pilot cell] {tag}/{dataset}: {len(common)} pairs, "
              f"bidir={summ['bidir_entailment_rate']:.3f}", flush=True)

    # grand summary from all cell summaries
    cells = []
    for p in sorted(OUT.glob("*__summary.json")):
        cells.append(json.loads(p.read_text(encoding="utf-8")))
    if cells:
        n = sum(x["n_pairs"] for x in cells)
        bi = sum(x["bidir_entailment_rate"] * x["n_pairs"] for x in cells)
        (OUT / "_grand_summary.json").write_text(json.dumps({
            "n_cells": len(cells), "n_pairs_total": n,
            "bidir_rate_weighted": round(bi / n, 4),
            "per_cell": cells,
        }, indent=2), encoding="utf-8")
        print(f"[grand] {len(cells)} cells, {n} pairs, weighted bidir={bi/n:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
