#!/usr/bin/env python3
"""Within-family aggregation on the MATCHED seeded 500-item subset per dataset.

Family: Qwen2.5-Instruct 3B/7B/14B/32B (sweep runs, n=500) plus
Qwen2.5-VL-7B and Qwen3.5-9B (full runs, subset to the same idx) and
Qwen3.5-9B thinking-on (sweep run, n=500).
"""
from pathlib import Path
import json, os

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
import sys as _sys
_sys.path.insert(0, str(_REPO_ROOT / "src"))
from release_paths import results_root  # noqa: E402
_RESULTS = results_root(_REPO_ROOT / "results")

ROOT = str(_RESULTS)
SWEEP = ["qwen2.5-0.5b", "qwen2.5-1.5b", "qwen2.5-3b", "qwen2.5-7b",
         "qwen2.5-14b", "qwen2.5-32b", "qwen3.5-think"]
FULL = ["qwen2.5", "qwen3.5"]
TAGS = SWEEP + FULL
CONDS = ["input", "output"]
DATASETS = ["gsm8k", "boolq", "arc_easy", "commonsenseqa", "mmlu_stem"]
LEVELS = ["L0", "L1", "L2", "L3", "L4"]


def read_jsonl(p):
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def path(tag, cond, ds, lv, ent=False):
    suf = "_with_entailment" if ent else ""
    return os.path.join(ROOT, f"{tag}_{cond}", ds, f"caveman_{tag}_{ds}_{cond}_{lv}{suf}.jsonl")

# Generations ship in the data bundle, not the code release.
_probe = Path(path("qwen2.5-32b", "output", "gsm8k", "L1"))
if not _probe.exists():
    raise SystemExit(
        f"missing sweep generations, e.g.\n  {_probe}\n\n"
        "Unpack the data bundle into results/, or run the sweep yourself with\n"
        "experiments/size_sweep/sweep_job.sh."
    )


# 1. Establish the matched idx set per dataset from the 32B sweep run (L1, output).
matched = {}
for ds in DATASETS:
    p = path("qwen2.5-32b", "output", ds, "L1")
    matched[ds] = {r["idx"] for r in read_jsonl(p)}

cells = {}
for tag in TAGS:
    for cond in CONDS:
        for ds in DATASETS:
            keep = matched[ds]
            for lv in LEVELS:
                p = path(tag, cond, ds, lv)
                if not os.path.exists(p):
                    continue
                recs = {r["idx"]: r for r in read_jsonl(p)}
                recs = {i: r for i, r in recs.items() if i in keep}
                if not recs:
                    continue
                n = len(recs)
                acc = sum(1 for r in recs.values() if r.get("correct")) / n
                tin = sum(r.get("input_tokens", 0) for r in recs.values()) / n
                tout = sum(r.get("output_tokens", 0) for r in recs.values()) / n
                nli, n_nli = None, 0
                pe = path(tag, cond, ds, lv, ent=True)
                if os.path.exists(pe):
                    er = {r["idx"]: r for r in read_jsonl(pe)}
                    vals = [r["bidirectional_entailment"] for i, r in er.items()
                            if i in keep and r.get("bidirectional_entailment") is not None]
                    if vals:
                        nli = sum(vals) / len(vals)
                        n_nli = len(vals)
                cells[(tag, cond, ds, lv)] = dict(n=n, acc=acc, nli=nli, n_nli=n_nli,
                                                  tin=tin, tout=tout)

out = _RESULTS / "revision" / "size_sweep" / "matched_cells.json"
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    json.dump({"|".join(k): v for k, v in cells.items()}, f, indent=1)


def mean(tag, cond, lv, field):
    vals = [cells[(tag, cond, ds, lv)][field] for ds in DATASETS
            if (tag, cond, ds, lv) in cells and cells[(tag, cond, ds, lv)][field] is not None]
    return sum(vals) / len(vals) if len(vals) == 5 else None


ORDER = ["qwen2.5-0.5b", "qwen2.5-1.5b", "qwen2.5-3b", "qwen2.5-7b", "qwen2.5",
         "qwen2.5-14b", "qwen2.5-32b",
         "qwen3.5", "qwen3.5-think"]
LABEL = {"qwen2.5-0.5b": "Qwen2.5-0.5B", "qwen2.5-1.5b": "Qwen2.5-1.5B",
         "qwen2.5-3b": "Qwen2.5-3B", "qwen2.5-7b": "Qwen2.5-7B",
         "qwen2.5": "Qwen2.5-VL-7B", "qwen2.5-14b": "Qwen2.5-14B",
         "qwen2.5-32b": "Qwen2.5-32B", "qwen3.5": "Qwen3.5-9B",
         "qwen3.5-think": "Qwen3.5-9B(think)"}

for cond in CONDS:
    print(f"\n{'='*84}\nCONDITION {cond.upper()}  --  5-benchmark mean, matched n=500/dataset\n{'='*84}")
    print(f"{'model':<20} {'metric':<5} " + " ".join(f"{l:>7}" for l in LEVELS))
    for tag in ORDER:
        for field, name in (("acc", "acc"), ("nli", "NLI"), ("tout", "out"), ("tin", "in")):
            row = []
            for lv in LEVELS:
                v = mean(tag, cond, lv, field)
                row.append("   ---" if v is None else (f"{v:7.3f}" if field in ("acc", "nli") else f"{v:7.1f}"))
            print(f"{LABEL[tag]:<20} {name:<5} " + " ".join(row))
        print()
