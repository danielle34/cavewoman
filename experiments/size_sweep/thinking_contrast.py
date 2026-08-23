#!/usr/bin/env python3
"""Qwen3.5-9B thinking ON vs OFF.

ON  = `qwen3.5`, subset to the sweep's 500 seeded idx
OFF = `qwen3.5-nothink`, run with --disable_thinking

Same checkpoint, seed, decoder budgets, and items, so the only difference is
the chat template's thinking branch. That branch is suppressed only by an
explicit False: a run with the flag left undefined is a thinking-ON run.

Usage:
  python experiments/size_sweep/thinking_contrast.py
"""
import json, os

from pathlib import Path as _Path
import sys as _sys
_REPO_ROOT = _Path(__file__).resolve().parent.parent.parent
_sys.path.insert(0, str(_REPO_ROOT / "src"))
from release_paths import results_root  # noqa: E402
RES = str(results_root(_REPO_ROOT / "results"))
DS = ["gsm8k", "boolq", "arc_easy", "commonsenseqa", "mmlu_stem"]
LV = ["L0", "L1", "L2", "L4"]


def jl(p):
    if not os.path.exists(p):
        raise SystemExit(
            f"missing generations:\n  {p}\n\n"
            "This compares the qwen3.5 and qwen3.5-nothink sweep arms. Run both\n"
            "(experiments/size_sweep/sweep_job.sh, the nothink arm with\n"
            "--disable_thinking), or point --results-root at a tree that has them."
        )
    with open(p, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def cell(tag, cond, ds, lv, keep=None):
    p = os.path.join(RES, f"{tag}_{cond}", ds, f"caveman_{tag}_{ds}_{cond}_{lv}.jsonl")
    if not os.path.exists(p):
        return None
    r = {x["idx"]: x for x in jl(p)}
    if keep:
        r = {i: x for i, x in r.items() if i in keep}
    if not r:
        return None
    e = os.path.join(RES, f"{tag}_{cond}", ds,
                     f"caveman_{tag}_{ds}_{cond}_{lv}_with_entailment.jsonl")
    nli = None
    if os.path.exists(e):
        d = {x["idx"]: x["bidirectional_entailment"] for x in jl(e)
             if x.get("bidirectional_entailment") is not None and (not keep or x["idx"] in keep)}
        if d:
            nli = 100 * sum(d.values()) / len(d)
    return dict(n=len(r),
                acc=100 * sum(1 for x in r.values() if x["correct"]) / len(r),
                out=sum(x["output_tokens"] for x in r.values()) / len(r),
                think=100 * sum(1 for x in r.values()
                                if "<think>" in x["output"] or "Thinking Process" in x["output"]) / len(r),
                nli=nli)


# matched idx set = whatever the OFF arm ran
KEEP = {}
for ds in DS:
    KEEP[ds] = {x["idx"] for x in jl(os.path.join(
        RES, "qwen3.5-nothink_output", ds, f"caveman_qwen3.5-nothink_{ds}_output_L1.jsonl"))}


def mean(tag, cond, lv, f):
    v = []
    for ds in DS:
        c = cell(tag, cond, ds, lv, KEEP[ds])
        if c and c[f] is not None:
            v.append(c[f])
    return sum(v) / len(v) if len(v) == 5 else None


for cond in ("output", "input"):
    print(f"\n{'='*72}\nCondition {'B (output)' if cond=='output' else 'A (input)'}"
          f"  -- 5-benchmark mean, matched n=500/dataset\n{'='*72}")
    print(f"{'':<22}" + "".join(f"{lv:>11}" for lv in LV))
    for tag, lab in (("qwen3.5", "thinking ON"), ("qwen3.5-nothink", "thinking OFF")):
        for f, nm in (("acc", "accuracy %"), ("nli", "NLI %"), ("out", "out tokens"),
                      ("think", "% w/ think blk")):
            row = []
            for lv in LV:
                v = mean(tag, cond, lv, f)
                row.append("        ---" if v is None else f"{v:11.1f}")
            print(f"  {lab:<10}{nm:<10}" + "".join(row))
        print()

print(f"\n{'='*72}\nPer-benchmark accuracy, Condition B (the headline contrast)\n{'='*72}")
print(f"{'Benchmark':<15}" + "".join(f"{lv+' ON':>10}{lv+' OFF':>11}" for lv in LV))
for ds in DS:
    row = ""
    for lv in LV:
        on = cell("qwen3.5", "output", ds, lv, KEEP[ds])
        off = cell("qwen3.5-nothink", "output", ds, lv, KEEP[ds])
        row += f"{on['acc']:>10.1f}{off['acc']:>11.1f}" if on and off else f"{'---':>10}{'---':>11}"
    print(f"{ds:<15}{row}")
