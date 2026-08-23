#!/usr/bin/env python3
"""Bidirectional NLI for the MGSM multilingual cells, with a selectable judge.

Why this exists: the multilingual entailment sidecars in results/multilingual/
covered only 4 of 10 languages, on 2 of 4 models, at L1-L2 only, and were scored
with `cross-encoder/nli-deberta-v3-base` -- an English-only judge applied to
Bengali, Chinese and Swahili text. Every generation is present, so the gap is
pure scoring.

This scores any (model, language, level) cell against its own L0 in both
directions, under either judge:

  --judge en    cross-encoder/nli-deberta-v3-base        (matches the English panel)
  --judge xnli  MoritzLaurer/mDeBERTa-v3-base-mnli-xnli  (multilingual, appropriate
                                                          for non-Latin scripts)

Sidecars are written as *_with_entailment.jsonl (en) or *_with_entailment_xnli.jsonl
(xnli), so the two judges never overwrite each other and can be compared on the
four languages that already have English-judge scores.
"""
import argparse, json, sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
from release_paths import results_root  # noqa: E402
MGSM = results_root(_REPO_ROOT / "results") / "multilingual"
LANGS = ["de", "es", "fr", "sw", "zh", "ja", "ru", "bn", "th", "te"]
MODELS = ["qwen2.5", "gemma4", "gpt4o", "haiku"]
LEVELS = ["L1", "L2", "L3", "L4"]

JUDGES = {
    "en":   "cross-encoder/nli-deberta-v3-base",
    "xnli": "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
}


def jl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def load_judge(name):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[judge] loading {name} on {dev}", flush=True)
    tok = AutoTokenizer.from_pretrained(name)
    mdl = AutoModelForSequenceClassification.from_pretrained(name).to(dev).eval()
    lab = {v.lower(): k for k, v in mdl.config.id2label.items()}
    ent_idx = lab.get("entailment")
    if ent_idx is None:
        raise SystemExit(f"cannot find an entailment label in {mdl.config.id2label}")
    return tok, mdl, dev, ent_idx


def entail_probs(tok, mdl, dev, ent_idx, premises, hypotheses, bs=32):
    """P(entailment) for each (premise, hypothesis) pair."""
    import torch
    out = []
    for i in range(0, len(premises), bs):
        p, h = premises[i:i + bs], hypotheses[i:i + bs]
        enc = tok(p, h, return_tensors="pt", truncation=True,
                  max_length=512, padding=True).to(dev)
        with torch.no_grad():
            probs = mdl(**enc).logits.softmax(-1)[:, ent_idx]
        out.extend(probs.float().cpu().tolist())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default=None,
                       help="Path to the results tree "
                            "(default: <repo>/results, or $CAVEWOMAN_RESULTS).")
    ap.add_argument("--judge", choices=list(JUDGES), default="xnli")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--langs", default=",".join(LANGS))
    ap.add_argument("--levels", default=",".join(LEVELS))
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="P(entailment) above which a direction counts as entailed")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    suffix = "_with_entailment" if args.judge == "en" else "_with_entailment_xnli"
    tok, mdl, dev, ent_idx = load_judge(JUDGES[args.judge])

    n_cells = n_skip = 0
    for m in args.models.split(","):
        for lg in args.langs.split(","):
            d = MGSM / f"{m}_output" / lg
            base = d / f"caveman_{m}_mgsm-{lg}_output"
            l0 = Path(f"{base}_L0.jsonl")
            if not l0.exists():
                continue
            ref = {r["idx"]: r["output"] for r in jl(l0)}
            for lv in args.levels.split(","):
                src = Path(f"{base}_{lv}.jsonl")
                dst = Path(f"{base}_{lv}{suffix}.jsonl")
                if not src.exists():
                    continue
                if dst.exists() and not args.overwrite:
                    n_skip += 1
                    continue
                rows = [r for r in jl(src) if r["idx"] in ref]
                if not rows:
                    continue
                lx = [r["output"] for r in rows]
                l0t = [ref[r["idx"]] for r in rows]
                fwd = entail_probs(tok, mdl, dev, ent_idx, lx, l0t)   # Lx |= L0
                bwd = entail_probs(tok, mdl, dev, ent_idx, l0t, lx)   # L0 |= Lx
                with open(dst, "w", encoding="utf-8") as f:
                    for r, a, b in zip(rows, fwd, bwd):
                        r = dict(r)
                        r["lx_entails_l0"] = bool(a > args.threshold)
                        r["l0_entails_lx"] = bool(b > args.threshold)
                        r["bidirectional_entailment"] = bool(a > args.threshold and b > args.threshold)
                        r["lx_entails_l0_prob"] = round(a, 6)
                        r["l0_entails_lx_prob"] = round(b, 6)
                        r["nli_judge"] = JUDGES[args.judge]
                        f.write(json.dumps(r) + "\n")
                rate = 100 * sum(1 for a, b in zip(fwd, bwd)
                                 if a > args.threshold and b > args.threshold) / len(rows)
                print(f"  {m:<8} {lg:<3} {lv}  n={len(rows):<4} bidir={rate:5.1f}%", flush=True)
                n_cells += 1
    print(f"[done] scored {n_cells} cells, skipped {n_skip} already present", flush=True)


if __name__ == "__main__":
    main()
