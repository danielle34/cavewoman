"""Add sentence-embedding cosine similarity (vs L0 output) to every CAVEWOMAN record.

Uses `sentence-transformers/all-MiniLM-L6-v2`.

CLI:
    python add_embedding_scores.py --results_dir <dir> --model_name qwen --dataset gsm8k

Outputs (one per level, including L0 for the self-similarity sanity check):
    caveman_<model_name>_<dataset>_L{0..4}_with_embeddings.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

LEVELS = ["L0", "L1", "L2", "L3", "L4"]
EMBED_MODEL = "all-MiniLM-L6-v2"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Add sentence-embedding cosine similarity to L0 for every CAVEWOMAN record."
    )
    p.add_argument("--results_dir", required=True)
    p.add_argument("--model_name", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--condition", choices=["output", "input"], default=None,
                   help="Condition tag in file names ('output' or 'input'). "
                        "If omitted, auto-detect from filenames; falls back to legacy "
                        "naming (no condition segment) if neither variant is present.")
    p.add_argument("--batch_size", type=int, default=64)
    return p.parse_args()


def _build_lane_paths(results_dir: Path, model_name: str, dataset: str,
                      condition: str, levels) -> dict:
    if condition:
        return {lvl: results_dir / f"caveman_{model_name}_{dataset}_{condition}_{lvl}.jsonl"
                for lvl in levels}
    return {lvl: results_dir / f"caveman_{model_name}_{dataset}_{lvl}.jsonl"
            for lvl in levels}


def _resolve_condition(results_dir: Path, model_name: str, dataset: str,
                       requested: str = None) -> str:
    if requested:
        return requested
    if (results_dir / f"caveman_{model_name}_{dataset}_output_L0.jsonl").exists():
        return "output"
    if (results_dir / f"caveman_{model_name}_{dataset}_input_L0.jsonl").exists():
        return "input"
    return ""


def read_jsonl(path: Path) -> List[Dict]:
    out: List[Dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def encode_texts(model: SentenceTransformer, texts: List[str], batch_size: int, desc: str) -> np.ndarray:
    """Encode texts in batches with tqdm progress and L2 normalization for cosine."""
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc=desc, unit="batch"):
        chunk = texts[i : i + batch_size]
        emb = model.encode(
            chunk,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        embeddings.append(emb)
    return np.vstack(embeddings) if embeddings else np.zeros((0, model.get_sentence_embedding_dimension()))


def process_level(
    model: SentenceTransformer,
    lvl: str,
    lx_path: Path,
    l0_emb_by_idx: Dict[int, np.ndarray],
    out_path: Path,
    batch_size: int,
) -> Dict:
    records = read_jsonl(lx_path)
    paired_idx = [i for i, r in enumerate(records) if r.get("idx") in l0_emb_by_idx]
    unpaired = len(records), len(paired_idx)
    if unpaired:
        print(f"  [{lvl}] WARNING: {unpaired} records have no L0 idx match (writing similarity=None)")

    texts = [records[i].get("output", "") or "" for i in paired_idx]
    lx_emb = encode_texts(model, texts, batch_size, f"{lvl} encode")
    l0_emb_aligned = np.vstack([l0_emb_by_idx[records[i]["idx"]] for i in paired_idx]) if paired_idx else np.zeros_like(lx_emb)

    if paired_idx:
        sims = np.einsum("ij,ij->i", lx_emb, l0_emb_aligned)
    else:
        sims = np.array([])

    sim_by_record_pos = {pos: float(sims[k]) for k, pos in enumerate(paired_idx)}

    sim_values: List[float] = []
    with open(out_path, "w") as fout:
        for pos, r in enumerate(records):
            new_rec = dict(r)
            if pos in sim_by_record_pos:
                s = sim_by_record_pos[pos]
                new_rec["embedding_similarity"] = round(s, 4)
                sim_values.append(s)
            else:
                new_rec["embedding_similarity"] = None
            new_rec["embedding_model"] = EMBED_MODEL
            fout.write(json.dumps(new_rec) + "\n")

    if not sim_values:
        return {"level": lvl, "n": 0, "mean": float("nan"), "std": float("nan"),
                "median": float("nan"), "min": float("nan"), "out_path": str(out_path)}

    arr = np.asarray(sim_values)
    stats = {
        "level": lvl,
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "out_path": str(out_path),
    }
    print(f"  [{lvl}] wrote {out_path.name}: mean={stats['mean']:.4f},"
          f" std={stats['std']:.4f}, median={stats['median']:.4f}, min={stats['min']:.4f}")
    return stats


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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[scan] results_dir={results_dir}  device={device}  condition={condition or '(legacy)'}")
    print(f"[embed] loading {EMBED_MODEL} ...")
    model = SentenceTransformer(EMBED_MODEL, device=device)
    print(f"[embed] dim={model.get_sentence_embedding_dimension()}")

    l0_records = read_jsonl(paths["L0"])
    l0_texts = [r.get("output", "") or "" for r in l0_records]
    l0_idxs = [r.get("idx") for r in l0_records]
    print(f"\n[L0] encoding {len(l0_texts)} reference outputs")
    l0_emb = encode_texts(model, l0_texts, args.batch_size, "L0 reference encode")
    l0_emb_by_idx = {idx: l0_emb[i] for i, idx in enumerate(l0_idxs) if idx is not None}

    # Plan which levels still need work so a timed-out re-run resumes instead
    # of redoing everything. Treat a level as "done" iff its output file
    # exists AND has the expected record count.
    plan = []
    for lvl in LEVELS:
        out_path = paths[lvl].with_name(paths[lvl].stem + "_with_embeddings.jsonl")
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
        print("[done] all 5 levels already have embedding files; nothing to do.")
        # still return cleanly so submit_postprocess_cpu.sh logs success
        return

    all_stats = []
    for lvl, out_path in plan:
        s = process_level(model, lvl, paths[lvl], l0_emb_by_idx, out_path, batch_size=args.batch_size)
        all_stats.append(s)

    l0_stats = next((s for s in all_stats if s["level"] == "L0"), None)
    if l0_stats is not None:
        print(f"\n[sanity] L0 self-similarity: mean={l0_stats['mean']:.6f},"
              f" min={l0_stats['min']:.6f}  (expected mean=1.000000, min=1.000000)")
        if abs(l0_stats["mean"], 1.0) > 1e-3 or l0_stats["min"] < 0.999:
            print("[sanity] WARNING: L0 self-similarity drifts from 1.0, verify idx alignment and encoder determinism")

    print("\n" + "=" * 80)
    print("Embedding similarity summary".center(80))
    print("=" * 80)
    print(f"{'level':<6} {'n':>6} {'mean':>10} {'std':>10} {'median':>10} {'min':>10}")
    print("-" * 80)
    for s in all_stats:
        print(f"{s['level']:<6} {s['n']:>6d} {s['mean']:>10.4f} {s['std']:>10.4f}"
              f" {s['median']:>10.4f} {s['min']:>10.4f}")
    print("=" * 80)
    print("[done] embedding-augmented JSONL files written.")


if __name__ == "__main__":
    main()
