"""CAVEWOMAN: sweep a local HuggingFace model over a reasoning dataset under L0-L4 linguistic constraints.

Imports shared modules from `src/` at the repository root and writes one
JSONL per level plus a final accuracy_summary.json.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Set

# Make repo-root `src/` importable regardless of cwd.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from constraint_prompts import (  # noqa: E402
    CONSTRAINT_PROMPTS,
    LEVEL_ORDER,
    get_max_tokens,
)
from dataset_loader import (  # noqa: E402
    load_gsm8k,
    check_answer_correct,
)
from metrics_utils import (  # noqa: E402
    extract_numeric_answer,
    count_semantic_units,
    compute_info_density,
    summarize_level_results,
)
from model_loader import (  # noqa: E402
    load_model,
    run_inference,
)
from dataset_loader_multi import (  # noqa: E402
    load_dataset_caveman,
    extract_answer,
    check_correct,
    compress_input,
)


# Condition B (input compression) uses a neutral system prompt for every level.
# Single source of truth, referenced from run_one_level() and snapshotted into
# run_config.json so the run directory is self-describing.
NEUTRAL_INPUT_SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Answer the following question accurately and completely."
)


def _snapshot_spacy_for_input_condition():
    """Return spaCy version metadata for Condition B reproducibility, or None on failure.

    Only call this when condition='input', for Condition A it's unused and we
    avoid the spaCy model load cost.
    """
    try:
        import spacy  # noqa: PLC0415

        from dataset_loader_multi import _get_spacy  # noqa: PLC0415

        nlp = _get_spacy()
        return {
            "spacy_version": spacy.__version__,
            "model_name": nlp.meta.get("name", "en_core_web_sm"),
            "model_version": nlp.meta.get("version", "unknown"),
            "model_lang": nlp.meta.get("lang", "en"),
        }
    except Exception as e:  # pragma: no cover, purely informational
        return {"error": f"could not introspect spaCy: {type(e).__name__}: {e}"}


# Generation kwargs default to deterministic decoding; --temperature can override.
# The actual value used at runtime is read from args.temperature; this constant is
# kept only as the legacy default in case anything imports it.
GENERATION_TEMPERATURE = 0.0
GENERATION_DO_SAMPLE = False


def _snapshot_generation_kwargs(levels, temperature=GENERATION_TEMPERATURE):
    """Capture every value passed to model.generate() so the run is self-describing."""
    return {
        "temperature": temperature,
        "do_sample": temperature > 0.0,
        "max_new_tokens_per_level": {L: get_max_tokens(L) for L in levels},
        "source": "src/model_loader.py::run_inference",
    }


def _snapshot_git():
    """Return commit SHA, dirty flag, working-tree porcelain status, and diffstat."""
    import subprocess  # noqa: PLC0415

    repo = Path(__file__).resolve().parent

    def _run(cmd, strip=True):
        try:
            out = subprocess.check_output(cmd, cwd=repo, stderr=subprocess.DEVNULL,
                                          timeout=10).decode("utf-8")
            return out.strip() if strip else out
        except Exception:
            return None

    sha = _run(["git", "rev-parse", "HEAD"])
    short = _run(["git", "rev-parse", "--short", "HEAD"])
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    # Do NOT strip porcelain, the leading status column is whitespace-sensitive.
    porcelain = _run(["git", "status", "--porcelain"], strip=False) or ""
    diffstat = _run(["git", "diff", "--stat"]) or ""
    dirty_files = [line[3:].rstrip("\r\n") for line in porcelain.splitlines() if line.strip()]
    return {
        "commit_sha": sha,
        "commit_short": short,
        "branch": branch,
        "dirty": bool(dirty_files),
        "dirty_files": dirty_files,
        "diffstat_summary": diffstat.splitlines()[-1] if diffstat else "",
    }


def _snapshot_python_env():
    """Snapshot Python + key library versions; missing libs reported as None."""
    import sys  # noqa: PLC0415

    versions = {"python": sys.version.split()[0]}
    for mod_name in ("torch", "transformers", "sentence_transformers", "datasets",
                     "spacy", "numpy", "scipy", "pandas", "tqdm"):
        try:
            mod = __import__(mod_name)
            versions[mod_name] = getattr(mod, "__version__", "unknown")
        except Exception:
            versions[mod_name] = None
    return versions


def _snapshot_hardware_and_slurm():
    """Hostname, GPU model + driver + CUDA, SLURM env vars."""
    import socket  # noqa: PLC0415
    import subprocess  # noqa: PLC0415

    hw = {"hostname": socket.gethostname()}
    try:
        import torch  # noqa: PLC0415

        hw["cuda_version"] = torch.version.cuda
        hw["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            hw["gpu_name"] = torch.cuda.get_device_name(0)
            hw["gpu_count"] = torch.cuda.device_count()
            try:
                props = torch.cuda.get_device_properties(0)
                hw["gpu_total_memory_gb"] = round(props.total_memory / (1024 ** 3), 2)
            except Exception:
                pass
    except Exception:
        pass
    try:
        nvsmi = subprocess.check_output(["nvidia-smi", "-L"], stderr=subprocess.DEVNULL,
                                        timeout=5).decode("utf-8").strip()
        hw["nvidia_smi_L"] = nvsmi
    except Exception:
        hw["nvidia_smi_L"] = None

    slurm_keys = ("SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_NODELIST", "SLURM_NTASKS",
                  "SLURM_CPUS_ON_NODE", "SLURM_MEM_PER_NODE", "SLURM_PARTITION",
                  "SLURM_GPUS_ON_NODE", "SLURM_SUBMIT_DIR")
    hw["slurm"] = {k: os.environ.get(k) for k in slurm_keys if os.environ.get(k)}
    return hw


def _snapshot_source_hashes():
    """SHA-1 of every script the pipeline depends on; defends against silent edits."""
    import hashlib  # noqa: PLC0415

    repo = Path(__file__).resolve().parent.parent
    tracked = [
        "experiments/run_experiment.py",
        "src/constraint_prompts.py",
        "src/dataset_loader.py",
        "src/dataset_loader_multi.py",
        "src/model_loader.py",
        "src/metrics_utils.py",
    ]
    hashes = {}
    for rel in tracked:
        p = repo / rel
        if not p.exists():
            hashes[rel] = None
            continue
        h = hashlib.sha1(p.read_bytes()).hexdigest()
        hashes[rel] = h
    return hashes


def _snapshot_conda_and_ldpath():
    """Capture the conda env and the load-bearing LD_LIBRARY_PATH."""
    return {
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "ld_library_path": os.environ.get("LD_LIBRARY_PATH", ""),
        "pythonunbuffered": os.environ.get("PYTHONUNBUFFERED"),
    }


def capture_runtime_provenance(levels, condition, temperature=GENERATION_TEMPERATURE):
    """Bundle every traceability field into one dict for run_config.json."""
    bundle = {
        "generation_kwargs": _snapshot_generation_kwargs(levels, temperature=temperature),
        "git": _snapshot_git(),
        "python_env": _snapshot_python_env(),
        "hardware_and_slurm": _snapshot_hardware_and_slurm(),
        "source_sha1": _snapshot_source_hashes(),
        "conda_and_ldpath": _snapshot_conda_and_ldpath(),
    }
    return bundle


# ---------- argument parsing ----------

def parse_args():
    p = argparse.ArgumentParser(
        description="CAVEWOMAN sweep over GSM8K under five constraint levels."
    )
    p.add_argument(
        "--level",
        default="all",
        choices=["all"] + LEVEL_ORDER,
        help="Which constraint level to run (default: all five).",
    )
    p.add_argument(
        "--n",
        type=int,
        default=None,
        help="Number of GSM8K items (default: full split, 1319 for test).",
    )
    p.add_argument(
        "--split",
        default="test",
        choices=["test", "train"],
        help="Dataset split.",
    )
    p.add_argument(
        "--output",
        default="results",
        help="Output directory for per-level JSONL files and summaries.",
    )
    p.add_argument(
        "--model_name",
        required=True,
        help="Tag used in JSONL filenames and the config. Use the HuggingFace "
             "release naming, e.g. qwen-2.5, qwen-3.5, deepseek-r1, gemma-4.",
    )
    p.add_argument(
        "--model_path",
        required=True,
        help="Local path to a model snapshot, or a HuggingFace model ID "
             "(e.g. Qwen/Qwen2.5-VL-7B-Instruct, google/gemma-4-e4b-it, "
             "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B).",
    )
    p.add_argument(
        "--no_resume",
        action="store_true",
        help="Disable resume; truncate per-level JSONL before writing.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Sampling seed when --n is set (so a resumed run sees the same items).",
    )
    p.add_argument(
        "--dataset",
        type=str,
        default="gsm8k",
        choices=["gsm8k", "boolq", "arc_easy", "commonsenseqa", "mmlu_stem"],
        help="Dataset to run (default: gsm8k).",
    )
    p.add_argument(
        "--condition",
        type=str,
        default="output",
        choices=["output", "input"],
        help="Condition A=output constrained, Condition B=input compressed (default: output).",
    )
    p.add_argument(
        "--start_idx",
        type=int,
        default=None,
        help="Start index for dataset chunking (default: None = start from beginning).",
    )
    p.add_argument(
        "--end_idx",
        type=int,
        default=None,
        help="End index for dataset chunking (default: None = run to end).",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Skip model load + inference; write placeholder records (verification only).",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=GENERATION_TEMPERATURE,
        help="Decoding temperature (default: 0.0 = greedy). Set >0 for paired-L0 noise-floor runs.",
    )
    p.add_argument(
        "--paired_suffix",
        type=str,
        default="",
        help="Suffix appended to output JSONL stem, e.g. '_paired' produces caveman_<model>_<ds>_<cond>_L0_paired.jsonl. Empty by default (no suffix).",
    )
    return p.parse_args()


# ---------- jsonl helpers ----------

def _read_completed_idx(path: Path) -> Set[int]:
    """Return the set of idx values already present in a JSONL file."""
    if not path.exists():
        return set()
    done: Set[int] = set()
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            idx = rec.get("idx")
            if isinstance(idx, int):
                done.add(idx)
    return done


def _read_records(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    out: List[Dict] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


# ---------- per-level sweep ----------

def run_level(
    level: str,
    items: List[Dict],
    tokenizer,
    model,
    model_name: str,
    out_path: Path,
    resume: bool,
    args,
) -> List[Dict]:
    """Run a single constraint level over the items list; append to JSONL.

    `args` is the argparse Namespace from parse_args(); used to read
    dataset / condition / start_idx / end_idx / dry_run.
    """
    if not resume and out_path.exists():
        out_path.unlink()

    completed = _read_completed_idx(out_path) if resume else set()
    todo = [it for it in items if it["idx"] not in completed]
    if completed:
        print(f"[{level}] Resuming: {len(completed)} done, {len(todo)} to go.")
    else:
        print(f"[{level}] Starting fresh: {len(todo)} items.")

    # Condition A (output): apply CAVEWOMAN output-constraint system prompt.
    # Condition B (input):  compress input; use a neutral system prompt.
    if args.condition == "input":
        system_prompt = NEUTRAL_INPUT_SYSTEM_PROMPT
    else:
        system_prompt = CONSTRAINT_PROMPTS[level]
    max_new = get_max_tokens(level)

    correct_count = 0
    processed = 0
    with open(out_path, "a") as fout:
        for item in todo:
            # Choose user message per condition.
            if args.condition == "input":
                user_message = compress_input(item["question_formatted"], level)
            else:
                user_message = item["question_formatted"]

            if args.dry_run:
                result = {
                    "output": f"[DRY RUN, condition={args.condition} level={level} idx={item['idx']}]",
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "latency_s": 0.0,
                }
            else:
                result = run_inference(
                    tokenizer=tokenizer,
                    model=model,
                    system_prompt=system_prompt,
                    user_message=user_message,
                    max_new_tokens=max_new,
                    temperature=args.temperature,
                )

            answer_type = item.get("answer_type", "numeric")
            gt_answer = item["answer_gt"]
            predicted = extract_answer(result["output"], answer_type)
            correct = check_correct(predicted, gt_answer, answer_type)
            sem = count_semantic_units(result["output"])
            density = compute_info_density(result["output"], result["output_tokens"])

            record = {
                "idx": item["idx"],
                "level": level,
                "model": model_name,
                "question": item.get("question_raw", item.get("question_formatted", "")),
                "gt_answer": gt_answer,
                "output": result["output"],
                "predicted_answer": predicted,
                "correct": correct,
                "input_tokens": result["input_tokens"],
                "output_tokens": result["output_tokens"],
                "latency_s": result["latency_s"],
                "semantic_units": sem,
                "info_density": density,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                # multi-dataset / multi-condition metadata
                "dataset": args.dataset,
                "condition": args.condition,
                "answer_type": answer_type,
                "question_formatted": item.get("question_formatted", item.get("question_raw", "")),
                "user_message": user_message,
                "system_prompt_kind": "neutral" if args.condition == "input" else f"caveman_{level}",
                "start_idx_filter": args.start_idx,
                "end_idx_filter": args.end_idx,
            }
            fout.write(json.dumps(record) + "\n")
            fout.flush()

            processed += 1
            if correct:
                correct_count += 1

            if processed % 10 == 0 or processed == len(todo):
                acc = correct_count / processed
                print(
                    f"[{level}] {processed}/{len(todo)} "
                    f"acc={acc:.3f}  out_tok={result['output_tokens']:3d}  "
                    f"info_den={density:.3f}  pred={predicted!r} gt={item['answer_gt']!r}"
                )

    return _read_records(out_path)


# ---------- final summary ----------

def print_summary_table(per_level: Dict[str, Dict]) -> None:
    bar = "=" * 78
    sep = "-" * 78
    print("\n" + bar)
    print(
        f"{'Level':<6} {'N':>6} {'Acc':>7} "
        f"{'OutTok':>8} {'InfoDen':>8} {'Extract':>9} {'L4Viol':>9}"
    )
    print(sep)
    for lvl in LEVEL_ORDER:
        s = per_level.get(lvl)
        if not s:
            continue
        print(
            f"{lvl:<6} {s['n']:>6d} {s['accuracy']:>7.3f} "
            f"{s['mean_output_tokens']:>8.1f} {s['mean_info_density']:>8.3f} "
            f"{s['answer_extraction_rate']*100:>8.1f}% "
            f"{s['l4_budget_violations']*100:>8.1f}%"
        )
    print(bar)


# ---------- main ----------

def main() -> None:
    args = parse_args()

    levels = list(LEVEL_ORDER) if args.level == "all" else [args.level]

    out_dir = Path(args.output).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Short tag for filenames + config; full directory name for the per-record "model" field.
    model_tag = args.model_name
    model_dir_name = os.path.basename(args.model_path.rstrip("/"))

    # Snapshot the exact prompt strings used by this run so the directory is
    # self-describing. Keeps record sizes flat (per-record system prompt would
    # add ~1 KB × N rows for zero new information).
    prompts_used = {
        "condition_a_constraint_prompts": {L: CONSTRAINT_PROMPTS[L] for L in levels},
        "condition_b_neutral_system_prompt": NEUTRAL_INPUT_SYSTEM_PROMPT,
        "active_condition": args.condition,
        "constraint_prompts_module": "src/constraint_prompts.py",
    }
    if args.condition == "input":
        prompts_used["spacy"] = _snapshot_spacy_for_input_condition()

    config = {
        "levels": levels,
        "n": args.n,
        "split": args.split,
        "output": str(out_dir),
        "model_name": model_tag,
        "model_path": args.model_path,
        "model_dir_name": model_dir_name,
        "no_resume": args.no_resume,
        "seed": args.seed,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "condition": args.condition,
        "start_idx": args.start_idx,
        "end_idx": args.end_idx,
        "dry_run": args.dry_run,
        "temperature": args.temperature,
        "paired_suffix": args.paired_suffix,
        "prompts_used": prompts_used,
        "provenance": capture_runtime_provenance(levels, args.condition, temperature=args.temperature),
    }
    # When --paired_suffix is set (paired-L0 noise-floor runs), keep summary +
    # config under a distinct filename so the canonical run's outputs are not
    # clobbered.
    config_path = out_dir / f"run_config{args.paired_suffix}.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"[run] config -> {config_path}")

    # Each dataset has a canonical split; if --split is the argparse default
    # ('test') but the dataset doesn't have that split, fall back to the
    # canonical one (e.g. boolq/commonsenseqa -> 'validation').
    _DATASET_CANONICAL_SPLIT = {
        "gsm8k": "test", "boolq": "validation", "arc_easy": "test",
        "commonsenseqa": "validation", "mmlu_stem": "test",
    }
    canonical_split = _DATASET_CANONICAL_SPLIT.get(args.dataset, args.split)
    effective_split = canonical_split if args.split == "test" and canonical_split != "test" else args.split

    print(f"[run] Loading dataset={args.dataset} split={effective_split} n={args.n} seed={args.seed}")
    items = load_dataset_caveman(
        name=args.dataset,
        split=effective_split,
        n=args.n,
        seed=args.seed,
    )
    print(f"[run] {len(items)} items loaded.")

    # Apply index range filter for chunked SLURM jobs.
    if args.start_idx is not None or args.end_idx is not None:
        start = args.start_idx if args.start_idx is not None else 0
        end = args.end_idx if args.end_idx is not None else max((it["idx"] for it in items), default=0)
        items = [it for it in items if start <= it["idx"] <= end]
        print(f"[CAVEWOMAN] Chunk filter applied: idx {start} to {end}, {len(items)} items")

    if args.dry_run:
        print("[dry_run] Skipping model load.")
        tokenizer, model = None, None
    else:
        # Let the loader sniff the family from the model path basename.
        tokenizer, model = load_model(args.model_path, model_type="auto")

    per_level_summary: Dict[str, Dict] = {}
    overall_start = time.time()
    # Seed the torch RNG so sampled generations are reproducible from --seed.
    # Greedy decoding (temperature=0) is already deterministic; the seed only
    # matters when --temperature > 0 (e.g. paired-L0 noise-floor runs).
    if args.temperature > 0.0 and not args.dry_run:
        try:
            import torch  # noqa: PLC0415
            torch.manual_seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)
        except Exception as e:  # pragma: no cover
            print(f"[run] WARN: could not seed torch RNG: {e}")

    for lvl in levels:
        out_path = out_dir / f"caveman_{model_tag}_{args.dataset}_{args.condition}_{lvl}{args.paired_suffix}.jsonl"
        print(f"\n[{lvl}] writing -> {out_path}")
        records = run_level(
            level=lvl,
            items=items,
            tokenizer=tokenizer,
            model=model,
            model_name=model_dir_name,
            out_path=out_path,
            resume=(not args.no_resume),
            args=args,
        )
        summary = summarize_level_results(records)
        per_level_summary[lvl] = summary
        print(f"[{lvl}] summary: {summary}")

    summary_path = out_dir / f"accuracy_summary{args.paired_suffix}.json"
    payload = {
        "config": config,
        "per_level": per_level_summary,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "wall_time_s": round(time.time(), overall_start, 2),
    }
    with open(summary_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[run] summary -> {summary_path}")
    print_summary_table(per_level_summary)


if __name__ == "__main__":
    main()
