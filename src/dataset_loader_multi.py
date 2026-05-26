"""Multi-dataset loader for CAVEWOMAN (5 datasets, 3 answer types, input compression).

Supported datasets:
, gsm8k         (openai/gsm8k, main, test, ~1319)            numeric
, boolq         (google/boolq,        validation, ~3270)     boolean
, arc_easy      (allenai/ai2_arc, ARC-Easy, test, ~2376)     multiple_choice
, commonsenseqa (tau/commonsense_qa,  validation, ~1221)     multiple_choice
, mmlu_stem     (cais/mmlu, all, test, STEM-filtered)        multiple_choice

Public API:
    load_dataset_caveman(name, split=None, n=None, seed=42) -> List[Dict]
    extract_answer(text, answer_type) -> Optional[str]
    check_correct(predicted, ground_truth, answer_type) -> bool
    compress_input(text, level) -> str

Pure CPU-side code: only `datasets` + `spacy`. No torch.
"""

from __future__ import annotations

import logging
import random
import re
from typing import Dict, List, Optional, Sequence

from datasets import load_dataset

LOG = logging.getLogger(__name__)


# spaCy lazy loader (input compression)

_NLP = None


def _get_spacy():
    """Load en_core_web_sm on first use (no parser/NER, only POS tagging)."""
    global _NLP
    if _NLP is None:
        import spacy
        try:
            _NLP = spacy.load(
                "en_core_web_sm",
                disable=["parser", "ner", "lemmatizer", "attribute_ruler"],
            )
        except OSError as e:
            raise RuntimeError(
                "spaCy model 'en_core_web_sm' not installed. "
                "Run: python -m spacy download en_core_web_sm"
            ) from e
    return _NLP


# GSM8K-style numeric ground-truth extraction

_NUM_PAT = r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?"
_NUM_RE = re.compile(_NUM_PAT)
_GSM8K_FINAL = re.compile(r"####\s*(" + _NUM_PAT + r")")


def _extract_gt_numeric(answer_str: str) -> Optional[str]:
    if not answer_str:
        return None
    m = _GSM8K_FINAL.search(answer_str)
    if not m:
        return None
    return m.group(1).replace(",", "").strip()


# MC formatting helper

def _format_mc(question: str, texts: Sequence[str], labels: Sequence[str]) -> str:
    """Format Question + lettered choices as one user-message string."""
    lines = [f"Question: {question}"]
    for lbl, txt in zip(labels, texts):
        lines.append(f"{lbl}: {txt}")
    return "\n".join(lines)


def _normalize_arc_labels(labels: Sequence[str], n_choices: int) -> List[str]:
    """ARC sometimes uses numeric labels ('1','2','3','4'); remap to letters."""
    labels = list(labels)
    if labels and all(str(x).isalpha() for x in labels):
        return labels
    # Numeric or mixed → positional A..E
    return ["A", "B", "C", "D", "E"][:n_choices]


# Per-dataset loaders

def _sample_indices(total: int, n: Optional[int], seed: int) -> List[int]:
    if n is None or n >= total:
        return list(range(total))
    return random.Random(seed).sample(range(total), n)


def _load_gsm8k(split, n, seed):
    ds = load_dataset("openai/gsm8k", "main", split=split or "test")
    out = []
    for idx in _sample_indices(len(ds), n, seed):
        row = ds[idx]
        out.append({
            "idx": idx,
            "question_raw": row["question"],
            "question_formatted": row["question"],
            "answer_gt": _extract_gt_numeric(row["answer"]),
            "answer_type": "numeric",
            "dataset_name": "gsm8k",
        })
    return out


def _load_boolq(split, n, seed):
    ds = load_dataset("google/boolq", split=split or "validation")
    out = []
    for idx in _sample_indices(len(ds), n, seed):
        row = ds[idx]
        formatted = f"Passage: {row['passage']} Question: {row['question']}"
        gt = "yes" if bool(row["answer"]) else "no"
        out.append({
            "idx": idx,
            "question_raw": row["question"],
            "question_formatted": formatted,
            "answer_gt": gt,
            "answer_type": "boolean",
            "dataset_name": "boolq",
        })
    return out


def _load_arc_easy(split, n, seed):
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split=split or "test")
    out = []
    for idx in _sample_indices(len(ds), n, seed):
        row = ds[idx]
        choices = row["choices"]
        texts = list(choices.get("text") or [])
        raw_labels = list(choices.get("label") or [])
        labels = _normalize_arc_labels(raw_labels, len(texts))

        # answerKey may be a letter or a numeric string like '1'. Normalize to letter.
        gt = row.get("answerKey")
        if gt is not None and not str(gt).isalpha():
            try:
                pos = raw_labels.index(str(gt))
                gt = labels[pos] if pos < len(labels) else None
            except ValueError:
                gt = None
        if gt is not None:
            gt = str(gt).upper()

        out.append({
            "idx": idx,
            "question_raw": row["question"],
            "question_formatted": _format_mc(row["question"], texts, labels),
            "answer_gt": gt,
            "answer_type": "multiple_choice",
            "dataset_name": "arc_easy",
        })
    return out


def _load_commonsenseqa(split, n, seed):
    ds = load_dataset("tau/commonsense_qa", split=split or "validation")
    out = []
    for idx in _sample_indices(len(ds), n, seed):
        row = ds[idx]
        choices = row["choices"]
        texts = list(choices.get("text") or [])
        labels = list(choices.get("label") or ["A", "B", "C", "D", "E"][: len(texts)])
        gt = row.get("answerKey")
        gt = str(gt).upper() if gt else None
        out.append({
            "idx": idx,
            "question_raw": row["question"],
            "question_formatted": _format_mc(row["question"], texts, labels),
            "answer_gt": gt,
            "answer_type": "multiple_choice",
            "dataset_name": "commonsenseqa",
        })
    return out


MMLU_STEM_SUBJECTS = frozenset({
    "abstract_algebra", "anatomy", "astronomy", "college_biology",
    "college_chemistry", "college_computer_science", "college_mathematics",
    "college_physics", "computer_security", "conceptual_physics",
    "electrical_engineering", "elementary_mathematics", "formal_logic",
    "high_school_biology", "high_school_chemistry",
    "high_school_computer_science", "high_school_mathematics",
    "high_school_physics", "high_school_statistics", "machine_learning",
})

_LETTERS_4 = ["A", "B", "C", "D"]


def _load_mmlu_stem(split, n, seed):
    ds = load_dataset("cais/mmlu", "all", split=split or "test")
    # Filter to STEM by subject column.
    subjects = ds["subject"]
    stem_idx = [i for i, s in enumerate(subjects) if s in MMLU_STEM_SUBJECTS]
    if not stem_idx:
        raise RuntimeError("MMLU 'all' split contained no STEM subjects after filtering.")
    sampled = _sample_indices(len(stem_idx), n, seed)
    chosen = [stem_idx[i] for i in sampled]
    out = []
    for idx in chosen:
        row = ds[idx]
        texts = list(row["choices"])  # list of 4 strings
        labels = _LETTERS_4[: len(texts)]
        ans = row["answer"]
        gt = _LETTERS_4[ans] if isinstance(ans, int) and 0 <= ans < len(_LETTERS_4) else None
        out.append({
            "idx": idx,
            "question_raw": row["question"],
            "question_formatted": _format_mc(row["question"], texts, labels),
            "answer_gt": gt,
            "answer_type": "multiple_choice",
            "dataset_name": "mmlu_stem",
        })
    return out


_LOADERS = {
    "gsm8k":         _load_gsm8k,
    "boolq":         _load_boolq,
    "arc_easy":      _load_arc_easy,
    "commonsenseqa": _load_commonsenseqa,
    "mmlu_stem":     _load_mmlu_stem,
}


def load_dataset_caveman(name: str, split: Optional[str] = None,
                         n: Optional[int] = None, seed: int = 42) -> List[Dict]:
    """Dispatch by dataset name; return list of canonical CAVEWOMAN items.

    Each item is a dict with:
        idx, question_raw, question_formatted, answer_gt, answer_type, dataset_name
    """
    if name not in _LOADERS:
        raise ValueError(f"Unknown dataset {name!r}. Expected one of {sorted(_LOADERS)}.")
    return _LOADERS[name](split, n, seed)


# Answer extraction (3 answer types)

# Strategy-2 cue patterns shared with the original GSM8K extractor.
_S2_PATTERNS = [
    re.compile(r"final\s*answer\s*[:\-]?\s*\$?\s*(" + _NUM_PAT + r")", re.IGNORECASE),
    re.compile(r"answer\s*(?:is|=|:)\s*\$?\s*(" + _NUM_PAT + r")",     re.IGNORECASE),
    re.compile(r"total\s*(?::|is)?\s*\$?\s*(" + _NUM_PAT + r")",       re.IGNORECASE),
    re.compile(r"=\s*\$?\s*(" + _NUM_PAT + r")",                       re.IGNORECASE),
]


def _extract_numeric(text: str) -> Optional[str]:
    if not text:
        return None
    m = _GSM8K_FINAL.search(text)
    if m:
        return m.group(1).replace(",", "").strip()
    best_pos, best = -1, None
    for pat in _S2_PATTERNS:
        for m in pat.finditer(text):
            if m.start() >= best_pos:
                best_pos = m.start()
                best = m.group(1)
    if best is not None:
        return best.replace(",", "").strip()
    matches = _NUM_RE.findall(text)
    if matches:
        return matches[-1].replace(",", "").strip()
    return None


_YES_RE = re.compile(r"\byes\b", re.IGNORECASE)
_NO_RE  = re.compile(r"\bno\b",  re.IGNORECASE)


def _extract_boolean(text: str) -> Optional[str]:
    if not text:
        return None
    y = _YES_RE.search(text)
    n = _NO_RE.search(text)
    if y and n:
        return "yes" if y.start() < n.start() else "no"
    if y:
        return "yes"
    if n:
        return "no"
    return None


_MC_CUE_PATTERNS = [
    re.compile(r"\bfinal\s*answer\s*[:\-]?\s*\(?\s*([A-E])\b", re.IGNORECASE),
    re.compile(r"\banswer\s*(?:is|=|:)\s*\(?\s*([A-E])\b",     re.IGNORECASE),
    re.compile(r"\boption\s*\(?\s*([A-E])\b",                  re.IGNORECASE),
    re.compile(r"\bchoice\s*\(?\s*([A-E])\b",                  re.IGNORECASE),
]
_MC_START_RE = re.compile(r"^\(?([A-E])\)?\b")
_MC_END_RE   = re.compile(r"\b([A-E])\)?\.?\s*$")


def _extract_multiple_choice(text: str) -> Optional[str]:
    if not text:
        return None
    # 1) cue phrases, take the latest match (rightmost wins).
    best_pos, best = -1, None
    for pat in _MC_CUE_PATTERNS:
        for m in pat.finditer(text):
            if m.start() >= best_pos:
                best_pos = m.start()
                best = m.group(1).upper()
    if best:
        return best
    # 2) isolated letter at start of trimmed text
    s = text.strip()
    m = _MC_START_RE.match(s)
    if m:
        return m.group(1).upper()
    # 3) isolated letter at end
    m = _MC_END_RE.search(s)
    if m:
        return m.group(1).upper()
    return None


def extract_answer(text: str, answer_type: str) -> Optional[str]:
    if answer_type == "numeric":
        return _extract_numeric(text)
    if answer_type == "boolean":
        return _extract_boolean(text)
    if answer_type == "multiple_choice":
        return _extract_multiple_choice(text)
    raise ValueError(f"Unknown answer_type: {answer_type!r}")


# Correctness comparison

def _to_float(s) -> Optional[float]:
    try:
        return float(str(s).replace(",", "").replace("$", "").strip())
    except (ValueError, TypeError):
        return None


def check_correct(predicted, ground_truth, answer_type: str) -> bool:
    if predicted is None or ground_truth is None:
        return False
    if answer_type == "numeric":
        p, g = _to_float(predicted), _to_float(ground_truth)
        if p is None or g is None:
            return False
        return abs(p, g) < 0.01
    if answer_type == "boolean":
        return str(predicted).strip().lower() == str(ground_truth).strip().lower()
    if answer_type == "multiple_choice":
        return str(predicted).strip().upper() == str(ground_truth).strip().upper()
    raise ValueError(f"Unknown answer_type: {answer_type!r}")


# Input compression (Condition B), spaCy POS-tag filtering

# L1: drop function words (determiners, prepositions, conjunctions, particles,
# infinitival 'to', modals).
_L1_DROP_TAGS = frozenset({"DT", "IN", "CC", "RP", "TO", "MD"})
# L2: keep nouns + main verbs + numbers.
_L2_KEEP_TAGS = frozenset({"NN", "NNS", "NNP", "NNPS",
                           "VB", "VBZ", "VBD", "VBN", "VBG",
                           "CD"})
# L3: keep nouns + numbers only (no verbs).
_L3_KEEP_TAGS = frozenset({"NN", "NNS", "NNP", "NNPS", "CD"})


def compress_input(text: str, level: str) -> str:
    """Apply Condition-B input compression at the named level."""
    if level == "L0" or not text:
        return text

    if level == "L4":
        # L4 is L3 truncated to 15 tokens, so it's a strict subset of L3 (the
        # CAVEWOMAN ladder is monotone). Previously L4 truncated the raw text,
        # which kept function words / verbs / adjectives and broke the ladder.
        l3_text = compress_input(text, "L3")
        tokens = l3_text.split()
        result = " ".join(tokens[:15]).strip()
        if not result:
            LOG.warning("compress_input L4 produced empty result; returning original.")
            return text
        return result

    if level not in {"L1", "L2", "L3"}:
        raise ValueError(f"Unknown compression level: {level!r}")

    nlp = _get_spacy()
    doc = nlp(text)

    if level == "L1":
        kept = [t.text for t in doc if not t.is_space and t.tag_ not in _L1_DROP_TAGS]
    elif level == "L2":
        kept = [t.text for t in doc if t.tag_ in _L2_KEEP_TAGS]
    else:  # L3
        kept = [t.text for t in doc if t.tag_ in _L3_KEEP_TAGS]

    result = " ".join(kept).strip()
    if not result:
        LOG.warning(
            "compress_input %s produced empty result for text of length %d; returning original.",
            level, len(text),
        )
        return text
    return result


# Self-test

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    bar = "=" * 78
    print(bar)
    print("Multi-dataset loader self-test (n=3 per dataset)".center(78))
    print(bar)

    first_gsm8k = None
    for name in ["gsm8k", "boolq", "arc_easy", "commonsenseqa", "mmlu_stem"]:
        print(f"\n--- {name} ---")
        items = load_dataset_caveman(name, n=3)
        for i, it in enumerate(items):
            fq = it["question_formatted"]
            preview = (fq[:240] + "…") if len(fq) > 240 else fq
            print(f"  [{i}] idx={it['idx']}  type={it['answer_type']}  gt={it['answer_gt']!r}")
            for line in preview.splitlines():
                print(f"        {line}")
        if name == "gsm8k" and items:
            first_gsm8k = items[0]

    print("\n" + bar)
    print("compress_input on the first GSM8K question (all 5 levels)".center(78))
    print(bar)
    text = first_gsm8k["question_formatted"]
    print(f"\n[original]  ({len(text.split())} tokens)\n  {text}\n")
    for lvl in ["L0", "L1", "L2", "L3", "L4"]:
        c = compress_input(text, lvl)
        print(f"[{lvl}] ({len(c.split())} tokens)\n  {c}\n")

    print(bar)
    print("extract_answer / check_correct sanity".center(78))
    print(bar)
    cases = [
        # text, type, gt, expect_correct
        ("Janet earned $18 total.\n#### 18",         "numeric",         "18",  True),
        ("The result is 1,200 dollars.",             "numeric",         "1200", True),
        ("she made eighteen dollars",                 "numeric",         "18",  False),  # word, not digit
        ("Yes, that's right.",                        "boolean",         "yes", True),
        ("No, the passage doesn't support that.",     "boolean",         "no",  True),
        ("I'm not sure.",                             "boolean",         "no",  False),
        ("Final answer: B",                           "multiple_choice", "B",   True),
        ("The answer is (C).",                        "multiple_choice", "C",   True),
        ("A",                                         "multiple_choice", "A",   True),
        ("...so the answer must be D.",               "multiple_choice", "D",   True),
        ("I have no idea.",                           "multiple_choice", "A",   False),
    ]
    for text, atype, gt, expected in cases:
        pred = extract_answer(text, atype)
        ok = check_correct(pred, gt, atype)
        mark = "OK  " if ok == expected else "FAIL"
        snip = text.replace("\n", " / ")[:60]
        print(f"  [{mark}] type={atype:<16} pred={str(pred):<6} gt={gt:<6} text={snip!r}")

    print("\n[done] self-test complete.")
