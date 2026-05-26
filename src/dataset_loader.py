"""GSM8K loader and answer-comparison helpers for the CAVEWOMAN experiment.

GSM8K answers end with a line of the form '#### N', where N is the final
numeric answer (may contain commas, a leading sign, or a decimal point).
We strip everything except a clean number string for evaluation.
"""

from __future__ import annotations

import random
import re
from typing import List, Dict, Optional

from datasets import load_dataset


_GT_PATTERN = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")
_NUM_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")


def extract_ground_truth(answer_str: str) -> Optional[str]:
    """Extract the numeric answer from GSM8K's '#### N' format.

    Returns the number as a clean string with commas stripped, or None
    if no '#### N' marker is found.
    """
    if not answer_str:
        return None
    m = _GT_PATTERN.search(answer_str)
    if not m:
        return None
    return m.group(1).replace(",", "").strip()


def load_gsm8k(split: str = "test", n: Optional[int] = None, seed: int = 42) -> List[Dict]:
    """Load GSM8K and return a list of {idx, question, answer_raw, answer_gt}.

    If n is given and smaller than the split size, sample n items uniformly
    at random with the supplied seed. The seed is used only for sampling;
    the underlying dataset order is preserved otherwise.
    """
    ds = load_dataset("openai/gsm8k", "main", split=split)

    indices = list(range(len(ds)))
    if n is not None and n < len(ds):
        rng = random.Random(seed)
        indices = rng.sample(indices, n)

    out: List[Dict] = []
    for idx in indices:
        row = ds[idx]
        question = row["question"]
        answer_raw = row["answer"]
        out.append(
            {
                "idx": idx,
                "question": question,
                "answer_raw": answer_raw,
                "answer_gt": extract_ground_truth(answer_raw),
            }
        )
    return out


def _to_float(s: str) -> Optional[float]:
    """Parse the first signed number in s as float, or return None."""
    if s is None:
        return None
    s = s.replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        pass
    m = _NUM_PATTERN.search(s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def check_answer_correct(predicted: Optional[str], ground_truth: Optional[str]) -> bool:
    """True if predicted matches ground_truth numerically or as strings.

    Numeric comparison is preferred (with a small tolerance for floats).
    Falls back to exact string comparison after stripping. Returns False
    if either argument is None.
    """
    if predicted is None or ground_truth is None:
        return False

    p_num = _to_float(predicted)
    g_num = _to_float(ground_truth)
    if p_num is not None and g_num is not None:
        return abs(p_num, g_num) < 1e-6

    return predicted.strip() == ground_truth.strip()


if __name__ == "__main__":
    print("Loading 5 GSM8K test items...")
    items = load_gsm8k(split="test", n=5, seed=42)
    for it in items:
        print("-" * 60)
        print(f"idx        : {it['idx']}")
        print(f"question   : {it['question'][:120]}{'...' if len(it['question']) > 120 else ''}")
        print(f"answer_gt  : {it['answer_gt']}")
        print(f"answer_raw : {it['answer_raw'][-80:]}")
    print("-" * 60)

    print("\nQuick comparator checks:")
    cases = [
        ("18", "18", True),
        ("18.0", "18", True),
        ("1,000", "1000", True),
        ("$18", "18", True),
        ("nineteen", "18", False),
        (None, "18", False),
        ("18", None, False),
    ]
    for p, g, expected in cases:
        got = check_answer_correct(p, g)
        mark = "OK" if got == expected else "FAIL"
        print(f"  [{mark}] check_answer_correct({p!r}, {g!r}) -> {got} (expected {expected})")
