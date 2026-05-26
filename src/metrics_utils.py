"""Metric and parsing utilities for the CAVEWOMAN experiment.

Pure stdlib (re, string, statistics). No torch, no transformers, safe to
import from any analysis script regardless of whether the LD_LIBRARY_PATH
fix has been applied.
"""

from __future__ import annotations

import re
import statistics
import string
from typing import Dict, List, Optional


# A numeric literal: optionally signed; either grouped-with-commas (1,234,567)
# or plain digits; optional decimal tail. Matches both '1,000.50' and '-42'.
_NUM_PAT = r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?"

# GSM8K-style trailing marker.
_GSM8K_FINAL = re.compile(r"####\s*(" + _NUM_PAT + r")")

# Strategy-2 "answer is N" / "= N" / "total: N" style patterns. Each captures
# the number into group 1. Searched case-insensitively; we keep the latest
# match across all patterns (rightmost wins).
_STRATEGY2_PATTERNS = [
    re.compile(r"final\s*answer\s*[:\-]?\s*\$?\s*(" + _NUM_PAT + r")", re.IGNORECASE),
    re.compile(r"answer\s*(?:is|=|:)\s*\$?\s*(" + _NUM_PAT + r")",     re.IGNORECASE),
    re.compile(r"total\s*(?::|is)?\s*\$?\s*(" + _NUM_PAT + r")",       re.IGNORECASE),
    re.compile(r"=\s*\$?\s*(" + _NUM_PAT + r")",                       re.IGNORECASE),
]

# Strategy-3 fallback: any number anywhere; we keep the last one.
_ANY_NUMBER = re.compile(_NUM_PAT)


STOPWORDS = frozenset({
    "a", "an", "the",
    "and", "or", "but",
    "in", "on", "at", "to", "for", "of", "with", "by", "from",
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "shall", "can",
    "that", "this", "it", "he", "she", "they", "we", "i", "you",
    "if", "as", "so", "then", "than",
})


def _clean_number(s: str) -> str:
    """Strip commas, currency symbols, and surrounding whitespace."""
    return s.replace(",", "").replace("$", "").strip()


def extract_numeric_answer(text: str) -> Optional[str]:
    """Pull the final numeric answer out of a model response.

    Three strategies, in priority order:
      1. GSM8K-style '#### N' marker (highest confidence).
      2. Rightmost match of an 'answer is N' / '= N' / 'total: N' style cue.
      3. Last standalone number anywhere in the text.

    Returns the number as a clean string (commas/'$' removed). Returns None
    if no number can be located.
    """
    if not text:
        return None

    # Strategy 1
    m = _GSM8K_FINAL.search(text)
    if m:
        return _clean_number(m.group(1))

    # Strategy 2: rightmost match across all cue patterns.
    best_pos = -1
    best_num: Optional[str] = None
    for pat in _STRATEGY2_PATTERNS:
        for m in pat.finditer(text):
            if m.start() >= best_pos:
                best_pos = m.start()
                best_num = m.group(1)
    if best_num is not None:
        return _clean_number(best_num)

    # Strategy 3
    matches = _ANY_NUMBER.findall(text)
    if matches:
        return _clean_number(matches[-1])

    return None


def count_tokens(text: str) -> int:
    """Whitespace-split token count. Used to police the L4 budget."""
    if not text:
        return 0
    return len(text.split())


def count_semantic_units(text: str) -> int:
    """Count words that are not in the stopword list.

    Each whitespace-delimited token is lowercased and stripped of leading/
    trailing punctuation before the stopword check. Numbers and bare math
    symbols survive and count as content.
    """
    if not text:
        return 0
    n = 0
    for raw in text.split():
        word = raw.strip(string.punctuation).lower()
        if word and word not in STOPWORDS:
            n += 1
    return n


def compute_info_density(text: str, token_count: int) -> float:
    """semantic_units / max(token_count, 1), rounded to 4 decimal places."""
    sem = count_semantic_units(text)
    denom = max(token_count, 1)
    return round(sem / denom, 4)


def check_l4_budget(text: str, budget: int = 15) -> bool:
    """True iff the whitespace token count fits within the L4 budget."""
    return count_tokens(text) <= budget


def summarize_level_results(records: List[Dict]) -> Dict:
    """Aggregate per-item results for a single constraint level."""
    n = len(records)
    if n == 0:
        return {
            "n": 0,
            "accuracy": 0.0,
            "mean_output_tokens": 0.0,
            "median_output_tokens": 0.0,
            "mean_info_density": 0.0,
            "answer_extraction_rate": 0.0,
            "l4_budget_violations": 0.0,
        }

    correct = sum(1 for r in records if r.get("correct"))
    out_tokens = [r.get("output_tokens", 0) for r in records]
    densities = [r.get("info_density", 0.0) for r in records]
    extracted = sum(1 for r in records if r.get("predicted_answer") is not None)
    violations = sum(
        1 for r in records if count_tokens(r.get("output", "") or "") > 15
    )

    return {
        "n": n,
        "accuracy": round(correct / n, 4),
        "mean_output_tokens": round(sum(out_tokens) / n, 2),
        "median_output_tokens": round(float(statistics.median(out_tokens)), 2),
        "mean_info_density": round(sum(densities) / n, 4),
        "answer_extraction_rate": round(extracted / n, 4),
        "l4_budget_violations": round(violations / n, 4),
    }


if __name__ == "__main__":
    print("=== extract_numeric_answer ===")
    extract_cases = [
        # (input, expected)
        ("Janet earned $18 a day.\n#### 18", "18"),
        ("clips April 48 May 24 total 72", "72"),
        ("72", "72"),
        ("The answer is 42.", "42"),
        ("Final answer: $1,200", "1200"),
        ("16, 3, 4 = 9. 9 * 2 = 18", "18"),
        ("She made $1,000.50 total.", "1000.50"),
        ("no numbers anywhere here", None),
        ("", None),
    ]
    for text, expected in extract_cases:
        got = extract_numeric_answer(text)
        mark = "OK  " if got == expected else "FAIL"
        print(f"  [{mark}] extract({text!r}) -> {got!r}  (expected {expected!r})")

    print("\n=== count_tokens ===")
    for t in ["72", "clips April 48 May 24 total 72", "  hello   world  ", ""]:
        print(f"  count_tokens({t!r}) = {count_tokens(t)}")

    print("\n=== count_semantic_units / info_density ===")
    sem_cases = [
        "The cat is on the mat",                       # 6 tokens, 2 content (cat, mat)
        "clips April 48 May 24 total 72",              # 7 tokens, 7 content (no stopwords)
        "Janet earned $18.",                           # 3 tokens, 2 content (janet, earned $18 -> '18' stripped)
        "Answer: 18",                                  # 2 tokens, 2 content (answer, 18)
    ]
    for t in sem_cases:
        tk = count_tokens(t)
        sem = count_semantic_units(t)
        dens = compute_info_density(t, tk)
        print(f"  text={t!r}\n    tokens={tk}  semantic={sem}  density={dens}")

    print("\n=== check_l4_budget (budget=15) ===")
    budget_cases = [
        "72",
        "Answer: 72",
        "16-3-4=9; 9*2=18. Answer: 18",
        "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen",
    ]
    for t in budget_cases:
        print(f"  tokens={count_tokens(t):2d}  ok={check_l4_budget(t)}  {t!r}")

    print("\n=== summarize_level_results ===")
    fake_records = [
        {"correct": True,  "output_tokens": 50, "info_density": 0.65,
         "predicted_answer": "18", "output": "Janet earned $18 per day. Answer: 18"},
        {"correct": False, "output_tokens": 80, "info_density": 0.60,
         "predicted_answer": "19", "output": "She earned $19. Answer: 19"},
        {"correct": True,  "output_tokens": 30, "info_density": 0.80,
         "predicted_answer": "72", "output": "Answer: 72"},
        {"correct": False, "output_tokens": 10, "info_density": 0.90,
         "predicted_answer": None, "output": ""},
    ]
    summary = summarize_level_results(fake_records)
    for k, v in summary.items():
        print(f"  {k}: {v}")

    print("\n  (empty input)")
    print(f"  {summarize_level_results([])}")
