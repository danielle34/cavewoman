"""Judge prompt templates for the CAVEWOMAN LLM-as-judge evaluation.

Two modes:

- `pair`     : given two reasoning traces (L0 unconstrained vs Lx compressed),
               decide whether they reach the same conclusion using consistent
               reasoning. Used as an 11th cross-validation of the
               accuracy/trace dissociation finding.

- `recovery` : given a question, the gold answer, and one reasoning trace,
               decide whether the trace actually arrives at the correct
               answer regardless of whether our regex could extract it.
               Used to recover signal from L0 records where extraction
               failed because the trace was verbose, malformed, or hit a
               token cap before producing a final "Answer:" line.

**Prompt design note (post-mortem 2026-06-03).**

The original prompts asked the model to reason first and emit a verdict
at the *end*. Qwen3.5-9B produced ~1100-char multi-paragraph analyses
that consistently exhausted a 300-token output budget before the verdict
line, leaving 96.3% of judgements unparseable.

The redesigned prompts emit `Verdict: <LABEL>` on the FIRST line,
followed by optional reasoning. This makes the parseable signal robust
to truncation, even at 100 tokens of output the verdict is captured.
We pair this with `max_new_tokens=800` in run_judge.py to give the model
room for explanation, but if it ever gets cut off the verdict is already
on disk.

Judge model: Qwen3.5-9B (open-source, locally loaded, no API spend).
"""
from __future__ import annotations

# Pair-comparison prompt (Use 1.5)

PAIR_SYSTEM = (
    "You are a precise evaluator of language-model reasoning. "
    "You will compare two reasoning traces produced by the SAME model on "
    "the SAME question under different output-format constraints. "
    "Your job is to decide whether the two traces reach the same final "
    "conclusion using semantically consistent reasoning, ignoring surface "
    "differences in wording, length, register, or formatting.\n\n"
    "CRITICAL: Your FIRST output line must be exactly one of:\n"
    "  Verdict: AGREE\n"
    "  Verdict: SAME_ANSWER\n"
    "  Verdict: DISAGREE\n"
    "  Verdict: UNCLEAR\n"
    "Optional reasoning may follow on subsequent lines, but the verdict "
    "line MUST come first."
)

PAIR_USER_TEMPLATE = """Question:
{question}

Trace A (unconstrained, written in full English sentences):
\"\"\"
{trace_a}
\"\"\"

Trace B (under an output-format constraint, may be terse / telegraphic / fragmented):
\"\"\"
{trace_b}
\"\"\"

Decide between these labels:

  AGREE       , same final answer AND substantively consistent reasoning
  SAME_ANSWER , same final answer BUT the reasoning differs in substance
                  (different key facts, different intermediate steps)
  DISAGREE    , different final answer
  UNCLEAR     , one or both traces are too fragmentary to judge

Rules:
- Be strict. If Trace B is so compressed that you cannot follow its
  reasoning, choose UNCLEAR rather than AGREE.
- If the two traces appeal to different facts or different intermediate
  steps to reach the same answer, choose SAME_ANSWER, not AGREE.

OUTPUT FORMAT (verdict FIRST, reasoning optional):

  Verdict: <one of AGREE | SAME_ANSWER | DISAGREE | UNCLEAR>
  Reasoning: <one or two short sentences explaining your verdict>

Begin your response with the verdict line."""

PAIR_LABELS = ["AGREE", "SAME_ANSWER", "DISAGREE", "UNCLEAR"]


# Trace-correctness recovery prompt (Use 2)

RECOVERY_SYSTEM = (
    "You are a precise evaluator of language-model reasoning. You will read "
    "one reasoning trace and decide whether it arrives at the correct answer "
    "to a question, regardless of how the answer is phrased or whether it "
    "appears on a separate line.\n\n"
    "CRITICAL: Your FIRST output line must be exactly one of:\n"
    "  Verdict: CORRECT\n"
    "  Verdict: INCORRECT\n"
    "  Verdict: NO_ANSWER\n"
    "  Verdict: AMBIGUOUS\n"
    "Optional reasoning may follow on subsequent lines, but the verdict "
    "line MUST come first."
)

RECOVERY_USER_TEMPLATE = """Question:
{question}

Choices:
{choices}

Correct answer: {gold_answer}

Reasoning trace produced by a language model:
\"\"\"
{trace}
\"\"\"

Decide whether the reasoning trace ends with, or unambiguously supports, 
the correct answer ({gold_answer}). The trace may not use the exact
format "Answer: X"; it may name the choice in words, abbreviate it, or
get cut off mid-sentence.

Decide between these labels:

  CORRECT     , trace arrives at the correct answer
  INCORRECT   , trace clearly arrives at a wrong answer
  NO_ANSWER   , trace stops before reaching any conclusion
                  (e.g. cut off mid-reasoning, no decisive claim)
  AMBIGUOUS   , trace gives multiple answers or contradicts itself

Rules:
- Be strict. If the trace is fragmentary and never commits to an answer,
  choose NO_ANSWER. Do not give credit for a trace that merely starts in
  the right direction without finishing.

OUTPUT FORMAT (verdict FIRST, reasoning optional):

  Verdict: <one of CORRECT | INCORRECT | NO_ANSWER | AMBIGUOUS>
  Reasoning: <one or two short sentences explaining your verdict>

Begin your response with the verdict line."""

RECOVERY_LABELS = ["CORRECT", "INCORRECT", "NO_ANSWER", "AMBIGUOUS"]


# Verdict extraction

import re

# Layered extraction: strict first, then fallbacks for models that don't
# fully honor the "verdict first" format.
#
# Strict: "Verdict: AGREE" with optional ** markdown
_VERDICT_STRICT_RE = re.compile(r"verdict\s*:\s*\*?\*?\s*([A-Z_]+)\b", re.IGNORECASE)


def _make_standalone_label_re(valid_labels: list[str]) -> "re.Pattern":
    """Compile a regex that matches any of the valid labels as a standalone
    word (word-boundaried). Used as a fallback when the model didn't follow
    the strict 'Verdict: LABEL' format but did write one of the labels
    somewhere in its analysis."""
    return re.compile(r"\b(" + "|".join(re.escape(lab) for lab in valid_labels) + r")\b")


def extract_verdict(text: str, valid_labels: list[str]) -> str | None:
    """Extract a verdict label from `text`. Two-layer strategy:

    1. STRICT: find 'Verdict: <LABEL>' (case-insensitive, optional markdown).
       Prefer the FIRST valid match, under the verdict-first prompt the
       canonical verdict is line 1, and the first match is most robust to
       truncation.
    2. FALLBACK: if no strict match, search the last 500 chars of `text`
       for any standalone valid label (e.g. ``it is AGREE``, ``the verdict
       is DISAGREE``). Prefer the LAST match, closer to the model's
       conclusion. Searches whole text if the tail has nothing.

    Returns None only if neither layer finds a valid label.
    """
    if not text:
        return None
    valid_set = set(valid_labels)

    # Layer 1: strict 'Verdict: LABEL'
    for m in _VERDICT_STRICT_RE.findall(text):
        label = m.strip().upper()
        if label in valid_set:
            return label

    # Layer 2: standalone valid label in the conclusion region
    standalone_re = _make_standalone_label_re(valid_labels)
    tail = text[-500:] if len(text) > 500 else text
    matches = standalone_re.findall(tail)
    if matches:
        # Last match in the tail, closer to the model's stated conclusion
        return matches[-1]
    # Last-ditch: search whole text for any standalone label
    matches = standalone_re.findall(text)
    if matches:
        return matches[-1]
    return None
