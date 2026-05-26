"""CAVEWOMAN constraint-level system prompts (task-agnostic).

Five levels of progressively tighter linguistic constraint on the model's
response. Each prompt instructs the model how to format its reasoning and
final answer.

These prompts are **task-neutral**: they describe only the linguistic
constraint and the required final-line format. They do not assume the
question is mathematical, factual, or any specific kind. The model is
expected to apply the linguistic rule to whatever the user's question
actually is (GSM8K math, BoolQ yes/no, ARC science MC, CommonsenseQA,
MMLU-STEM).

The answer-format directive 'Answer: <answer>' accepts:
, a number for numeric questions     (GSM8K)
, 'yes' or 'no' for boolean          (BoolQ)
, a single letter (A, B, C, ...) for multiple choice
                                       (ARC-Easy, CommonsenseQA, MMLU-STEM)

The post-record extractor (`dataset_loader_multi.extract_answer`)
branches on `answer_type` and finds the right token regardless of the
exact phrasing.

Usage:
    from constraint_prompts import CONSTRAINT_PROMPTS, get_max_tokens
    system = CONSTRAINT_PROMPTS["L2"]
    max_new = get_max_tokens("L2")

History: pre-2026-05-12 versions of these prompts framed every level as a
"grade-school math word problem" and used a Janet/ducks worked example.
That confounded non-math datasets, the model was being told to do math
on yes/no questions. The generalised prompts in this file restore
task-agnosticism. See git log for the original math-themed prompts.
"""

LEVEL_ORDER = ["L0", "L1", "L2", "L3", "L4"]

LEVEL_DESCRIPTIONS = {
    "L0": "Unconstrained, full sentences, complete chain-of-thought reasoning.",
    "L1": "Telegraphic, no function words (articles, conjunctions, prepositions).",
    "L2": "Keyword-only, nouns and main verbs only, fragments and lists.",
    "L3": "Noun-phrase skeleton, no verbs, only nominal fragments and numbers.",
    "L4": "Hard token budget, entire response must be 15 tokens or fewer.",
}


# Shared answer-format reminder appended where useful. Kept concise so it
# doesn't dominate the constraint description.
_ANSWER_FORMAT_HINT = (
    "The final-line answer matches what the question asks for: a number for "
    "numeric questions, 'yes' or 'no' for yes/no questions, or a single "
    "letter (A, B, C, ...) for multiple-choice questions."
)


CONSTRAINT_PROMPTS = {
    "L0": (
        "Answer the following question accurately.\n"
        "Reason step by step in full, grammatical English sentences. "
        "Conclude with the final answer on its own line in the form "
        "'Answer: <answer>'.\n"
        "\n"
        f"{_ANSWER_FORMAT_HINT}"
    ),

    "L1": (
        "Answer the question under a TELEGRAPHIC constraint.\n"
        "\n"
        "Rules:\n"
        ", DO NOT use any function words. No articles (the, a, an). "
        "No conjunctions (and, but, or, so). No prepositions (of, in, to, "
        "for, at, with, from, by, on, per).\n"
        ", DO use nouns, main verbs, numbers, and standard symbols "
        "(+, -, *, /, =).\n"
        ", Show each reasoning step.\n"
        ", End with a line: 'Answer: <answer>'.\n"
        "\n"
        "Example of the telegraphic style (task-neutral demonstration):\n"
        "  Premise mentions item X. Property Y holds X. Match: yes.\n"
        "  Answer: <answer>\n"
        "\n"
        f"{_ANSWER_FORMAT_HINT}"
    ),

    "L2": (
        "Answer the question under a KEYWORD-ONLY constraint.\n"
        "\n"
        "Rules:\n"
        ", Use ONLY nouns and main verbs. No grammar, no full sentences.\n"
        ", Output as fragments, short labels, or list items.\n"
        ", Numbers and standard symbols (+, -, *, /, =) are allowed.\n"
        ", Each reasoning step appears as a fragment.\n"
        ", End with a line: 'Answer: <answer>'.\n"
        "\n"
        "Example of the keyword-only style:\n"
        "  Item: X\n"
        "  Property Y: holds\n"
        "  Match: yes\n"
        "  Answer: <answer>\n"
        "\n"
        f"{_ANSWER_FORMAT_HINT}"
    ),

    "L3": (
        "Answer the question under a NOUN-PHRASE SKELETON constraint.\n"
        "\n"
        "Rules:\n"
        ", NO verbs of any kind. None.\n"
        ", Use only nominal fragments: nouns, noun compounds, numbers, "
        "and standard symbols (+, -, *, /, =).\n"
        ", Each step is a noun phrase labelling a quantity, claim, or "
        "property.\n"
        ", End with a line: 'Answer: <answer>'.\n"
        "\n"
        "Example of the noun-phrase skeleton style:\n"
        "  Item: X\n"
        "  Property in question: Y\n"
        "  Match status: positive\n"
        "  Answer: <answer>\n"
        "\n"
        f"{_ANSWER_FORMAT_HINT}"
    ),

    "L4": (
        "Answer the question under a HARD TOKEN BUDGET.\n"
        "\n"
        "Rules:\n"
        ", Your ENTIRE response must be 15 tokens or fewer.\n"
        ", The response MUST include the final answer.\n"
        ", Prefer the raw answer over prose.\n"
        "\n"
        "Example: 'Answer: <answer>'\n"
        "\n"
        f"{_ANSWER_FORMAT_HINT}"
    ),
}


_MAX_TOKENS = {
    "L0": 400,
    "L1": 300,
    "L2": 200,
    "L3": 150,
    "L4": 20,
}


def get_max_tokens(level: str) -> int:
    """Return the max_new_tokens budget for a given constraint level."""
    if level not in _MAX_TOKENS:
        raise ValueError(
            f"Unknown constraint level: {level!r}. Expected one of {LEVEL_ORDER}."
        )
    return _MAX_TOKENS[level]


if __name__ == "__main__":
    for lvl in LEVEL_ORDER:
        print(f"=== {lvl} ({LEVEL_DESCRIPTIONS[lvl]}) ===")
        print(f"max_new_tokens = {get_max_tokens(lvl)}")
        print(CONSTRAINT_PROMPTS[lvl])
        print()
