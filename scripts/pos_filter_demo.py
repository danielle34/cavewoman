#!/usr/bin/env python3
"""CAVEWOMAN, POS-filter demo.

A self-contained illustration of the Condition A POS filter at L0..L4.
Requires only spaCy and the en_core_web_sm pipeline:

    pip install spacy
    python -m spacy download en_core_web_sm

Usage:

    python scripts/pos_filter_demo.py \\
        --text "What is the capital of France?" \\
        --level L0,L1,L2,L3,L4

If --text is omitted, a built-in example is used.

The rules are reproduced verbatim from
src/dataset_loader_multi.py::compress_input() in the working
research repo. See also:
prompts/condition_A_input_compression/README.md.
"""

from __future__ import annotations

import argparse
import sys
from typing import Iterable

DROP_TAGS_L1 = frozenset({"DT", "IN", "CC", "RP", "TO", "MD"})
KEEP_TAGS_L2 = frozenset(
    {"NN", "NNS", "NNP", "NNPS", "VB", "VBZ", "VBD", "VBN", "VBG", "CD"}
)
KEEP_TAGS_L3 = frozenset({"NN", "NNS", "NNP", "NNPS", "CD"})


def _load_nlp():
    try:
        import spacy
    except ImportError as e:
        sys.exit("spaCy is required. Install with: pip install spacy")
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        sys.exit(
            "spaCy model en_core_web_sm not found.\n"
            "Install with: python -m spacy download en_core_web_sm"
        )


def compress(text: str, level: str, nlp=None) -> str:
    """Apply the Condition A compression at the given level.

    Returns the original text if the level produces an empty string,
    matching the working-repo runner's empty-after-filter fallback.
    """
    if level == "L0":
        return text
    if level == "L4":
        toks = text.split()
        return " ".join(toks[:15])

    if nlp is None:
        nlp = _load_nlp()
    doc = nlp(text)

    if level == "L1":
        kept = [t.text for t in doc if t.tag_ not in DROP_TAGS_L1]
    elif level == "L2":
        kept = [t.text for t in doc if t.tag_ in KEEP_TAGS_L2]
    elif level == "L3":
        kept = [t.text for t in doc if t.tag_ in KEEP_TAGS_L3]
    else:
        raise ValueError(f"Unknown level: {level}")

    out = " ".join(kept).strip()
    return out if out else text


def parse_levels(s: str) -> Iterable[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--text",
        default=(
            "If a train leaves Boston at 8am going 60mph, "
            "when does it reach New York?"
        ),
        help="Text to compress. Defaults to a built-in example.",
    )
    parser.add_argument(
        "--level",
        default="L0,L1,L2,L3,L4",
        help="Comma-separated list of levels to show. Default: all five.",
    )
    args = parser.parse_args()

    nlp = _load_nlp()
    print(f"Input: {args.text!r}\n")
    for lvl in parse_levels(args.level):
        out = compress(args.text, lvl, nlp=nlp)
        print(f"{lvl}: {out}")


if __name__ == "__main__":
    main()
