"""Condition-B (output constraint) system prompts for MGSM.

Same structure as the English prompts (L0 unconstrained, L1 telegraphic,
L2 keyword-only), with two differences:

  1. A final line pins the response language to the question language
     (otherwise models often reason in English).
  2. The L1 function-word examples are language-adapted: the English
     closed-class examples are replaced with the target language's
     articles / conjunctions / adpositions so the register rule is
     meaningful in-language. The rule set itself is unchanged.

System prompts stay in English (one instrument across languages; avoids a
translation-quality confound). MGSM is numeric-answer math, so the
answer-format line is numeric-only.
"""

FUNCTION_WORDS = {
    "en": {
        "articles": "the, a, an",
        "conjunctions": "and, but, or, so",
        "adpositions": "of, in, to, for, at, with, from, by, on, per",
    },
    "de": {
        "articles": "der, die, das, ein, eine",
        "conjunctions": "und, aber, oder, also",
        "adpositions": "von, in, zu, fuer, bei, mit, aus, durch, auf, pro",
    },
    "zh": {
        "articles": "的, 了, 一个",
        "conjunctions": "和, 但是, 或者, 所以",
        "adpositions": "在, 对, 从, 给, 把, 被, 每",
    },
    "bn": {
        "articles": "একটা, টি, টা",
        "conjunctions": "এবং, কিন্তু, অথবা, তাই",
        "adpositions": "থেকে, দিয়ে, জন্য, সাথে, প্রতি",
    },
    "sw": {
        "articles": "(Swahili has no articles; also drop demonstratives: huyu, hii, ile)",
        "conjunctions": "na, lakini, au, kwa hivyo",
        "adpositions": "ya, wa, katika, kwa, kutoka, hadi, kila",
    },
    "es": {
        "articles": "el, la, los, las, un, una",
        "conjunctions": "y, pero, o, asi que",
        "adpositions": "de, en, a, para, con, por, desde, hasta",
    },
    "fr": {
        "articles": "le, la, les, un, une, des",
        "conjunctions": "et, mais, ou, donc",
        "adpositions": "de, a, en, pour, avec, dans, sur, par",
    },
    "ru": {
        "articles": "(Russian has no articles)",
        "conjunctions": "и, но, или, поэтому",
        "adpositions": "в, на, с, для, от, по, из, к",
    },
    "ja": {
        "articles": "(no articles; drop case particles)",
        "conjunctions": "そして, しかし, または, だから",
        "adpositions": "は, が, を, に, で, と, の, へ, も, から",
    },
    "th": {
        "articles": "(Thai has no articles)",
        "conjunctions": "และ, แต่, หรือ, ดังนั้น",
        "adpositions": "ใน, จาก, ของ, ให้, กับ, ที่, โดย",
    },
    "te": {
        "articles": "(Telugu has no articles)",
        "conjunctions": "మరియు, కానీ, లేదా, కాబట్టి",
        "adpositions": "లో, నుండి, కు, తో, కోసం, పై",
    },
}

LANG_NAMES = {
    "en": "English", "de": "German", "zh": "Chinese",
    "bn": "Bengali", "sw": "Swahili", "es": "Spanish",
    "fr": "French", "ru": "Russian", "ja": "Japanese",
    "th": "Thai", "te": "Telugu",
}

ANSWER_LINE = (
    "The final-line answer is a number (digits only, no units, no thousands "
    "separators)."
)


def lang_pin(lang: str) -> str:
    return (f"Respond in {LANG_NAMES[lang]}, the language of the question. "
            f"Only the final 'Answer:' line keeps the English word 'Answer'.")


def l0_prompt(lang: str) -> str:
    return (
        "Answer the following question accurately.\n"
        "Reason step by step in full, grammatical sentences. Conclude with "
        "the final answer on its own line in the form 'Answer: <answer>'.\n\n"
        f"{ANSWER_LINE}\n{lang_pin(lang)}"
    )


def l1_prompt(lang: str) -> str:
    fw = FUNCTION_WORDS[lang]
    return (
        "Answer the question under a TELEGRAPHIC constraint.\n\n"
        "Rules:\n"
        f"  - DO NOT use any function words. No articles ({fw['articles']}). "
        f"No conjunctions ({fw['conjunctions']}). "
        f"No adpositions or case particles ({fw['adpositions']}).\n"
        "  - DO use nouns, main verbs, numbers, and standard symbols "
        "(+, -, *, /, =).\n"
        "  - Show each reasoning step.\n"
        "  - End with a line: 'Answer: <answer>'.\n\n"
        "Example (task-neutral):\n"
        "  Eggs 16. Breakfast 3. Muffins 4. Rest 16-3-4=9. 9*2=18.\n"
        "  Answer: 18\n\n"
        f"{ANSWER_LINE}\n{lang_pin(lang)}"
    )


def l2_prompt(lang: str) -> str:
    return (
        "Answer the question under a KEYWORD-ONLY constraint.\n\n"
        "Rules:\n"
        "  - Use ONLY nouns and main verbs. No grammar, no full sentences.\n"
        "  - Output as fragments, short labels, or list items.\n"
        "  - Numbers and standard symbols (+, -, *, /, =) are allowed.\n"
        "  - Each reasoning step appears as a fragment.\n"
        "  - End with a line: 'Answer: <answer>'.\n\n"
        "Example:\n"
        "  Eggs: 16\n  Eaten: 3\n  Muffins: 4\n  Sold: 9\n  9*2 = 18\n"
        "  Answer: 18\n\n"
        f"{ANSWER_LINE}\n{lang_pin(lang)}"
    )


def l3_prompt(lang: str) -> str:
    return (
        "Answer the question under a NOUN-PHRASE SKELETON constraint.\n\n"
        "Rules:\n"
        "  - NO verbs of any kind. None.\n"
        "  - Use only nominal fragments: nouns, noun compounds, numbers, and "
        "standard symbols (+, -, *, /, =).\n"
        "  - Each step is a noun phrase labeling a quantity, claim, or property.\n"
        "  - End with a line: 'Answer: <answer>'.\n\n"
        "Example:\n"
        "  Eggs: 16\n  Breakfast: 3\n  Muffins: 4\n  Remainder: 9\n"
        "  Price each: 2\n  Total: 18\n  Answer: 18\n\n"
        f"{ANSWER_LINE}\n{lang_pin(lang)}"
    )


def l4_prompt(lang: str) -> str:
    return (
        "Answer the question under a HARD TOKEN BUDGET.\n\n"
        "Rules:\n"
        "  - Your ENTIRE response must be 15 tokens or fewer.\n"
        "  - The response MUST include the final answer.\n"
        "  - Prefer the raw answer over prose.\n\n"
        "Example: 'Answer: 18'\n\n"
        f"{ANSWER_LINE}\n{lang_pin(lang)}"
    )


def system_prompt(level: str, lang: str) -> str:
    if level == "L0":
        return l0_prompt(lang)
    if level == "L1":
        return l1_prompt(lang)
    if level == "L2":
        return l2_prompt(lang)
    if level == "L3":
        return l3_prompt(lang)
    if level == "L4":
        return l4_prompt(lang)
    raise ValueError(f"unsupported level {level}")


MAX_NEW_TOKENS = {"L0": 400, "L1": 300, "L2": 200, "L3": 150, "L4": 20}
