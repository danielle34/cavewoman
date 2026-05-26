# Condition B, Output constraint

Question unchanged. System prompt changes per level to constrain output
register.

| Level | Register | Max out |
|---|---|---|
| L0 | full prose, "Answer: X" at end | 400 |
| L1 | telegraphic, no articles/conjunctions/preps | 300 |
| L2 | nouns + main verbs + numbers | 200 |
| L3 | no verbs | 150 |
| L4 | response ≤ 15 tokens | 20 |

Verbatim prompts: [`prompts/condition_B_output_constraint/`](../../prompts/condition_B_output_constraint/).

## Worked example

Q: *"What is the capital of France?"*

| Level | Response |
|---|---|
| L0 | "The capital of France is Paris. Answer: Paris" |
| L1 | "France capital is Paris. Answer: Paris" |
| L2 | "France capital: Paris. Answer: Paris" |
| L3 | "France capital: Paris. Answer: Paris" |
| L4 | "Answer: Paris" |

## Task-neutral prompts

Prompts must work across math, yes/no, and MC. Shared answer-format hint:
"number / yes-no / single letter". Avoids leaking task framing.
