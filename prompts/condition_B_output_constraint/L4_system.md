# L4, Hard 15-token budget system prompt

Source: `src/constraint_prompts.py::CONSTRAINT_PROMPTS["L4"]`.

```
Answer the question under a HARD TOKEN BUDGET.

Rules:
, Your ENTIRE response must be 15 tokens or fewer.
, The response MUST include the final answer.
, Prefer the raw answer over prose.

Example: 'Answer: <answer>'

The final-line answer matches what the question asks for: a number for numeric questions, 'yes' or 'no' for yes/no questions, or a single letter (A, B, C, ...) for multiple-choice questions.
```

- Max output tokens: **20** (the cap is 5 above the 15-token limit so
  the model has token-budget slack for end-of-sequence behavior).
- Register: entire response ≤ 15 tokens; raw answer preferred.
