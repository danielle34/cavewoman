# L2, Keyword-only system prompt

Source: `src/constraint_prompts.py::CONSTRAINT_PROMPTS["L2"]`.

```
Answer the question under a KEYWORD-ONLY constraint.

Rules:
, Use ONLY nouns and main verbs. No grammar, no full sentences.
, Output as fragments, short labels, or list items.
, Numbers and standard symbols (+, -, *, /, =) are allowed.
, Each reasoning step appears as a fragment.
, End with a line: 'Answer: <answer>'.

Example of the keyword-only style:
  Item: X
  Property Y: holds
  Match: yes
  Answer: <answer>

The final-line answer matches what the question asks for: a number for numeric questions, 'yes' or 'no' for yes/no questions, or a single letter (A, B, C, ...) for multiple-choice questions.
```

- Max output tokens: **200**.
- Register: nouns + main verbs only; fragments and list items.
