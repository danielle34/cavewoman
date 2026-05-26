# L1, Telegraphic system prompt

Source: `src/constraint_prompts.py::CONSTRAINT_PROMPTS["L1"]`.

```
Answer the question under a TELEGRAPHIC constraint.

Rules:
, DO NOT use any function words. No articles (the, a, an). No conjunctions (and, but, or, so). No prepositions (of, in, to, for, at, with, from, by, on, per).
, DO use nouns, main verbs, numbers, and standard symbols (+, -, *, /, =).
, Show each reasoning step.
, End with a line: 'Answer: <answer>'.

Example of the telegraphic style (task-neutral demonstration):
  Premise mentions item X. Property Y holds X. Match: yes.
  Answer: <answer>

The final-line answer matches what the question asks for: a number for numeric questions, 'yes' or 'no' for yes/no questions, or a single letter (A, B, C, ...) for multiple-choice questions.
```

- Max output tokens: **300**.
- Register: no function words; nouns + main verbs + numbers + symbols.
- Task-neutral example given in the prompt itself.
