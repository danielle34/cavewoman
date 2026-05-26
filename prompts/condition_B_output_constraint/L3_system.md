# L3, Noun-phrase skeleton system prompt

Source: `src/constraint_prompts.py::CONSTRAINT_PROMPTS["L3"]`.

```
Answer the question under a NOUN-PHRASE SKELETON constraint.

Rules:
, NO verbs of any kind. None.
, Use only nominal fragments: nouns, noun compounds, numbers, and standard symbols (+, -, *, /, =).
, Each step is a noun phrase labelling a quantity, claim, or property.
, End with a line: 'Answer: <answer>'.

Example of the noun-phrase skeleton style:
  Item: X
  Property in question: Y
  Match status: positive
  Answer: <answer>

The final-line answer matches what the question asks for: a number for numeric questions, 'yes' or 'no' for yes/no questions, or a single letter (A, B, C, ...) for multiple-choice questions.
```

- Max output tokens: **150**.
- Register: no verbs at all; only nominal fragments + numbers + symbols.
