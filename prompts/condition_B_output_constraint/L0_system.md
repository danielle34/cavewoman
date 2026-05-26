# L0, Unconstrained system prompt

Source: `src/constraint_prompts.py::CONSTRAINT_PROMPTS["L0"]`.

```
Answer the following question accurately.
Reason step by step in full, grammatical English sentences. Conclude with the final answer on its own line in the form 'Answer: <answer>'.

The final-line answer matches what the question asks for: a number for numeric questions, 'yes' or 'no' for yes/no questions, or a single letter (A, B, C, ...) for multiple-choice questions.
```

- Max output tokens: **400**.
- Register: full prose, complete chain-of-thought reasoning.
- Final-line answer required.
