# Cond A, neutral system prompt (constant across L0..L4)

Under Condition A (input compression) the system prompt is **constant
across all five reduction levels**. Only the user message is filtered
through the deterministic POS rule at level Lₖ. The model is free to
respond however it wants.

Source: `src/constraint_prompts.py`. Echoed inline as
`prompts_used.condition_b_neutral_system_prompt` in every released
per-cell summary JSON.

```
You are a helpful assistant. Answer the following question accurately and completely.
```

- Used at: every level L0..L4.
- Companion to: the per-level POS-filter rules described in
  [`README.md`](README.md).
- Symmetric to: the five per-level system prompts under
  [`../condition_B_output_constraint/`](../condition_B_output_constraint/),
  which restrict the response register.
