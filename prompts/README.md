# prompts/

The exact system prompts used in both channels.

```
condition_A_input_compression/
  neutral_system.md   # one prompt, reused for every input-compressed level
condition_B_output_constraint/
  L0_system.md        # unconstrained baseline
  L1_system.md        # telegraphic
  L2_system.md        # keyword only
  L3_system.md        # noun-phrase skeleton
  L4_system.md        # hard 15-token budget
```

In Condition A (input compression) the user prompt is filtered through
a spaCy POS rule at level Lk and the system prompt stays neutral. In
Condition B (output constraint) the user prompt is left verbatim and
the system prompt carries the level-specific register restriction.
