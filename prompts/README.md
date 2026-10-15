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

![One question at all five levels](../figures/teaser.png)

## Using these without the rest of the repo

The five Condition B prompts are plain Markdown with no dependencies. This is
the cheapest way to try this on your own task: drop them into whatever
harness you already have, run your eval once per level, and compare cost and
accuracy across the ladder.

That gets you the cost and accuracy axes. The third axis, whether the
compressed answer still says what the model would have said unconstrained, needs
[`../evaluation/add_entailment_scores.py`](../evaluation/add_entailment_scores.py)
scored against your own L0 run. It is the measure that produces Finding 2, and
accuracy alone will not show it.

Note that L0 ships in two forms in the paper: the step-by-step prompt here, and
a neutral variant that asks only for the answer. Dropping the step-by-step
instruction alone accounts for part of the cost saving, so compare against
whichever one matches your production prompt.
