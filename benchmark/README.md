# benchmark/

The benchmark definition: what the five compression levels are, what
the two channels are, and the metadata that pins down which datasets,
models, and metrics are in scope. Every entry carries a `panel` field
saying whether it belongs to the main run, the size sweep, or one of
the extension panels.

![Framework](../figures/framework.png)

One question goes down one of two paths. **Condition A** filters the question
by part of speech before the model sees it. **Condition B** leaves the question
untouched and swaps the system prompt for one that demands the same reduction in
the response. Both run at five levels, L0 through L4, and every response is
scored on accuracy, realized cost, and agreement with that model's own
uncompressed answer.

Here is the L0-L4 input filter on a real GSM8K question, with the surviving
token count at each level:

![Input filter example](../figures/input_filter_example.png)

```
configs/
  compression_levels.yaml   # L0-L4 specifications shared by both channels
  example_run_config.json   # one cell's run configuration as a worked example
conditions/
  condition_A_input_compression.md   # input-compression channel definition
  condition_B_output_constraint.md   # output-constraint channel definition
metadata/
  conditions.json           # structured channel metadata
  datasets.json             # 8 datasets: 5 main + MGSM, XSum, CNN/DailyMail
  levels.json               # L0-L4 token budgets and notes
  metrics.json              # the 12 semantic measures
  models.json               # 17 models: 9 in the main panel + 8 size-sweep variants
```

The actual L0-L4 system prompts live in [`../prompts/`](../prompts/).
