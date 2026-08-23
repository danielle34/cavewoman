# benchmark/

The benchmark definition: what the five compression levels are, what
the two channels are, and the metadata that pins down which datasets,
models, and metrics are in scope. Every entry carries a `panel` field
saying whether it belongs to the main run, the size sweep, or one of
the extension panels.

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
