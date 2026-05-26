# benchmark/

The benchmark definition: what the five compression levels are, what
the two channels are, and the metadata that pins down which datasets,
models, and metrics are in scope.

```
configs/
  compression_levels.yaml   # L0-L4 specifications shared by both channels
  example_run_config.json   # one cell's run configuration as a worked example
conditions/
  condition_A_input_compression.md   # input-compression channel definition
  condition_B_output_constraint.md   # output-constraint channel definition
metadata/
  conditions.json           # structured channel metadata
  datasets.json             # five datasets, schemas, splits
  levels.json               # L0-L4 token budgets and notes
  metrics.json              # full metric catalog
  models.json               # nine models run (8 in headline + Kimi-K2.6 in appendix)
```

The actual L0-L4 system prompts live in [`../prompts/`](../prompts/).
