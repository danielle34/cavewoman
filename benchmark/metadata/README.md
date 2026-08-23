# benchmark/metadata/

Structured metadata for everything the benchmark fixes, channels,
datasets, levels, metrics, models. Machine-readable so other tooling
can validate runs against the canonical configuration.

Reference data, not read by the pipeline at runtime. The analysis
scripts take their own lists from `analysis/config.yaml`; these files
are the canonical record to validate against.

`tag` is the result-tree identifier: it is the directory name under
`results/` and appears in every generated filename. `display_name` is the
human-facing form used in the paper. Match on `tag`, not `display_name`.
