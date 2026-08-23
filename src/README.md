# src/

Shared modules used by everything else in the repo. These are imported by
`experiments/`, `evaluation/`, and `analysis/` via a `sys.path` bootstrap
in each runner.

| Module | What it does |
|---|---|
| `constraint_prompts.py` | Canonical L0-L4 system prompts and the per-level output-token caps. |
| `dataset_loader_multi.py` | Loads all five datasets, extracts answers, checks correctness, and applies the spaCy POS input compression at each level. |
| `model_loader.py` | Local-model loader (HuggingFace `transformers`) and a text-only inference helper. |
| `metrics_utils.py` | Answer extraction (strict and relaxed), semantic-unit counting, per-level summarization. |
| `entailment_scorer.py` | Cross-encoder NLI scoring for bidirectional entailment between Lx and L0 outputs. |
| `release_paths.py` | Resolves the results tree from `--results-root`, `$CAVEWOMAN_RESULTS`, or the repo default. |
