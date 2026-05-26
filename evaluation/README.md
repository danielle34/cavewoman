# evaluation/

Three scoring passes that turn raw generations into the numbers reported
in the paper. Each script reads the per-record JSONLs produced by the
`experiments/` runners and writes augmented JSONLs / summary JSONs
alongside them.

| Script | Adds |
|---|---|
| `evaluate_results.py` | Strict and relaxed answer-extraction accuracy per level, plus a `accuracy_summary.json` per (model, dataset, condition). |
| `add_embedding_scores.py` | Sentence-embedding cosine similarity between each Lx response and the same item's L0 response. |
| `add_entailment_scores.py` | Bidirectional NLI entailment (Lx vs L0) using a local cross-encoder. |

All three import shared utilities from [`../src/`](../src/). The NLI and
embedding scorers will download their respective models (a few hundred
MB total) on first run.
