# analysis/

Statistical analysis and figure generation that consumes the augmented
JSONLs from `evaluation/` and produces every table and figure in the
paper. Each script is standalone and reads from a common
[`config.yaml`](config.yaml); run them in the order below or invoke
`run_all_analysis.py` to chain them all.

| Order | Script | Purpose |
|---:|---|---|
| 1 | `validate_results.py` | Sanity-check the input tree (file presence, schema, expected per-cell row counts). |
| 2 | `descriptive_stats.py` | Per-cell aggregates with 10,000-sample bootstrap 95% CIs. |
| 3 | `collapse_thresholds.py` | Semantic / accuracy collapse detection and length-controlled NLI re-scoring. |
| 4 | `hypothesis_tests.py` | Paired Wilcoxon signed-rank with Benjamini-Hochberg FDR correction. |
| 5 | `token_economics.py` | Realized per-item cost and the open-weight cost projection. |
| 6 | `metric_validation.py` | Twelve-measure cross-metric robustness replication. |

Install pipeline-specific dependencies with `pip install -r requirements.txt`.

## Audit and verification scripts

Standalone, not part of the numbered pipeline:

| Script | Reproduces |
|---|---|
| `extraction_audit.py` | Per-cell answer-extraction-rate audit (paper appendix on extraction audit). |
| `l4_length_distribution.py` | L4 output-length distribution and budget-violation rate. |
| `verify_paper_numbers.py` | Spot-checks every numeric claim in the paper against the released per-cell summary CSVs. |

## Subpipelines

| Folder | Reproduces |
|---|---|
| [`llmlingua/`](llmlingua/) | LLMLingua-2 input-compression comparison (paper appendix on LLMLingua comparison). |
| [`judge/`](judge/) | LLM-judge runs and verdict extraction for NLI judge reliability calibration (paper appendix on judge reliability and semantic robustness). |

Both subpipelines are self-contained and only used when reproducing
those appendix results.
