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

All scripts here, numbered or standalone, accept `--results-root PATH` and
honour `$CAVEWOMAN_RESULTS`. Model tags must match the result directory names
(`gpt4o`, `qwen2.5`, `deepseek`, ...), not the display names used in the paper.

## Audit and verification scripts

Standalone, not part of the numbered pipeline:

| Script | Reproduces |
|---|---|
| `extraction_audit.py` | Per-cell answer-extraction-rate audit (paper appendix on extraction audit). |
| `l4_length_distribution.py` | L4 output-length distribution and budget-violation rate. |
| `price_vectors.py` | Projected per-item cost for the open-weight models under public-tier pricing. |

## Subpipelines

| Folder | Reproduces |
|---|---|
| [`llmlingua/`](llmlingua/) | LLMLingua-2 input-compression comparison (paper appendix on LLMLingua comparison). |
| [`judge/`](judge/) | LLM-judge runs and verdict extraction for NLI judge reliability calibration (paper appendix on judge reliability and semantic robustness). |
| [`floors/`](floors/) | Reference floors: sampling, register, and length-controlled. |
| [`stats/`](stats/) | Bootstrap CIs, Benjamini-Hochberg correction, and adjudicated per-item labels. |

Each subfolder is self-contained and only needed when reproducing that
part of the paper.
