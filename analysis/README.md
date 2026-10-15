# analysis/

Statistical analysis and figure generation that consumes the augmented
JSONLs from `evaluation/` and produces every table and figure in the
paper. Each script is standalone and reads from a common
[`config.yaml`](config.yaml); run them in the order below or invoke
`run_all_analysis.py` to chain them all.

What this stage produces, end to end:

![Four outcomes](../figures/four_outcomes.png)

The 2x2 split of every answer, correct or not against still saying the same
thing or not, is the output of `collapse_thresholds.py` and the per-item labels
in [`stats/`](stats/).

![Twelve measures](../figures/alt_measures.png)

`metric_validation.py` re-scores the same responses under twelve semantic
measures. The gap survives all of them, which is what rules out a single-judge
artifact.

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

| Folder | Reproduces | Figure |
|---|---|---|
| [`llmlingua/`](llmlingua/) | LLMLingua-2 input-compression comparison (paper appendix on LLMLingua comparison). | [comparison](../figures/llmlingua_comparison.png) |
| [`judge/`](judge/) | LLM-judge runs and verdict extraction for NLI judge reliability calibration (paper appendix on judge reliability and semantic robustness). | none |
| [`floors/`](floors/) | Reference floors: sampling, register, and length-controlled. | [both references](../figures/l0neutral_ladder.png) |
| [`stats/`](stats/) | Bootstrap CIs, Benjamini-Hochberg correction, and judge-corrected per-item labels. | [accuracy grid](../figures/accuracy_grid.png) |

Each subfolder is self-contained and only needed when reproducing that
part of the paper.
