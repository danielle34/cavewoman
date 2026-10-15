# results/

The headline numbers, as figures first and CSVs underneath.

Full per-record generations are on Hugging Face at
[rayascript/cavewoman-data](https://huggingface.co/datasets/rayascript/cavewoman-data):
**450 cells** (`model x dataset x channel x level`) for the main panel, plus the
multilingual, long-form, and size-sweep runs, **1,280,850 generations** total.

---

## 1. Compressing the prompt costs you money

![Cost by channel](../figures/cost_heatmap.png)

Change in the dollar cost of a question against the uncompressed baseline. Left
pair is input compression, right pair is output compression. Red is more
expensive.

Constraining the response is cheaper on **18 of 20** model x benchmark pairs,
by **1.7x** on average and up to **3x**. Compressing the prompt raises net cost
on **3 of the 4** API models, by up to **+96%**, because the saved input tokens
are smaller than the extra output tokens they cause.

## 2. Correct is not the same as unchanged

![Four outcomes](../figures/four_outcomes.png)

Every answer split four ways: correct or not, against whether it still says
what that model said without compression. On every non-reasoning model the
biggest group is **correct but changed** (50.1% of all responses).

Numbers behind it: [`tables/caveman_dissociation_table.csv`](tables/caveman_dissociation_table.csv)
gives per-cell accuracy, NLI rate, and the C1-C4 split.

## 3. Which level works depends on your setup

![Accuracy grid](../figures/accuracy_grid.png)

Eight models x five benchmarks x five levels, both channels. Blue is input
compression, orange is output compression, darker is stricter.

![Robustness](../figures/robustness.png)

Accuracy change under the telegraphic constraint by model, by size, and by
language. The spread here is why the recommendation is to run the ladder on
your own task rather than copy a level out of this table.

---

## What ships in this folder

```
tables/   headline CSVs reported in the paper
samples/  one per-record JSONL per (model x condition x dataset) at L1,
          so the schema is inspectable end to end
```

| Table | What it holds |
|---|---|
| [`caveman_dissociation_table.csv`](tables/caveman_dissociation_table.csv) | Per cell: accuracy, bidirectional NLI rate, cosine, and the C1-C4 split. The source of Finding 2. |
| [`method_comparison_table.csv`](tables/method_comparison_table.csv) | The same per-cell quantities with our filter and LLMLingua-2 side by side. |
| [`llmlingua_dissociation_table.csv`](tables/llmlingua_dissociation_table.csv) | LLMLingua-2 input compression at tau=0.5 and tau=0.8. |
| [`llmlingua_summary_for_paper.csv`](tables/llmlingua_summary_for_paper.csv) | Four-row summary: mean accuracy, NLI, C2 share, and cosine per method. |
| [`length_controlled_nli_grand.csv`](tables/length_controlled_nli_grand.csv) | The length control. Cutting the uncompressed answer to the compressed answer's length makes the mismatch worse, not better. |
| [`length_controlled_nli_per_cell.csv`](tables/length_controlled_nli_per_cell.csv) | The same control broken out per model and dataset. |
| [`master_robustness_table.csv`](tables/master_robustness_table.csv) | All twelve semantic measures per channel and level. The source of [`alt_measures.png`](../figures/alt_measures.png). |

Each released record carries the question, the gold answer, the model output,
the extracted answer, token counts, cost, and the similarity scores added by [`../evaluation/`](../evaluation/).

Anything you generate yourself lands under `results/<tag>_<condition>/` and is
gitignored.

---

## Reproducing these

```bash
# from scored JSONLs to the tables above
python ../analysis/descriptive_stats.py --results-root /path/to/results
python ../analysis/collapse_thresholds.py --results-root /path/to/results
python ../analysis/hypothesis_tests.py --results-root /path/to/results
```

See [`../analysis/`](../analysis/) for the full ordered pipeline.
