# Statistics

Accuracy in the paper is **adjudicated**, not raw regex. Any generation the
extractor cannot parse is ruled on by an LLM judge rather than scored wrong,
because extraction failures correlate with the condition and would otherwise
be counted as compression damage.

That judge is validated two ways:

- **LLM-judge validation.** An independent second judge re-rules the same
  items; the two agree on 90.9% of binary correct/not decisions. The
  bidirectional-NLI judge is separately calibrated on synthetic positive pairs
  in [`../judge/`](../judge/).
- **Human validation.** Four human labelers check a sample of the judge's
  `CORRECT` verdicts. The residual disagreement (an 11.7% false-recovery rate)
  is reported in the paper as a sensitivity bound, not folded into the numbers.

83,147 judgements cover all 80 model x dataset x condition cells.

| File | Purpose |
|---|---|
| `adjudicated_labels_and_stats.py` | Joins per-item generations with judge verdicts into `adj_correct = regex_correct OR judge == CORRECT`, then reruns paired Wilcoxon with BH correction on the adjudicated labels. |
| `bootstrap_cis.py` | 95% confidence intervals on per-cell rates, 10,000 resamples at a fixed seed. |
| `fdr_correction.py` | Benjamini-Hochberg correction across the Wilcoxon test family. |

Run `adjudicated_labels_and_stats.py` first: it writes the labels to
`results/revision/adjudicated/` that the other two read.

Items never judged stay incorrect, which is conservative.

The human-audit worksheets are not redistributed, so the labeling step itself
is described in the paper rather than scripted here.
