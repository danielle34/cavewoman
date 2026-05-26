# results/

What ships in this repository:

```
tables/   # headline CSVs reported in the paper (dissociation tables,
          # length-controlled NLI grand summary, master robustness table,
          # method comparison, llmlingua summary).
samples/  # one per-record JSONL per (model x condition x dataset)
          # at L1, so the schema is inspectable end-to-end.
```

Full per-record JSONLs across all 450 cells (`model x dataset x channel
x level`) are published on Hugging Face at
[rayascript/cavewoman-data](https://huggingface.co/datasets/rayascript/cavewoman-data).

Each per-record JSONL row carries the question, the gold answer, the
model output, the extracted answer, token counts, realized cost, and
the semantic-fidelity scores added by `evaluation/`.
