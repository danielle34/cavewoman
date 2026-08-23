# results/

What ships in this repository:

```
tables/   # headline CSVs reported in the paper (dissociation tables,
          # length-controlled NLI grand summary, master robustness table,
          # method comparison, llmlingua summary).
samples/  # one per-record JSONL per (model x condition x dataset)
          # at L1, so the schema is inspectable end-to-end.
```

The full per-record JSONLs are published on Hugging Face at
[rayascript/cavewoman-data](https://huggingface.co/datasets/rayascript/cavewoman-data):
**450 cells** (`model x dataset x channel x level`) for the main panel, plus
the multilingual, long-form, and size-sweep runs, **1,280,850 generations** in
total.

Each row carries the question, the gold answer, the model output, the extracted
answer, token counts, realized cost, and the semantic-fidelity scores added by
`evaluation/`.

Anything you generate yourself lands under `results/<tag>_<condition>/` and is
gitignored.
