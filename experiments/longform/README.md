# Long-form summarization

The register ladder where the answer *is* the text, so entailment is measured on
the generation itself. XSum and CNN/DailyMail. Condition B only, since the
article is the input.

```bash
python run_summarisation.py --model_name qwen2.5 \
    --model_path ~/models/Qwen2.5-VL-7B \
    --dataset xsum --levels L0,L1,L2,L3,L4 --n 200
```

| Flag | |
|---|---|
| `--temperature 0.7 --suffix _paired` | second draw for the sampling floor |
| `--disable_thinking` | required for Qwen3-family models |
| `--dataset cnndm` | CNN/DailyMail instead of XSum |

Budgets are `{L0:200, L1:150, L2:100, L3:75, L4:20}`. Articles truncate at 6000
characters, flagged per row as `doc_truncated`.

Two things to know when reading the output: length is not monotone in the level
(L1 is shortest, L2 and L3 run longer, only L4 compresses reliably), and these
numbers need the long-form floor from [`../../analysis/floors/`](../../analysis/floors/),
not the short-answer one.
