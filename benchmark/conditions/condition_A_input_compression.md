# Condition A, Input compression

User question filtered through spaCy POS rules at `Lₖ` before sending.
System prompt is the neutral string ("You are a helpful assistant…"),
constant across levels.

## POS rules (en_core_web_sm)

| Level | Rule |
|---|---|
| L0 | unchanged |
| L1 | drop `{DT, IN, CC, RP, TO, MD}` |
| L2 | keep `{NN*, VB*, CD}` |
| L3 | keep `{NN*, CD}` |
| L4 | first 15 whitespace tokens |

## Worked example

Q: *"If a train leaves Boston at 8am going 60mph, when does it reach New York?"*

| Level | Compressed |
|---|---|
| L0 | If a train leaves Boston at 8am going 60mph, when does it reach New York? |
| L1 | train leaves Boston 8am going 60mph, when does reach New York? |
| L2 | train leaves Boston going does reach New York |
| L3 | train Boston York |
| L4 | If a train leaves Boston at 8am going 60mph, when does it |

## Empty-result fallback

If filtering yields an empty string, original is kept with
`empty_after_filter = true` in the record.

## Reference

`src/dataset_loader_multi.py::compress_input(text, level)`. Demo:
[`scripts/pos_filter_demo.py`](../../scripts/pos_filter_demo.py).
