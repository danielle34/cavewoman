# analysis/judge/

LLM-judge calibration reported in the paper appendix on judge
reliability. Calibrates the bidirectional-NLI judge by running an
independent LLM judge on 100 POS-filtered synthetic positive pairs
per compression level, then back-computing the false-negative rate
that anchors every Finding 2 claim (L1 FN = 2.9%).

```
run_judge.py             # main judge runner; writes per-item verdicts
prompts.py               # the judge system prompts
re_extract_verdicts.py   # post-hoc verdict re-extraction from raw outputs
```

The headline calibration in the paper uses Qwen3.5 as the judge;
swap models by changing the runner's CLI flag.
