# Size sweep and thinking-mode contrast

Scale ladder on the same 500 seeded items per benchmark, so cells are
item-matched across sizes.

| File | |
|---|---|
| `sweep_job.sh` | SLURM body for one (model, condition) |
| `score_sweep_job.sh` | entailment + embedding sidecars |
| `aggregate_family.py` | builds the size-sweep table |
| `thinking_contrast.py` | Qwen3.5 reasoning on vs off |

```bash
export CAVEWOMAN_ENV_SETUP='conda activate myenv'   # optional
export SWEEP_TAG=qwen2.5-7b SWEEP_PATH=~/models/Qwen2.5-7B-Instruct
export SWEEP_COND=output SWEEP_LEVELS="L0 L1 L2 L3 L4"
sbatch sweep_job.sh
```

Also: `SWEEP_N` (`full` for the whole split), `SWEEP_SEED`, `SWEEP_DATASETS`,
`SWEEP_EXTRA` (extra flags for `../run_experiment.py`).

Sweep Qwen3.5 with `SWEEP_EXTRA=--disable_thinking`. With thinking on, the
reasoning block saturates the budget at every level and the levels stop being
distinguishable.

Reading the results: robustness is non-monotone below 3B. The 0.5B and 1.5B
models look *more* robust because they start far lower unconstrained, so a ratio
against their own L0 has less room to fall.
