# Run CAVEWOMAN on my task

Hand this file to your coding agent (Claude Code, Cursor, Codex, whatever you
use). It contains everything the agent needs to run this benchmark on your model
and your data, and to tell you which compression level to ship.

---

## For the agent

You are helping someone find out how much they can shorten their LLM prompts or
responses before it costs them accuracy. You are working inside this repository.

### What this benchmark does

It takes one question and shortens it in one of two directions, at five levels:

| | |
|---|---|
| **Input compression** | Strip grammar words out of the question before the model sees it. The system prompt stays neutral. |
| **Output compression** | Leave the question alone. Tell the model to answer in stripped-down language. |

Levels: **L0** the text as written, **L1** drop the small words (the, a, of, in,
and), **L2** keep only nouns, verbs, and numbers, **L3** keep only nouns and
numbers, **L4** L3 cut to 15 tokens.

Every response is scored three ways. Accuracy, dollar cost, and whether the
response still says what the model said at L0. That third one is the reason this
benchmark exists: in our paper, half of all compressed answers came back correct
but no longer saying what the model would have said. Accuracy alone hides it.

### Ask the user these first

1. **Which model?** An API model (OpenAI, Anthropic, Azure) or a local
   HuggingFace model? API is far cheaper and faster to try.
2. **Which direction?** Output compression is the one that saves money. Run that
   first. Only run input compression if they specifically want to know about
   shortening prompts.
3. **What data?** One of the five built-in benchmarks (`gsm8k`, `boolq`,
   `arc_easy`, `commonsenseqa`, `mmlu_stem`), or their own task? If their own,
   you will add a loader, see below.
4. **How many questions?** A few hundred per level is enough. Tell them the
   rough API cost before running anything.
5. **Do they have API keys set?** They go in `.env` at the repo root.

### Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm   # only needed for input compression
cp .env.example .env                      # add OPENAI_API_KEY / ANTHROPIC_API_KEY
```

### Run it

Generate all five levels, output-compressed:

```bash
python experiments/run_experiment_api.py \
    --model gpt-4o --dataset gsm8k \
    --condition output --level all \
    --output_dir ./results/gpt4o_output/gsm8k
```

Use `--dry_run` first to confirm the wiring without spending anything. It writes
a fixed placeholder response, so accuracy will be near zero: a dry run proves the
loader, the file layout, and the cost plumbing, not answer extraction. To keep
the bill small, cap the item count: the API runner has no `--n`, so use
`--end_idx 300`.

For local HuggingFace models use `experiments/run_experiment.py` instead. Its
flags differ: the output directory is `--output` (not `--output_dir`) and it does
take `--n 300`. Pass `--disable_thinking` for any Qwen3-family model or the
reasoning block will eat the token budget before an answer is written.

Score accuracy. This reports accuracy, extraction rate, and output tokens, but
**not** dollar cost. For cost, read the runner's own final summary or run
`python analysis/token_economics.py --results-root ./results`:

```bash
python evaluation/evaluate_results.py --results_dir ./results/gpt4o_output/gsm8k
```

Score whether the answer changed. This compares each compressed response against
that model's own L0 response, so the L0 run must exist in the same
`--results_dir` (running `--level all` above gives you that). It needs
`sentence-transformers` installed and downloads an NLI model of about 700MB on
first use. It pulls in torch, so on a CPU-only or CUDA-mismatched machine the
import fails before any scoring happens. Check with
`python -c "from sentence_transformers import CrossEncoder"` before starting.
A GPU is optional, it falls back to CPU:

```bash
python evaluation/add_entailment_scores.py \
    --results_dir ./results/gpt4o_output/gsm8k \
    --model_name gpt4o --dataset gsm8k --condition output
```

Aggregate:

```bash
python analysis/descriptive_stats.py --results-root ./results
```

### Using the user's own data

Add a loader to `src/dataset_loader_multi.py`:

1. Write `_load_<name>(split, n, seed)` returning a list of dicts with keys
   `idx`, `question_raw`, `question_formatted`, `answer_gt`, `answer_type`,
   `dataset_name`.
2. Register the function in the `_LOADERS` dict in the same file.
3. Add the name to the `--dataset` choices in all three argparse sites:
   `experiments/run_experiment.py`, `experiments/run_experiment_api.py`, and
   `analysis/judge/run_judge.py`.
4. Add a split entry in the `_DATASET_CANONICAL_SPLIT` dict in
   `experiments/run_experiment_api.py`, for example `"mytask": "test"`. That
   runner does a bare dict lookup and raises `KeyError` without it. The local
   runner falls back to `--split`, so it is optional there.
5. Add it under `design.datasets` in `analysis/config.yaml` as
   `<name>: {n_expected: <N>, task_type: <label>}`. The analysis pipeline
   enumerates datasets from that file, not from disk, so without this entry every
   analysis script reports "discovered 0 complete lanes" and exits without
   raising an error.

**If the model tag is not one of the nine built-in ones**, also add it under
`design.models` in `analysis/config.yaml`, or no lane is discovered, and under
`pricing:`, or `analysis/token_economics.py` prices it as free and the cost
numbers come out silently at zero.

**What each `answer_type` expects:**

| `answer_type` | `answer_gt` | `question_formatted` |
|---|---|---|
| `numeric` | digit string, e.g. `"168"` | the question text |
| `boolean` | lowercase `"yes"` or `"no"`, not `True` / `False` | the question text |
| `multiple_choice` | the bare uppercase letter, e.g. `"B"`, not the choice text | `Question: <q>`, then one `"<LETTER>: <text>"` line per choice; use the module's `_format_mc` helper |

`idx` must be a unique integer. It is the join key for `--start_idx` and
`--end_idx`, and for pairing each level against L0 in the entailment step.

If their task has no single extractable answer, accuracy will not work. In that
case run only the cost and agreement measures and say so.

### File layout the scripts expect

```
<results_root>/<tag>_<condition>/<dataset>/caveman_<tag>_<dataset>_<condition>_<level>.jsonl
```

`<tag>` is the short model name used in directory and file names: `gpt4o`,
`gpt5.4`, `haiku`, `sonnet`, `qwen2.5`, `qwen3.5`, `deepseek`, `gemma4`, `kimi`.

Always pass `--results-root PATH` explicitly to the `analysis/` scripts. They do
not read `$CAVEWOMAN_RESULTS`, so setting that variable alone makes them silently
aggregate the default `results/` tree instead of yours. The `evaluation/` scripts
use `--results_dir` with an underscore and point at a single
`<tag>_<condition>/<dataset>` cell. The generations shipped with this repo are
under `results/samples/`, not `results/`.

### What to tell the user at the end

Report, per level:

- **Accuracy**, against L0
- **Cost per question** in dollars, and the multiple against L0
- **Share of answers that changed**, against L0

Then make a recommendation: the strictest level where accuracy is within their
tolerance and the share of changed answers is acceptable for how they use the
output. Spell out the tradeoff rather than just naming a level. If they only
ever read the final answer and never the reasoning, a high changed-answer rate
may be fine. If the reasoning is shown to users, logged, or audited, it is not.

Flag these honestly if they come up:

- If the model's L0 responses are already hitting the token limit, the cost
  saving will look smaller than it is. Check `analysis/` for the token-limit audit.
- Reasoning models that spend the token budget on a hidden or visible reasoning
  trace behave differently from the rest and their numbers are not comparable.
- Cost figures assume direct per-token API billing. Through a subscription or a
  product wrapper the math does not carry.

---

## Context

Paper: [aclanthology.org/2026.emnlp-main.1235](https://aclanthology.org/2026.emnlp-main.1235/) (EMNLP 2026)
Data: [rayascript/cavewoman-data](https://huggingface.co/datasets/rayascript/cavewoman-data)
Figures and findings: [`figures/`](figures/) and [`results/`](results/)
