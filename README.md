# CAVEWOMAN: How Large Language Models Behave Under Linguistic Input and Output Compression (EMNLP 2026)

![CAVEWOMAN](cavewoman_hero.png)

*Paper: [alphaxiv.org/pdf/2606.24083v1](https://www.alphaxiv.org/pdf/2606.24083v1) ([ACL Anthology](https://aclanthology.org/2026.emnlp-main.1235/), EMNLP 2026 Main Track) &nbsp;•&nbsp; Data: [rayascript/cavewoman-data](https://huggingface.co/datasets/rayascript/cavewoman-data) &nbsp;•&nbsp; [All figures](figures/)*

---

## TL;DR

**To cut inference cost, compress the output, not the input.**
[**Run the CAVEWOMAN framework**](#run-it-yourself) on your own task to find the
cheapest level that still gives you good answers.

![Input compression versus output compression](figures/tldr_comparison.png)

- **Compressing the response cuts cost 1.7x, up to 3x.**
- **Compressing the prompt raises cost up to 96% and gives worse answers at every level.**
- **Half of all answers stay correct but change their reasoning.**
- **Every model is different. You should test yours in the framework!**

The CAVEWOMAN framework strips text grammatically, at five levels:

- **L0**: the question as written
- **L1**: drop the small words (the, a, of, in, and)
- **L2**: keep only nouns, verbs, and numbers
- **L3**: keep only nouns and numbers
- **L4**: same as L3, cut to 15 tokens

![The five compression levels](figures/teaser.png)

---

## Run it yourself

> **Using a coding agent?** Hand it [`RUN_WITH_AGENT.md`](RUN_WITH_AGENT.md).
> Otherwise the commands are below.

**Our study found that two models from the same family can react very differently
to the same compression.** So there is no level we can tell you to use. Run the
CAVEWOMAN framework on your own model and your own task to find yours. A few dollars
and some questions are enough.

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm   # only needed for input compression
cp .env.example .env                      # add OPENAI_API_KEY / ANTHROPIC_API_KEY
```

**Step 1. Generate.** All five levels, output-compressed.

```bash
python experiments/run_experiment_api.py \
    --model gpt-4o --dataset gsm8k \
    --condition output --level all \
    --output_dir ./results/gpt4o_output/gsm8k
```

Add `--end_idx 300` to cap the item count while you are trying it out, and
`--dry_run` to check the wiring without spending anything.

**Step 2. Score accuracy.** This reports accuracy, extraction rate, and output
tokens. For dollar cost read the runner's own final summary, or run
`python analysis/token_economics.py --results-root ./results`.

```bash
python evaluation/evaluate_results.py --results_dir ./results/gpt4o_output/gsm8k
```

**Step 3. Score whether the answer changed.** This is the measure accuracy
misses. It needs the L0 run from step 1 in the same directory, plus
`sentence-transformers` and a one-time NLI model download.

```bash
python evaluation/add_entailment_scores.py \
    --results_dir ./results/gpt4o_output/gsm8k \
    --model_name gpt4o --dataset gsm8k --condition output
```

**Step 4. Read it.**

```bash
python analysis/descriptive_stats.py --results-root ./results
```

You get, per level: accuracy, dollars per question, and the share of answers
that changed. Pick the level where the saving is worth what you give up.

### On your own data

Add a loader to [`src/dataset_loader_multi.py`](src/dataset_loader_multi.py):

1. write `_load_yourtask(split, n, seed)` returning items with
   `idx, question_raw, question_formatted, answer_gt, answer_type, dataset_name`
   (`answer_type` is `numeric`, `boolean`, or `multiple_choice`)
2. register it in the `_LOADERS` dict
3. add the name to the `--dataset` choices in all three places that list them:
   `experiments/run_experiment.py`, `experiments/run_experiment_api.py`, and
   `analysis/judge/run_judge.py`
4. add a split entry in `_DATASET_CANONICAL_SPLIT` in
   `experiments/run_experiment_api.py`, or the runner raises `KeyError`
5. add it under `design.datasets` in `analysis/config.yaml`, or every analysis
   script reports "discovered 0 complete lanes" and exits without an error

[`RUN_WITH_AGENT.md`](RUN_WITH_AGENT.md) spells all five out, plus the exact
shape each `answer_type` expects.

Already have your own eval harness? The five prompts in
[`prompts/condition_B_output_constraint/`](prompts/condition_B_output_constraint/)
are plain Markdown. Drop them in and run your eval once per level.

> Measured on single-turn tasks with a short extractable answer. Long-form
> behaves differently. Cost assumes you call the API directly and are billed per
> token.

---

## What it does

![Framework](figures/framework.png)

One question, two ways to shrink it. **Input:** strip the small words out of the
question before the model sees it. **Output:** leave the question alone and tell
the model to answer in clipped language. Five levels each, L0 (untouched) to L4
(15-token limit), defined in
[`benchmark/configs/compression_levels.yaml`](benchmark/configs/compression_levels.yaml).

Every answer gets scored three ways: is it right, what did it cost, and does it
still say what the model said without compression.

---

## What we found

**1. Compressing the prompt costs you money.** You save 15 to 16 input tokens,
the model answers at greater length, and output tokens are priced 4 to 6 times
higher. Compressing the response is cheaper on 18 of 20 model and benchmark
pairs.

![Cost by channel](figures/cost_heatmap.png)

**2. Cheaper and correct is not the same as unchanged.** Split every answer by
correct or not against changed or not, and the biggest group on every
non-reasoning model is correct but changed. Cutting the original answer to the
same length makes it worse, not better, and all twelve similarity measures we
tried agree.

![Four outcomes](figures/four_outcomes.png)

**3. Which level works is specific to your setup.** Model, size, and language
all move it, and a model being strong to begin with does not predict where it
lands.

![Robustness](figures/robustness.png)

Full grid, 8 models x 5 benchmarks x 5 levels, both directions:
[accuracy_grid.png](figures/accuracy_grid.png).

---

## Scale

| | |
|---|---:|
| Model generations | **1,280,850** |
| API calls | **573,250** |
| Local-GPU time | **610 GPU-hours** |
| Models | **9** (4 open-weight + 5 API) |
| Questions per model, level, and direction | 11,465 |
| Datasets | 8 |
| Similarity measures | 12 |
| Accuracy checked by | **4 people + a second LLM judge** (90.9% agreement) |

Accuracy is judge-corrected, not raw regex: anything the extractor cannot parse
is read by an LLM judge, checked against a second judge and four people. See
[`analysis/stats/`](analysis/stats/).

---

## Repository layout

```
figures/        every figure from the paper, with captions
benchmark/      what the levels and the two directions are
prompts/        the L0-L4 system prompts
src/            prompts, dataset loaders, model loader, answer extraction, NLI
experiments/    generation runners (+ multilingual, longform, size_sweep)
evaluation/     accuracy, cost, embedding and NLI scoring
analysis/       scored JSONLs to the paper's tables and figures
results/        headline tables (CSV) and one sample JSONL per cell
jobs/           SLURM and shell runners used for the paper
docs/           project website
```

Order is always generate, score, analyze. Every stage is resume-safe, and
anything a run produces is gitignored.

Scripts read `<results_root>/<tag>_<condition>/<dataset>/caveman_<tag>_<dataset>_<condition>_<level>.jsonl`.
Tags are `gpt4o`, `gpt5.4`, `haiku`, `sonnet`, `qwen2.5`, `qwen3.5`, `deepseek`,
`gemma4`, `kimi`.

Always pass `--results-root PATH` to the `analysis/` scripts; they ignore
`$CAVEWOMAN_RESULTS`. The shipped generations live under `results/samples/`.

---

## Extensions

| Experiment | Code | Figure |
|---|---|---|
| Multilingual (MGSM, ten languages) | [`experiments/multilingual/`](experiments/multilingual/) | [ladder](figures/mgsm_ladder.png) |
| Long-form summarization (XSum, CNN/DM) | [`experiments/longform/`](experiments/longform/) | [token flows](figures/longform_tokens.png) |
| Size sweep (Qwen2.5, 0.5B-32B) | [`experiments/size_sweep/`](experiments/size_sweep/) | [sweep](figures/size_sweep.png) |
| LLMLingua-2 comparison | [`analysis/llmlingua/`](analysis/llmlingua/) | [comparison](figures/llmlingua_comparison.png) |
| Reference floors | [`analysis/floors/`](analysis/floors/) | [twelve measures](figures/alt_measures.png) |

> **Running a Qwen3-family model?** Pass `--disable_thinking`. The chat template
> emits a reasoning block unless thinking is explicitly `False`, and that
> reasoning eats the token budget before an answer is written.

---

## Cite

```bibtex
@inproceedings{adeyemi-etal-2026-cavewoman,
    title = "{CAVEWOMAN}: How Large Language Models Behave Under Linguistic Input and Output Compression",
    author = "Adeyemi, Morayo Danielle  and
      Rossi, Ryan A.  and
      Dernoncourt, Franck",
    editor = "Lee, Hung-Yi  and
      Dev, Sunipa  and
      Li, Jessy  and
      Elliott, Desmond",
    booktitle = "Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing",
    month = oct,
    year = "2026",
    address = "Budapest, Hungary",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.emnlp-main.1235/",
    pages = "28162--28188",
    ISBN = "979-8-89176-459-0",
}
```

## Contact

Morayo Adeyemi, morayo.danielle@gmail.com
