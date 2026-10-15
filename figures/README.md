# figures/

Every figure from the paper, as PNG. Each one is linked from the folder whose
code produces it.

## The summary

| | |
|---|---|
| ![](tldr_comparison.png) | **`tldr_comparison.png`**: the whole result in one table. Input compression raises cost and lowers accuracy; output compression lowers cost and holds accuracy, but half the answers change. |

## How it works

| | |
|---|---|
| ![](framework.png) | **`framework.png`**: the two channels. Input compression filters the question before the model sees it; output compression leaves the question alone and constrains the response. Five levels, three measures. |
| ![](teaser.png) | **`teaser.png`**: one question pushed through all five levels of the real input filter. |
| ![](input_filter_example.png) | **`input_filter_example.png`**: the part-of-speech filter on a GSM8K question, level by level, with the surviving token count. |

Defined in [`../benchmark/`](../benchmark/), prompts in [`../prompts/`](../prompts/).

## Finding 1: the two directions do not cost the same

| | |
|---|---|
| ![](cost_heatmap.png) | **`cost_heatmap.png`**: change in the dollar cost of a question against L0, by benchmark and by model, on both channels. Red is more expensive, green is cheaper. |
| ![](cost_mechanism.png) | **`cost_mechanism.png`**: why. Prompt tokens on the left, response tokens on the right. Compressing the prompt shortens the prompt but the response does not follow it down. |
| ![](api_spend.png) | **`api_spend.png`**: what the four API models cost to evaluate across the whole paper, by experiment. $852 in total. |

## Finding 2: cheaper and correct is not the same as unchanged

| | |
|---|---|
| ![](four_outcomes.png) | **`four_outcomes.png`**: every answer split by correct or not, against whether it still says what the model said without compression. The biggest group on every non-reasoning model is *correct but changed*. |
| ![](alt_measures.png) | **`alt_measures.png`**: the same gap under all twelve semantic measures, so it is not an artifact of one NLI judge. |
| ![](l0neutral_ladder.png) | **`l0neutral_ladder.png`**: scored against both unconstrained references, with and without the step-by-step instruction. Same pattern either way. |

## Finding 3: the right level is model-specific

| | |
|---|---|
| ![](robustness.png) | **`robustness.png`**: accuracy change under the telegraphic constraint, by model, by model size, and by language. The spread is the reason to run this yourself. |
| ![](accuracy_grid.png) | **`accuracy_grid.png`**: the full grid: 8 models x 5 benchmarks x 5 levels x both channels. |
| ![](cap_hit.png) | **`cap_hit.png`**: how often each model hit the token limit rather than stopping on its own. Needed to read the reasoning models correctly. |

## Extensions

| | |
|---|---|
| ![](mgsm_ladder.png) | **`mgsm_ladder.png`**: MGSM in ten languages, ordered by how many tokens a question costs relative to English. |
| ![](size_sweep.png) | **`size_sweep.png`**: Qwen2.5 from 0.5B to 32B on the same questions. |
| ![](longform_tokens.png) | **`longform_tokens.png`**: token flows when summaries are forced into clipped style, on XSum and CNN/DailyMail. |
| ![](llmlingua_comparison.png) | **`llmlingua_comparison.png`**: LLMLingua-2 as a learned input-compression baseline, against our filter. |
| ![](kimi_accuracy.png) | **`kimi_accuracy.png`**: Kimi-K2.6 on raw extraction. Bars at zero are cells where nearly every visible response was empty. |

## Regenerating

These are rendered from the paper's vector sources. The analysis scripts in
[`../analysis/`](../analysis/) regenerate the underlying numbers from the
released per-record JSONLs.
