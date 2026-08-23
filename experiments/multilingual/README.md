# Multilingual (MGSM)

The same L0-L4 ladder outside English. Ten non-English languages plus English
(`bn de es fr ja ru sw te th zh` + `en`).

| File | |
|---|---|
| `prompts_mgsm.py` | per-language prompts and token budgets |
| `run_mgsm_pilot.py` | local models |
| `run_mgsm_api.py` | OpenAI / Anthropic |
| `score_mgsm_entailment.py` | bidirectional NLI |

```bash
python run_mgsm_pilot.py --model_name qwen2.5 --model_path ~/models/Qwen2.5-VL-7B \
    --langs de,zh,bn,sw --levels L0,L1,L2,L3,L4 --n 250

python score_mgsm_entailment.py --judge xnli
```

Two judges: `--judge en` (monolingual, for English-target comparisons) and
`--judge xnli` (multilingual, default). The English judge is not valid on
in-language pairs, so report which one a number came from.

Add `--disable_thinking` for Qwen3-family models.
